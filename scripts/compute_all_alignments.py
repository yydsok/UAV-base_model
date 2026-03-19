#!/usr/bin/env python3
"""
Compute RGB-IR alignment offsets for ALL paired datasets.

Methods per dataset:
  - UTUAV: bbox center difference (per-sequence) — already computed, merge existing
  - DroneVehicle: skip (inherently aligned, 0px offset)
  - msuav-RGBMS: MI per-image (verified effective, NCC +0.1)
  - LLVIP, DroneRGBT, RGBT-Tiny, VT-Tiny-MOT, M3OT, TarDAL-M3FD, aerial-rgbt:
    MI per-image with CLAHE preprocessing

For datasets with negative NCC (IR inverts brightness), MI still captures
statistical dependence even when linear correlation fails. We compute MI
offsets for all paired datasets and validate with a quality filter.

Output: alignment_offsets_v2.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# MI alignment (works for cross-modal even with brightness inversion)
# ---------------------------------------------------------------------------

def _negative_mutual_info(params, rgb_gray, ir_gray, w, h):
    """Negative mutual information for optimization."""
    dx, dy = params
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    ir_shifted = cv2.warpAffine(ir_gray, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)
    hist2d = np.histogram2d(
        rgb_gray.ravel(), ir_shifted.ravel(), bins=32,
        range=[[0, 256], [0, 256]]
    )[0]
    hist2d = hist2d + 1e-10
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = np.outer(px, py)
    mi = np.sum(pxy * np.log(pxy / (px_py + 1e-10)))
    return -mi


def compute_mi_offset(rgb_path, ir_path, search_range=80, target_size=256):
    """Compute alignment offset for one image pair using mutual information."""
    rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        return None

    if rgb.shape != ir.shape:
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

    h_orig, w_orig = rgb.shape

    # Downscale for speed
    scale = min(target_size / h_orig, target_size / w_orig, 1.0)
    if scale < 1.0:
        rgb_s = cv2.resize(rgb, None, fx=scale, fy=scale)
        ir_s = cv2.resize(ir, None, fx=scale, fy=scale)
    else:
        rgb_s = rgb
        ir_s = ir
        scale = 1.0

    # CLAHE to reduce cross-modal appearance gap
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rgb_eq = clahe.apply(rgb_s)
    ir_eq = clahe.apply(ir_s)

    h, w = rgb_eq.shape
    sr = search_range * scale

    result = minimize(
        _negative_mutual_info,
        x0=[0.0, 0.0],
        args=(rgb_eq, ir_eq, w, h),
        method='Powell',
        bounds=[(-sr, sr), (-sr, sr)],
        options={'maxiter': 200, 'ftol': 1e-6}
    )

    if result.success or result.fun < -0.01:
        dx = result.x[0] / scale
        dy = result.x[1] / scale
        return (float(dx), float(dy))
    return None


def _mi_worker(args):
    """Worker for parallel MI computation."""
    ir_path, rgb_path, search_range = args
    try:
        result = compute_mi_offset(rgb_path, ir_path, search_range=search_range)
    except Exception:
        result = None
    return ir_path, result


# ---------------------------------------------------------------------------
# UTUAV bbox alignment (reuse existing)
# ---------------------------------------------------------------------------

def compute_utuav_sequence_offset(seq_dir):
    """Compute median (dx, dy) offset for one UTUAV sequence from bbox annotations."""
    rgb_txt = os.path.join(seq_dir, 'rgb.txt')
    ir_txt = os.path.join(seq_dir, 'ir.txt')
    if not os.path.exists(rgb_txt) or not os.path.exists(ir_txt):
        return None
    try:
        rgb_b = np.loadtxt(rgb_txt, delimiter=' ')
        ir_b = np.loadtxt(ir_txt, delimiter=' ')
    except Exception:
        return None
    if rgb_b.ndim == 1:
        rgb_b = rgb_b.reshape(1, -1)
        ir_b = ir_b.reshape(1, -1)
    if rgb_b.shape != ir_b.shape or len(rgb_b) < 3:
        return None
    dx_arr = (ir_b[:, 0] + ir_b[:, 2] / 2) - (rgb_b[:, 0] + rgb_b[:, 2] / 2)
    dy_arr = (ir_b[:, 1] + ir_b[:, 3] / 2) - (rgb_b[:, 1] + rgb_b[:, 3] / 2)
    q1x, q3x = np.percentile(dx_arr, [25, 75])
    q1y, q3y = np.percentile(dy_arr, [25, 75])
    iqr_x, iqr_y = q3x - q1x, q3y - q1y
    mask = (
        (dx_arr >= q1x - 1.5 * iqr_x) & (dx_arr <= q3x + 1.5 * iqr_x) &
        (dy_arr >= q1y - 1.5 * iqr_y) & (dy_arr <= q3y + 1.5 * iqr_y)
    )
    if mask.sum() < 3:
        dx = float(np.median(dx_arr))
        dy = float(np.median(dy_arr))
    else:
        dx = float(np.median(dx_arr[mask]))
        dy = float(np.median(dy_arr[mask]))
    return (dx, dy)


def compute_utuav_offsets(utuav_root):
    """Compute per-sequence offsets for UTUAV."""
    sequence_offsets = {}
    if not os.path.isdir(utuav_root):
        print(f"  WARNING: UTUAV root not found: {utuav_root}")
        return sequence_offsets
    seqs = sorted(os.listdir(utuav_root))
    for seq in seqs:
        seq_dir = os.path.join(utuav_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        result = compute_utuav_sequence_offset(seq_dir)
        if result is not None:
            sequence_offsets[f"UTUAV/{seq}"] = list(result)
    return sequence_offsets


# ---------------------------------------------------------------------------
# Sequence-based MI: compute per-sequence global offset from sampled frames
# ---------------------------------------------------------------------------

def compute_sequence_mi_offset(samples_in_seq, n_sample=10, search_range=60):
    """Compute a single offset for a sequence by sampling frames and taking median."""
    indices = np.linspace(0, len(samples_in_seq) - 1,
                          min(n_sample, len(samples_in_seq)), dtype=int)
    dxs, dys = [], []
    for idx in indices:
        s = samples_in_seq[idx]
        rgb_p = s.get('rgb_path')
        ir_p = s.get('ir_path')
        if not rgb_p or not ir_p:
            continue
        result = compute_mi_offset(rgb_p, ir_p, search_range=search_range)
        if result is not None:
            dxs.append(result[0])
            dys.append(result[1])
    if len(dxs) >= 2:
        return (float(np.median(dxs)), float(np.median(dys)))
    elif len(dxs) == 1:
        return (dxs[0], dys[0])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute RGB-IR alignment offsets for ALL paired datasets")
    parser.add_argument('--manifest',
                        default='/root/autodl-tmp/train1/manifests/'
                                'full_pretrain_v5_cleaned.json')
    parser.add_argument('--existing_offsets',
                        default='/root/autodl-tmp/train1/manifests/'
                                'alignment_offsets.json',
                        help='Existing offsets to merge (UTUAV bbox)')
    parser.add_argument('--utuav_root',
                        default='/root/autodl-tmp/data/UTUAV')
    parser.add_argument('--output',
                        default='/root/autodl-tmp/train1/manifests/'
                                'alignment_offsets_v2.json')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--recompute_utuav', action='store_true',
                        help='Recompute UTUAV offsets (default: merge existing)')
    args = parser.parse_args()

    print("Loading manifest...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    all_samples = manifest.get('image_samples', manifest.get('samples', []))
    paired = [s for s in all_samples if s.get('type') == 'paired']
    by_ds = defaultdict(list)
    for s in paired:
        by_ds[s.get('dataset', 'unknown')].append(s)

    print(f"Paired datasets: {', '.join(f'{k}({len(v)})' for k, v in sorted(by_ds.items()))}")

    output = {
        'version': '2.0',
        'method_log': {},
        'sequence_offsets': {},
        'per_image_offsets': {},
    }

    # ---- 1. UTUAV: bbox method ----
    if 'UTUAV' in by_ds:
        if not args.recompute_utuav and os.path.exists(args.existing_offsets):
            print("\n=== UTUAV: merging existing bbox offsets ===")
            with open(args.existing_offsets) as f:
                existing = json.load(f)
            seq_offsets = {k: v for k, v in existing.get('sequence_offsets', {}).items()
                          if k.startswith('UTUAV/')}
            output['sequence_offsets'].update(seq_offsets)
            output['method_log']['UTUAV'] = 'bbox_center_diff (merged)'
            print(f"  Merged {len(seq_offsets)} sequence offsets")
        else:
            print("\n=== UTUAV: computing bbox offsets ===")
            t0 = time.time()
            seq_offsets = compute_utuav_offsets(args.utuav_root)
            output['sequence_offsets'].update(seq_offsets)
            output['method_log']['UTUAV'] = 'bbox_center_diff'
            print(f"  {len(seq_offsets)} sequences in {time.time()-t0:.1f}s")

    # ---- 2. DroneVehicle: skip (inherently aligned) ----
    if 'DroneVehicle' in by_ds:
        print("\n=== DroneVehicle: SKIP (inherently aligned, 0px offset) ===")
        output['method_log']['DroneVehicle'] = 'skip_inherently_aligned'

    # ---- 3. Sequence-based datasets: per-sequence MI ----
    seq_datasets = {
        'RGBT-Tiny': 60,
        'VT-Tiny-MOT': 60,
        'M3OT': 60,
    }
    for ds_name, search_range in seq_datasets.items():
        if ds_name not in by_ds:
            continue
        samples = by_ds[ds_name]
        # Group by sequence
        by_seq = defaultdict(list)
        for s in samples:
            seq = s.get('sequence', 'default')
            by_seq[seq].append(s)

        print(f"\n=== {ds_name}: per-sequence MI ({len(by_seq)} sequences, "
              f"{len(samples)} images) ===")
        t0 = time.time()
        n_computed = 0
        for seq_name, seq_samples in sorted(by_seq.items()):
            offset = compute_sequence_mi_offset(
                seq_samples, n_sample=10, search_range=search_range)
            if offset is not None:
                key = f"{ds_name}/{seq_name}"
                output['sequence_offsets'][key] = list(offset)
                n_computed += 1

        output['method_log'][ds_name] = 'mutual_info_per_sequence'
        elapsed = time.time() - t0
        print(f"  {n_computed}/{len(by_seq)} sequences computed in {elapsed:.1f}s")

        if n_computed > 0:
            offsets_vals = [v for k, v in output['sequence_offsets'].items()
                           if k.startswith(f"{ds_name}/")]
            mags = [np.sqrt(v[0]**2 + v[1]**2) for v in offsets_vals]
            print(f"  Offset stats: median={np.median(mags):.1f}px, "
                  f"mean={np.mean(mags):.1f}px")

    # ---- 4. Per-image MI datasets ----
    mi_datasets = {
        'msuav-RGBMS': 80,
        'LLVIP': 80,
        'DroneRGBT': 60,
        'TarDAL-M3FD': 60,
        'aerial-rgbt': 60,
    }
    for ds_name, search_range in mi_datasets.items():
        if ds_name not in by_ds:
            continue
        samples = by_ds[ds_name]
        print(f"\n=== {ds_name}: per-image MI ({len(samples)} images) ===")
        t0 = time.time()

        tasks = []
        for s in samples:
            rgb_p = s.get('rgb_path')
            ir_p = s.get('ir_path')
            if rgb_p and ir_p:
                tasks.append((ir_p, rgb_p, search_range))

        offsets = {}
        done = 0
        total = len(tasks)

        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_mi_worker, t): t for t in tasks}
            for future in as_completed(futures):
                ir_path, result = future.result()
                done += 1
                if result is not None:
                    offsets[ir_path] = list(result)
                if done % 1000 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{ds_name}] {done}/{total} "
                          f"({rate:.0f} img/s, ETA {eta:.0f}s)")

        output['per_image_offsets'].update(offsets)
        output['method_log'][ds_name] = 'mutual_info_per_image'

        if offsets:
            mags = [np.sqrt(v[0]**2 + v[1]**2) for v in offsets.values()]
            print(f"  {len(offsets)}/{len(samples)} computed in "
                  f"{time.time()-t0:.1f}s")
            print(f"  Offset stats: median={np.median(mags):.1f}px, "
                  f"mean={np.mean(mags):.1f}px, max={np.max(mags):.1f}px")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    n_seq = len(output['sequence_offsets'])
    n_img = len(output['per_image_offsets'])
    print(f"\n{'='*60}")
    print(f"Saved to {args.output}")
    print(f"  Sequence offsets: {n_seq}")
    print(f"  Per-image offsets: {n_img}")
    print(f"  Methods: {output['method_log']}")

    # Coverage summary
    print(f"\nCoverage summary:")
    for ds_name, samples in sorted(by_ds.items()):
        if ds_name == 'DroneVehicle':
            print(f"  {ds_name}: {len(samples)} (inherently aligned)")
            continue
        if ds_name == 'UTUAV':
            n_seq_ds = sum(1 for k in output['sequence_offsets'] if k.startswith('UTUAV/'))
            print(f"  {ds_name}: {len(samples)} images, {n_seq_ds} sequence offsets")
            continue
        # Check sequence offsets
        n_seq_ds = sum(1 for k in output['sequence_offsets'] if k.startswith(f'{ds_name}/'))
        # Check per-image offsets
        ir_paths = set(s.get('ir_path') for s in samples if s.get('ir_path'))
        n_img_ds = sum(1 for p in ir_paths if p in output['per_image_offsets'])
        if n_seq_ds > 0:
            print(f"  {ds_name}: {len(samples)} images, {n_seq_ds} sequence offsets")
        else:
            print(f"  {ds_name}: {n_img_ds}/{len(samples)} per-image offsets")


if __name__ == '__main__':
    main()

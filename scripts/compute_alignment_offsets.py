#!/usr/bin/env python3
"""
Compute RGB-IR alignment offsets for all paired datasets.

Methods:
  - UTUAV: bbox center difference from tracking annotations (per-sequence)
  - VT-Tiny-MOT / RGBT-Tiny / aerial-rgbt: mutual information (per-image)
  - M3OT: mutual information (global offset from sample)

Output: alignment_offsets.json
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
# UTUAV: bbox-based alignment
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

    # bbox center differences
    dx_arr = (ir_b[:, 0] + ir_b[:, 2] / 2) - (rgb_b[:, 0] + rgb_b[:, 2] / 2)
    dy_arr = (ir_b[:, 1] + ir_b[:, 3] / 2) - (rgb_b[:, 1] + rgb_b[:, 3] / 2)

    # IQR outlier removal
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
    """Compute per-sequence offsets for the entire UTUAV dataset."""
    sequence_offsets = {}
    seqs = sorted(os.listdir(utuav_root))
    for seq in seqs:
        seq_dir = os.path.join(utuav_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        result = compute_utuav_sequence_offset(seq_dir)
        if result is not None:
            key = f"UTUAV/{seq}"
            sequence_offsets[key] = list(result)
    return sequence_offsets


# ---------------------------------------------------------------------------
# Mutual Information alignment
# ---------------------------------------------------------------------------

def _negative_mutual_info(params, rgb_gray, ir_gray, w, h):
    """Negative mutual information for optimization."""
    dx, dy = params
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    ir_shifted = cv2.warpAffine(ir_gray, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)

    # Joint histogram → MI
    hist2d = np.histogram2d(
        rgb_gray.ravel(), ir_shifted.ravel(), bins=32,
        range=[[0, 256], [0, 256]]
    )[0]
    hist2d = hist2d + 1e-10  # avoid log(0)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = sum pxy * log(pxy / (px * py))
    px_py = np.outer(px, py)
    mi = np.sum(pxy * np.log(pxy / (px_py + 1e-10)))
    return -mi


def compute_mi_offset(rgb_path, ir_path, search_range=80, target_size=256):
    """Compute alignment offset for one image pair using mutual information."""
    rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        return None

    # Resize IR to match RGB if needed
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
    """Worker function for parallel MI computation."""
    ir_path, rgb_path, search_range = args
    result = compute_mi_offset(rgb_path, ir_path, search_range=search_range)
    return ir_path, result


def compute_mi_offsets_batch(samples, search_range=80, n_workers=8,
                             dataset_name=""):
    """Compute MI offsets for a batch of samples in parallel."""
    tasks = []
    for s in samples:
        rgb_k = next((k for k in s if any(x in k.lower() for x in
                      ['rgb', 'visible', 'img'])), None)
        ir_k = next((k for k in s if any(x in k.lower() for x in
                     ['ir', 'thermal', 'infrared'])), None)
        if rgb_k and ir_k and s.get(rgb_k) and s.get(ir_k):
            tasks.append((s[ir_k], s[rgb_k], search_range))

    offsets = {}
    done = 0
    total = len(tasks)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_mi_worker, t): t for t in tasks}
        for future in as_completed(futures):
            ir_path, result = future.result()
            done += 1
            if result is not None:
                offsets[ir_path] = list(result)
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{dataset_name}] {done}/{total}  "
                      f"({rate:.0f} img/s, ETA {eta:.0f}s)")

    return offsets


def compute_m3ot_global_offset(samples, n_sample=50):
    """Compute a single global offset for M3OT by sampling n images."""
    indices = np.linspace(0, len(samples) - 1, min(n_sample, len(samples)),
                          dtype=int)
    dxs, dys = [], []
    for idx in indices:
        s = samples[idx]
        rgb_k = next((k for k in s if any(x in k.lower() for x in
                      ['rgb', 'visible', 'img'])), None)
        ir_k = next((k for k in s if any(x in k.lower() for x in
                     ['ir', 'thermal', 'infrared'])), None)
        if not rgb_k or not ir_k:
            continue
        result = compute_mi_offset(s[rgb_k], s[ir_k], search_range=60)
        if result is not None:
            dxs.append(result[0])
            dys.append(result[1])

    if len(dxs) >= 3:
        return (float(np.median(dxs)), float(np.median(dys)))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute RGB-IR alignment offsets for paired datasets")
    parser.add_argument('--manifest',
                        default='/root/autodl-tmp/train1/manifests/'
                                'merged_full_pretrain_v2.json',
                        help='Path to manifest JSON')
    parser.add_argument('--utuav_root',
                        default='/root/autodl-tmp/data/UTUAV',
                        help='Path to UTUAV dataset root')
    parser.add_argument('--output',
                        default='/root/autodl-tmp/train1/manifests/'
                                'alignment_offsets.json',
                        help='Output JSON path')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers for MI computation')
    parser.add_argument('--skip_mi', action='store_true',
                        help='Skip MI computation (only compute UTUAV bbox)')
    args = parser.parse_args()

    print("Loading manifest...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    paired = [s for s in manifest['samples'] if s.get('type') == 'paired']

    # Group by dataset
    by_dataset = defaultdict(list)
    for s in paired:
        by_dataset[s.get('dataset', 'unknown')].append(s)

    print(f"Paired datasets: {', '.join(f'{k}({len(v)})' for k, v in by_dataset.items())}")

    output = {
        'version': '1.0',
        'method_log': {},
        'sequence_offsets': {},
        'per_image_offsets': {},
    }

    # ---- UTUAV: bbox method ----
    if 'UTUAV' in by_dataset:
        print("\n=== UTUAV: bbox center difference ===")
        t0 = time.time()
        seq_offsets = compute_utuav_offsets(args.utuav_root)
        output['sequence_offsets'].update(seq_offsets)
        output['method_log']['UTUAV'] = 'bbox_center_diff'

        # Stats
        totals = [np.sqrt(v[0]**2 + v[1]**2) for v in seq_offsets.values()]
        print(f"  {len(seq_offsets)} sequences computed in {time.time()-t0:.1f}s")
        if totals:
            print(f"  Offset stats: median={np.median(totals):.1f}px, "
                  f"mean={np.mean(totals):.1f}px, "
                  f"max={np.max(totals):.1f}px")

    if not args.skip_mi:
        # ---- M3OT: global MI offset ----
        if 'M3OT' in by_dataset:
            print("\n=== M3OT: mutual information (global) ===")
            t0 = time.time()
            global_offset = compute_m3ot_global_offset(by_dataset['M3OT'])
            if global_offset is not None:
                print(f"  Global offset: dx={global_offset[0]:+.1f}, "
                      f"dy={global_offset[1]:+.1f}")
                # Apply to all M3OT samples
                for s in by_dataset['M3OT']:
                    ir_k = next((k for k in s if any(x in k.lower() for x in
                                 ['ir', 'thermal', 'infrared'])), None)
                    if ir_k and s.get(ir_k):
                        output['per_image_offsets'][s[ir_k]] = list(global_offset)
                output['method_log']['M3OT'] = 'mutual_info_global'
            print(f"  Done in {time.time()-t0:.1f}s")

        # ---- Per-image MI datasets ----
        mi_datasets = {
            'VT-Tiny-MOT': 80,
            'RGBT-Tiny': 80,
            'aerial-rgbt': 80,
        }
        for ds_name, search_range in mi_datasets.items():
            if ds_name not in by_dataset:
                continue
            samples = by_dataset[ds_name]
            print(f"\n=== {ds_name}: mutual information ({len(samples)} images) ===")
            t0 = time.time()
            offsets = compute_mi_offsets_batch(
                samples, search_range=search_range,
                n_workers=args.n_workers, dataset_name=ds_name
            )
            output['per_image_offsets'].update(offsets)
            output['method_log'][ds_name] = 'mutual_info_per_image'

            if offsets:
                totals = [np.sqrt(v[0]**2 + v[1]**2) for v in offsets.values()]
                print(f"  {len(offsets)}/{len(samples)} computed in "
                      f"{time.time()-t0:.1f}s")
                print(f"  Offset stats: median={np.median(totals):.1f}px, "
                      f"mean={np.mean(totals):.1f}px, "
                      f"max={np.max(totals):.1f}px")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    n_seq = len(output['sequence_offsets'])
    n_img = len(output['per_image_offsets'])
    print(f"\nSaved to {args.output}")
    print(f"  Sequence offsets: {n_seq}")
    print(f"  Per-image offsets: {n_img}")


if __name__ == '__main__':
    main()

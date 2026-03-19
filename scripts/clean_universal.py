#!/usr/bin/env python3
"""
Universal multi-modal dataset cleaning script.

Supports:
- RGBT-Tiny: 640x512 RGB + 640x512 IR (matched resolution)
- DVTOD: 1920x1080 RGB + 640x512 IR (mismatched resolution)
- Other datasets with various structures

Key features:
- NEVER modifies original data
- Outputs cleaned/resized images to a new directory
- Generates JSON manifest for training
- Handles pseudo-RGB IR (3ch IR → extract single channel)
- Handles resolution mismatch (resize to target)

Usage:
    # RGBT-Tiny (resolutions match, just symlink + manifest)
    python scripts/clean_universal.py \
        --dataset rgbt_tiny \
        --data_root /root/autodl-tmp/data/RGBT-Tiny \
        --output_dir /root/autodl-tmp/train1/rgbt_tiny \
        --split train

    # DVTOD (RGB 1920x1080, IR 640x512 → unify to 640x512)
    python scripts/clean_universal.py \
        --dataset dvtod \
        --data_root "/root/autodl-tmp/data/DVTOD/DVTOD dataset/DVTOD dataset" \
        --output_dir /root/autodl-tmp/train1/dvtod \
        --split train \
        --target_w 640 --target_h 512
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser('Universal Multi-Modal Data Cleaning')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['rgbt_tiny', 'dvtod', 'llvip', 'drone_rgbt',
                                 'drone_vehicle', 'utuav', 'tardal', 'm3ot',
                                 'vt_tiny_mot'],
                        help='Dataset type')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of original dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for cleaned data and manifest')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test', 'val', 'all'])
    parser.add_argument('--target_w', type=int, default=0,
                        help='Target width (0 = auto from dataset defaults)')
    parser.add_argument('--target_h', type=int, default=0,
                        help='Target height (0 = auto from dataset defaults)')
    parser.add_argument('--copy_mode', type=str, default='auto',
                        choices=['auto', 'copy', 'symlink', 'resize'],
                        help='How to handle output images: '
                             'auto = symlink if no resize needed, else resize; '
                             'copy = always copy original files; '
                             'symlink = always symlink; '
                             'resize = always resize to target')
    parser.add_argument('--max_per_seq', type=int, default=0,
                        help='Max samples per sequence (0 = unlimited). '
                             'Useful for large datasets like UTUAV (1.6M images)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train ratio when dataset has no split file '
                             '(used for UTUAV etc.)')
    parser.add_argument('--mean_low', type=float, default=5.0)
    parser.add_argument('--mean_high', type=float, default=250.0)
    parser.add_argument('--std_low', type=float, default=10.0)
    return parser.parse_args()


# ============================================================================
# Dataset-specific scanners
# ============================================================================

def scan_rgbt_tiny(data_root, split):
    """
    RGBT-Tiny: images/{seq}/00/ (RGB), images/{seq}/01/ (IR)
    Both 640x512. Split files in data_split/.
    """
    images_dir = os.path.join(data_root, 'images')

    # Get sequences from split file
    if split == 'all':
        seqs = sorted([d for d in os.listdir(images_dir)
                       if os.path.isdir(os.path.join(images_dir, d))])
    else:
        split_map = {'train': '00_train.txt', 'test': '00_test.txt'}
        split_file = os.path.join(data_root, 'data_split', split_map.get(split, '00_train.txt'))
        with open(split_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        seqs = sorted(set(l.split('/')[0] for l in lines))

    pairs = []
    for seq in seqs:
        rgb_dir = os.path.join(images_dir, seq, '00')
        ir_dir = os.path.join(images_dir, seq, '01')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
            continue
        rgb_files = set(os.listdir(rgb_dir))
        ir_files = set(os.listdir(ir_dir))
        common = sorted(rgb_files & ir_files)
        for fname in common:
            pairs.append({
                'rgb_path': os.path.join(rgb_dir, fname),
                'ir_path': os.path.join(ir_dir, fname),
                'sequence': seq,
                'frame': fname,
                'rgb_resolution': (640, 512),  # w, h
                'ir_resolution': (640, 512),
                'ir_is_pseudo_rgb': False,
            })
    return pairs, (640, 512)


def scan_dvtod(data_root, split):
    """
    DVTOD: vis1/images/{split}/ (RGB 1920x1080), ir1/images/{split}/ (IR 640x512 pseudo-RGB)
    """
    split_name = split if split != 'test' else 'val'
    if split == 'all':
        split_names = ['train', 'val']
    else:
        split_names = [split_name]

    pairs = []
    for sn in split_names:
        rgb_dir = os.path.join(data_root, 'vis1', 'images', sn)
        ir_dir = os.path.join(data_root, 'ir1', 'images', sn)
        if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
            print(f"Warning: {rgb_dir} or {ir_dir} not found")
            continue
        rgb_files = set(os.listdir(rgb_dir))
        ir_files = set(os.listdir(ir_dir))
        common = sorted(rgb_files & ir_files, key=lambda x: int(x.split('.')[0]))
        for fname in common:
            pairs.append({
                'rgb_path': os.path.join(rgb_dir, fname),
                'ir_path': os.path.join(ir_dir, fname),
                'sequence': f'dvtod_{sn}',
                'frame': fname,
                'rgb_resolution': (1920, 1080),
                'ir_resolution': (640, 512),
                'ir_is_pseudo_rgb': True,  # IR stored as 3-channel JPEG
            })
    return pairs, (640, 512)  # Default target = IR resolution


def scan_utuav(data_root, split):
    """
    UTUAV: RGB-T/{seq}/rgb/ (RGB 1920x1080), RGB-T/{seq}/ir/ (IR 1920x1080 pseudo-RGB)
    500 sequences, ~1.66M pairs total. No official split — use train_ratio.
    Only uses the RGB-T/ directory (paired data).
    """
    rgbt_dir = os.path.join(data_root, 'RGB-T')
    if not os.path.isdir(rgbt_dir):
        print(f"Error: RGB-T directory not found: {rgbt_dir}")
        return [], (640, 512)

    all_seqs = sorted([d for d in os.listdir(rgbt_dir)
                       if os.path.isdir(os.path.join(rgbt_dir, d))])
    print(f"UTUAV: found {len(all_seqs)} sequences in RGB-T/")

    # Split sequences deterministically (sorted order, 80/20)
    n_train = int(len(all_seqs) * 0.8)
    if split == 'train':
        seqs = all_seqs[:n_train]
    elif split in ('test', 'val'):
        seqs = all_seqs[n_train:]
    else:
        seqs = all_seqs

    print(f"Using {len(seqs)} sequences for split='{split}'")

    pairs = []
    for seq in seqs:
        rgb_dir = os.path.join(rgbt_dir, seq, 'rgb')
        ir_dir = os.path.join(rgbt_dir, seq, 'ir')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
            continue
        rgb_files = set(f for f in os.listdir(rgb_dir) if f.endswith('.jpg'))
        ir_files = set(f for f in os.listdir(ir_dir) if f.endswith('.jpg'))
        common = sorted(rgb_files & ir_files)

        for fname in common:
            pairs.append({
                'rgb_path': os.path.join(rgb_dir, fname),
                'ir_path': os.path.join(ir_dir, fname),
                'sequence': seq,
                'frame': fname,
                'rgb_resolution': (1920, 1080),
                'ir_resolution': (1920, 1080),
                'ir_is_pseudo_rgb': True,
            })
    return pairs, (640, 512)  # Downsample both to 640x512


DATASET_SCANNERS = {
    'rgbt_tiny': scan_rgbt_tiny,
    'dvtod': scan_dvtod,
    'utuav': scan_utuav,
}


# ============================================================================
# Image quality check
# ============================================================================

def check_quality(img_arr, mean_low=5.0, mean_high=250.0, std_low=10.0):
    """Check image for anomalies. Returns list of anomaly strings."""
    anomalies = []
    if img_arr is None or img_arr.size == 0:
        return ['corrupt_empty']
    mean_val = float(img_arr.mean())
    std_val = float(img_arr.std())
    if mean_val < mean_low:
        anomalies.append(f'all_black(mean={mean_val:.1f})')
    if mean_val > mean_high:
        anomalies.append(f'all_white(mean={mean_val:.1f})')
    if std_val < std_low:
        anomalies.append(f'low_contrast(std={std_val:.1f})')
    return anomalies


# ============================================================================
# Main cleaning pipeline
# ============================================================================

def process_pair(pair, output_dir, target_w, target_h, copy_mode, args):
    """
    Process a single RGB+IR pair:
    1. Read images
    2. Convert pseudo-RGB IR to grayscale
    3. Resize if needed
    4. Quality check
    5. Save to output directory

    Returns:
        sample_dict or None (if excluded), anomaly_list
    """
    anomalies = []

    # --- Read RGB ---
    try:
        rgb = cv2.imread(pair['rgb_path'], cv2.IMREAD_COLOR)
        if rgb is None:
            return None, ['rgb_corrupt']
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None, [f'rgb_error({e})']

    # --- Read IR ---
    try:
        ir_raw = cv2.imread(pair['ir_path'], cv2.IMREAD_UNCHANGED)
        if ir_raw is None:
            return None, ['ir_corrupt']

        # Handle pseudo-RGB IR (3-channel IR stored as JPEG)
        if pair.get('ir_is_pseudo_rgb', False) and len(ir_raw.shape) == 3:
            ir = ir_raw[:, :, 0]  # Extract first channel
        elif len(ir_raw.shape) == 3 and ir_raw.shape[2] == 3:
            # Auto-detect: if all 3 channels are identical, it's pseudo-RGB
            if np.allclose(ir_raw[:, :, 0], ir_raw[:, :, 1]) and \
               np.allclose(ir_raw[:, :, 0], ir_raw[:, :, 2]):
                ir = ir_raw[:, :, 0]
            else:
                ir = cv2.cvtColor(ir_raw, cv2.COLOR_BGR2GRAY)
        elif len(ir_raw.shape) == 2:
            ir = ir_raw
        else:
            ir = ir_raw[:, :, 0] if len(ir_raw.shape) == 3 else ir_raw
    except Exception as e:
        return None, [f'ir_error({e})']

    # --- Quality check on originals ---
    rgb_anomalies = check_quality(rgb, args.mean_low, args.mean_high, args.std_low)
    ir_anomalies = check_quality(ir, args.mean_low, args.mean_high, args.std_low)

    for a in rgb_anomalies:
        anomalies.append(f'RGB:{a}')
    for a in ir_anomalies:
        anomalies.append(f'IR:{a}')

    if anomalies:
        return None, anomalies

    # --- Determine if resize is needed ---
    rgb_h, rgb_w = rgb.shape[:2]
    ir_h, ir_w = ir.shape[:2]

    rgb_needs_resize = (rgb_w != target_w or rgb_h != target_h)
    ir_needs_resize = (ir_w != target_w or ir_h != target_h)
    any_resize = rgb_needs_resize or ir_needs_resize

    # --- Decide copy mode ---
    actual_mode = copy_mode
    if actual_mode == 'auto':
        actual_mode = 'resize' if any_resize else 'symlink'

    # --- Prepare output paths ---
    seq = pair.get('sequence', 'default')
    frame = pair['frame']
    rgb_out_dir = os.path.join(output_dir, 'images', seq, 'rgb')
    ir_out_dir = os.path.join(output_dir, 'images', seq, 'ir')
    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(ir_out_dir, exist_ok=True)

    # Change extension to .png for IR (lossless grayscale)
    frame_base = os.path.splitext(frame)[0]
    rgb_out_path = os.path.join(rgb_out_dir, frame)
    ir_out_path = os.path.join(ir_out_dir, frame_base + '.png')

    # --- Save images ---
    if actual_mode == 'symlink' and not any_resize:
        # No resize needed: symlink RGB, save IR as grayscale PNG
        rgb_src = os.path.abspath(pair['rgb_path'])
        if not os.path.exists(rgb_out_path):
            os.symlink(rgb_src, rgb_out_path)
        # For IR: always save as single-channel PNG
        if pair.get('ir_is_pseudo_rgb', False):
            cv2.imwrite(ir_out_path, ir)
        else:
            ir_src = os.path.abspath(pair['ir_path'])
            if not os.path.exists(ir_out_path):
                if pair['ir_path'].endswith('.png') and len(ir_raw.shape) == 2:
                    os.symlink(ir_src, ir_out_path)
                else:
                    cv2.imwrite(ir_out_path, ir)
    elif actual_mode in ('resize', 'auto'):
        # Resize to target resolution
        if rgb_needs_resize:
            interp = cv2.INTER_AREA if rgb_w > target_w else cv2.INTER_LINEAR
            rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=interp)
            rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_out_path, rgb_bgr)
        else:
            if not os.path.exists(rgb_out_path):
                shutil.copy2(pair['rgb_path'], rgb_out_path)

        if ir_needs_resize:
            interp = cv2.INTER_AREA if ir_w > target_w else cv2.INTER_LINEAR
            ir_resized = cv2.resize(ir, (target_w, target_h), interpolation=interp)
            cv2.imwrite(ir_out_path, ir_resized)
        else:
            cv2.imwrite(ir_out_path, ir)
    else:
        # Copy mode
        if not os.path.exists(rgb_out_path):
            shutil.copy2(pair['rgb_path'], rgb_out_path)
        cv2.imwrite(ir_out_path, ir)

    return {
        'rgb_path': os.path.abspath(rgb_out_path),
        'ir_path': os.path.abspath(ir_out_path),
        'type': 'paired',
        'dataset': args.dataset,
        'sequence': seq,
        'original_rgb_size': f'{rgb_w}x{rgb_h}',
        'original_ir_size': f'{ir_w}x{ir_h}',
        'output_size': f'{target_w}x{target_h}',
    }, []


def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"Universal Multi-Modal Data Cleaning")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data root: {args.data_root}")
    print(f"  Output: {args.output_dir}")
    print(f"  Split: {args.split}")
    print()

    # --- Scan dataset ---
    if args.dataset not in DATASET_SCANNERS:
        print(f"Error: No scanner for dataset '{args.dataset}'")
        sys.exit(1)

    pairs, default_target = DATASET_SCANNERS[args.dataset](args.data_root, args.split)

    target_w = args.target_w if args.target_w > 0 else default_target[0]
    target_h = args.target_h if args.target_h > 0 else default_target[1]

    print(f"Scanned {len(pairs)} pairs")
    print(f"Target resolution: {target_w}x{target_h}")

    # Apply max_per_seq limit
    if args.max_per_seq > 0:
        from collections import defaultdict as dd
        seq_counts = dd(int)
        filtered = []
        for p in pairs:
            seq = p.get('sequence', 'default')
            if seq_counts[seq] < args.max_per_seq:
                filtered.append(p)
                seq_counts[seq] += 1
        print(f"After max_per_seq={args.max_per_seq}: {len(filtered)} pairs "
              f"(from {len(pairs)})")
        pairs = filtered

    if pairs:
        print(f"Original RGB: {pairs[0]['rgb_resolution'][0]}x{pairs[0]['rgb_resolution'][1]}")
        print(f"Original IR:  {pairs[0]['ir_resolution'][0]}x{pairs[0]['ir_resolution'][1]}")
        needs_resize = (pairs[0]['rgb_resolution'] != (target_w, target_h) or
                        pairs[0]['ir_resolution'] != (target_w, target_h))
        print(f"Resize needed: {'YES' if needs_resize else 'NO'}")
    print()

    # --- Process each pair ---
    os.makedirs(args.output_dir, exist_ok=True)

    clean_samples = []
    excluded = []
    anomaly_counts = defaultdict(int)

    for i, pair in enumerate(pairs):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(pairs)}...")

        sample, anomalies = process_pair(
            pair, args.output_dir, target_w, target_h, args.copy_mode, args)

        if sample is not None:
            clean_samples.append(sample)
        else:
            for a in anomalies:
                anomaly_counts[a.split('(')[0]] += 1
            excluded.append({
                'rgb_path': pair['rgb_path'],
                'ir_path': pair['ir_path'],
                'anomalies': anomalies,
            })

    # --- Generate manifest ---
    manifest = {
        'dataset': args.dataset,
        'split': args.split,
        'target_resolution': f'{target_w}x{target_h}',
        'total_scanned': len(pairs),
        'total_clean': len(clean_samples),
        'total_excluded': len(excluded),
        'samples': clean_samples,
    }

    manifest_path = os.path.join(args.output_dir, f'manifest_{args.split}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # --- Generate report ---
    report = [
        '=' * 60,
        f'{args.dataset.upper()} Data Cleaning Report',
        '=' * 60,
        f'Data root: {args.data_root}',
        f'Output dir: {args.output_dir}',
        f'Split: {args.split}',
        f'Target resolution: {target_w}x{target_h}',
        f'Total scanned: {len(pairs)}',
        f'Clean: {len(clean_samples)}',
        f'Excluded: {len(excluded)}',
        '',
    ]

    if anomaly_counts:
        report.append('Anomaly distribution:')
        for k, v in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            report.append(f'  {k}: {v}')
        report.append('')

    if excluded:
        report.append(f'Excluded samples (first 20):')
        for s in excluded[:20]:
            report.append(f'  {os.path.basename(s["rgb_path"])}: {", ".join(s["anomalies"])}')

    report_text = '\n'.join(report)
    report_path = os.path.join(args.output_dir, f'cleaning_report_{args.split}.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f'\nManifest: {manifest_path}')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()

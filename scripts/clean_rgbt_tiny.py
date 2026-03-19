#!/usr/bin/env python3
"""
RGBT-Tiny dataset cleaning and manifest generation script.

Usage:
    python scripts/clean_rgbt_tiny.py \
        --data_root /root/autodl-tmp/data/RGBT-Tiny \
        --output_dir /root/autodl-tmp/data/RGBT-Tiny \
        --split train

Produces:
    - manifest_train.json (or manifest_test.json)
    - cleaning_report.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser('RGBT-Tiny Data Cleaning')
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/data/RGBT-Tiny',
                        help='Root directory of RGBT-Tiny dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/data/RGBT-Tiny',
                        help='Output directory for manifest and report')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--mean_threshold_low', type=float, default=5.0,
                        help='Threshold for all-black detection')
    parser.add_argument('--mean_threshold_high', type=float, default=250.0,
                        help='Threshold for all-white detection')
    parser.add_argument('--std_threshold', type=float, default=10.0,
                        help='Threshold for low-contrast detection')
    parser.add_argument('--expected_width', type=int, default=640,
                        help='Expected image width')
    parser.add_argument('--expected_height', type=int, default=512,
                        help='Expected image height')
    return parser.parse_args()


def get_sequences(data_root, split):
    """Get sequence names for the given split."""
    images_dir = os.path.join(data_root, 'images')

    if split == 'all':
        # Use all sequences found in images/
        sequences = sorted([d for d in os.listdir(images_dir)
                           if os.path.isdir(os.path.join(images_dir, d))])
        return sequences

    # Try to read from split file
    # Split files contain per-image paths like "DJI_0022_1/00/00000"
    # We extract unique sequence names (first component)
    split_file = os.path.join(data_root, 'data_split',
                              '00_train.txt' if split == 'train' else '00_test.txt')

    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Extract unique sequence names
        sequences = sorted(set(line.split('/')[0] for line in lines))
        print(f"Loaded {len(lines)} entries -> {len(sequences)} unique sequences from {split_file}")
        return sequences

    # Fallback: use all sequences
    print(f"Split file not found: {split_file}, using all sequences")
    sequences = sorted([d for d in os.listdir(images_dir)
                       if os.path.isdir(os.path.join(images_dir, d))])
    return sequences


def scan_sequence(images_dir, seq_name):
    """
    Scan a sequence directory for paired RGB+IR images.

    Expected structure:
        images/{seq_name}/00/  -> RGB images
        images/{seq_name}/01/  -> IR images

    Returns:
        pairs: list of (rgb_path, ir_path, frame_name)
        errors: list of error descriptions
    """
    rgb_dir = os.path.join(images_dir, seq_name, '00')
    ir_dir = os.path.join(images_dir, seq_name, '01')

    pairs = []
    errors = []

    if not os.path.isdir(rgb_dir):
        errors.append(f"RGB dir missing: {rgb_dir}")
        return pairs, errors

    if not os.path.isdir(ir_dir):
        errors.append(f"IR dir missing: {ir_dir}")
        return pairs, errors

    rgb_files = set(os.listdir(rgb_dir))
    ir_files = set(os.listdir(ir_dir))

    # Find matching pairs
    common = sorted(rgb_files & ir_files)
    rgb_only = sorted(rgb_files - ir_files)
    ir_only = sorted(ir_files - rgb_files)

    if rgb_only:
        errors.append(f"  {seq_name}: {len(rgb_only)} RGB-only files (no IR match)")
    if ir_only:
        errors.append(f"  {seq_name}: {len(ir_only)} IR-only files (no RGB match)")

    for fname in common:
        rgb_path = os.path.join(rgb_dir, fname)
        ir_path = os.path.join(ir_dir, fname)
        pairs.append((rgb_path, ir_path, fname))

    return pairs, errors


def check_image(path, expected_w, expected_h):
    """
    Check a single image for anomalies.

    Returns:
        anomalies: list of anomaly strings (empty if OK)
        stats: dict with mean, std, size
    """
    anomalies = []
    stats = {}

    try:
        img = Image.open(path)
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        stats['width'] = w
        stats['height'] = h
        stats['mean'] = float(arr.mean())
        stats['std'] = float(arr.std())
        stats['mode'] = img.mode

        # Size check
        if w != expected_w or h != expected_h:
            anomalies.append(f"size={w}x{h} (expected {expected_w}x{expected_h})")

        # All black
        if stats['mean'] < 5.0:
            anomalies.append(f"all_black (mean={stats['mean']:.1f})")

        # All white
        if stats['mean'] > 250.0:
            anomalies.append(f"all_white (mean={stats['mean']:.1f})")

        # Low contrast
        if stats['std'] < 10.0:
            anomalies.append(f"low_contrast (std={stats['std']:.1f})")

    except Exception as e:
        anomalies.append(f"corrupt ({str(e)})")
        stats = {'error': str(e)}

    return anomalies, stats


def main():
    args = parse_args()
    data_root = args.data_root
    images_dir = os.path.join(data_root, 'images')

    print(f"RGBT-Tiny Data Cleaning")
    print(f"  Data root: {data_root}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output_dir}")
    print()

    # Get sequences for this split
    sequences = get_sequences(data_root, args.split)
    print(f"Processing {len(sequences)} sequences...")

    # Scan all sequences
    all_pairs = []
    scan_errors = []
    for seq in sequences:
        pairs, errors = scan_sequence(images_dir, seq)
        for rgb_path, ir_path, fname in pairs:
            all_pairs.append({
                'rgb_path': rgb_path,
                'ir_path': ir_path,
                'frame': fname,
                'sequence': seq,
            })
        scan_errors.extend(errors)

    print(f"Found {len(all_pairs)} paired images across {len(sequences)} sequences")
    if scan_errors:
        print(f"Scan warnings: {len(scan_errors)}")
        for e in scan_errors[:10]:
            print(f"  {e}")

    # Check each pair for anomalies
    print("\nChecking image quality...")
    clean_samples = []
    excluded_samples = []
    anomaly_counts = defaultdict(int)

    for i, pair in enumerate(all_pairs):
        if (i + 1) % 5000 == 0:
            print(f"  Checked {i+1}/{len(all_pairs)}...")

        # Check RGB
        rgb_anomalies, rgb_stats = check_image(
            pair['rgb_path'], args.expected_width, args.expected_height)

        # Check IR
        ir_anomalies, ir_stats = check_image(
            pair['ir_path'], args.expected_width, args.expected_height)

        all_anomalies = []
        for a in rgb_anomalies:
            all_anomalies.append(f"RGB: {a}")
            anomaly_counts[f"RGB_{a.split('(')[0].strip()}"] += 1
        for a in ir_anomalies:
            all_anomalies.append(f"IR: {a}")
            anomaly_counts[f"IR_{a.split('(')[0].strip()}"] += 1

        if all_anomalies:
            excluded_samples.append({
                **pair,
                'anomalies': all_anomalies,
            })
        else:
            clean_samples.append({
                'rgb_path': pair['rgb_path'],
                'ir_path': pair['ir_path'],
                'type': 'paired',
                'dataset': 'RGBT-Tiny',
                'sequence': pair['sequence'],
            })

    # Generate manifest
    manifest = {
        'dataset': 'RGBT-Tiny',
        'split': args.split,
        'total_scanned': len(all_pairs),
        'total_clean': len(clean_samples),
        'total_excluded': len(excluded_samples),
        'samples': clean_samples,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, f'manifest_{args.split}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Generate report
    report_lines = [
        "=" * 60,
        "RGBT-Tiny Data Cleaning Report",
        "=" * 60,
        f"Data root: {data_root}",
        f"Split: {args.split}",
        f"Sequences: {len(sequences)}",
        f"Total pairs scanned: {len(all_pairs)}",
        f"Clean pairs: {len(clean_samples)}",
        f"Excluded pairs: {len(excluded_samples)}",
        "",
        "Anomaly distribution:",
    ]
    for anomaly_type, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {anomaly_type}: {count}")

    if excluded_samples:
        report_lines.append("")
        report_lines.append(f"Excluded samples (first 20):")
        for s in excluded_samples[:20]:
            report_lines.append(f"  {s['sequence']}/{s['frame']}: {', '.join(s['anomalies'])}")

    report_lines.append("")
    report_lines.append("Sequence summary:")
    seq_counts = defaultdict(int)
    for s in clean_samples:
        seq_counts[s['sequence']] += 1
    for seq in sorted(seq_counts.keys()):
        report_lines.append(f"  {seq}: {seq_counts[seq]} pairs")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.output_dir, f'cleaning_report_{args.split}.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()

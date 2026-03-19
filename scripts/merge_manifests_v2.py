#!/usr/bin/env python3
"""
Merge all manifests (train/val/test) into a single pretraining manifest.

Logic:
  1. Load ALL per-dataset manifests from --manifest_dir (train + val + test + all)
  2. REPLACE old LLVIP registered manifests with new LLVIP_unreg_all.json
  3. EXCLUDE WebUAV-3M (no manifest exists, so nothing to drop)
  4. EXCLUDE UTUAV (kept separate due to its size — 1.6M samples)
  5. EXCLUDE already-merged files (merged_train*.json)
  6. Ensure every sample has a 'type' field (paired / rgb_only / ir_only)
  7. Concatenate selected manifests and output statistics
  8. Output merged_train_all_v2.json

Usage:
    python scripts/merge_manifests_v2.py [--manifest_dir /root/autodl-tmp/train1/manifests]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def infer_type(sample: dict, manifest_modality: str | None) -> str:
    """Infer sample type from its fields or parent manifest modality."""
    # Already has type
    if sample.get("type") in ("paired", "rgb_only", "ir_only"):
        return sample["type"]

    # Infer from paths
    has_rgb = sample.get("rgb_path") is not None
    has_ir = sample.get("ir_path") is not None

    if has_rgb and has_ir:
        return "paired"
    elif has_rgb:
        return "rgb_only"
    elif has_ir:
        return "ir_only"

    # Fallback to manifest-level modality_type
    if manifest_modality:
        return manifest_modality

    return "paired"  # conservative fallback


def load_manifest(path: str) -> tuple[list[dict], dict]:
    """Load a manifest and return (samples, metadata)."""
    with open(path, 'r') as f:
        data = json.load(f)

    modality_type = data.get("modality_type")
    samples = data.get("samples", [])

    # Ensure every sample has 'type'
    for s in samples:
        if "type" not in s or s["type"] is None:
            s["type"] = infer_type(s, modality_type)

    return samples, data


def main():
    parser = argparse.ArgumentParser(
        description="Merge all manifests into a single pretraining manifest")
    parser.add_argument("--manifest_dir", type=str,
                        default="/root/autodl-tmp/train1/manifests",
                        help="Directory containing all manifest JSON files")
    parser.add_argument("--output", type=str,
                        default="/root/autodl-tmp/train1/manifests/merged_train_all_v2.json",
                        help="Output path for merged manifest")
    args = parser.parse_args()

    manifest_dir = args.manifest_dir
    output_path = args.output

    # --- Gather all manifest files ---
    all_files = sorted(f for f in os.listdir(manifest_dir)
                       if f.endswith('.json'))

    # Files to skip
    skip_prefixes = (
        "merged_",       # old merged files
        "UTUAV_",        # too large, kept separate
    )
    # Old LLVIP registered manifests — replaced by LLVIP_unreg_all.json
    old_llvip_files = {"LLVIP_train.json", "LLVIP_test.json"}

    print("=" * 60)
    print("Merging manifests for DINO-MM pretraining v2")
    print(f"  manifest_dir: {manifest_dir}")
    print(f"  output:       {output_path}")
    print("=" * 60)

    all_samples = []
    dataset_counts = {}  # {display_name: {"count": N, "type": ...}}
    loaded_files = []
    skipped_files = []

    for fn in all_files:
        # Skip merged / UTUAV
        if any(fn.startswith(p) for p in skip_prefixes):
            skipped_files.append((fn, "skip prefix"))
            continue

        # Skip old LLVIP registered
        if fn in old_llvip_files:
            skipped_files.append((fn, "replaced by LLVIP_unreg"))
            continue

        path = os.path.join(manifest_dir, fn)
        try:
            samples, meta = load_manifest(path)
        except Exception as e:
            skipped_files.append((fn, f"load error: {e}"))
            continue

        if not samples:
            skipped_files.append((fn, "empty"))
            continue

        # Build display name
        dataset_name = meta.get("dataset", fn.replace(".json", ""))
        split = meta.get("split", "")
        display = f"{dataset_name}({split})" if split else dataset_name

        # Determine dominant type
        type_counter = Counter(s["type"] for s in samples)
        dominant_type = type_counter.most_common(1)[0][0]

        all_samples.extend(samples)
        dataset_counts[display] = {
            "count": len(samples),
            "type": dominant_type,
        }
        loaded_files.append(fn)

    # --- Statistics ---
    paired_count = sum(1 for s in all_samples if s["type"] == "paired")
    rgb_only_count = sum(1 for s in all_samples if s["type"] == "rgb_only")
    ir_only_count = sum(1 for s in all_samples if s["type"] == "ir_only")
    total = len(all_samples)

    # Per-dataset sample counts
    ds_counter = Counter(s.get("dataset", "unknown") for s in all_samples)

    print(f"\n--- Loaded {len(loaded_files)} manifests ---")
    for fn in loaded_files:
        print(f"  + {fn}")

    print(f"\n--- Skipped {len(skipped_files)} files ---")
    for fn, reason in skipped_files:
        print(f"  - {fn} ({reason})")

    print(f"\n--- Summary ---")
    print(f"  Total samples:  {total:,}")
    print(f"  Paired (RGB+IR): {paired_count:,}")
    print(f"  RGB-only:        {rgb_only_count:,}")
    print(f"  IR-only:         {ir_only_count:,}")
    print(f"\n--- Per-dataset counts ---")
    for ds, count in sorted(ds_counter.items(), key=lambda x: -x[1]):
        print(f"  {ds:30s}  {count:>8,}")

    # --- Write output ---
    merged = {
        "description": "Merged pretraining manifest v2 — all datasets excl. WebUAV-3M & UTUAV",
        "total_clean": total,
        "paired_count": paired_count,
        "rgb_only_count": rgb_only_count,
        "ir_only_count": ir_only_count,
        "dataset_counts": dataset_counts,
        "samples": all_samples,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f)
    print(f"\n-> Written to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

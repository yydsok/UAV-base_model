#!/usr/bin/env python3
"""
Clean manifest: verify all file paths exist on disk, remove broken entries.
Uses multiprocessing for speed on large manifests.
"""
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from collections import Counter

MANIFEST_IN  = "/root/autodl-tmp/train1/manifests/merged_full_pretrain_v2.json"
MANIFEST_OUT = "/root/autodl-tmp/train1/manifests/merged_full_pretrain_v2_clean.json"
NUM_WORKERS  = min(cpu_count(), 32)

def check_sample(sample):
    """Return (sample, is_valid, reason) — checks that all referenced files exist."""
    rgb = sample.get("rgb_path")
    ir  = sample.get("ir_path")
    stype = sample.get("type", "unknown")

    # Must have at least one path
    if not rgb and not ir:
        return (sample, False, "no_path")

    # Check RGB path if present
    if rgb and not os.path.isfile(rgb):
        return (sample, False, f"rgb_missing:{rgb}")

    # Check IR path if present
    if ir and not os.path.isfile(ir):
        return (sample, False, f"ir_missing:{ir}")

    # For paired type, both must exist
    if stype == "paired":
        if not rgb or not ir:
            return (sample, False, "paired_incomplete")

    return (sample, True, "ok")


def main():
    t0 = time.time()

    print(f"Loading manifest: {MANIFEST_IN}")
    with open(MANIFEST_IN) as f:
        data = json.load(f)

    samples = data["samples"]
    total = len(samples)
    print(f"Total samples: {total:,}")
    print(f"Checking file existence with {NUM_WORKERS} workers...")

    # Parallel file existence check
    good = []
    bad_reasons = Counter()
    bad_datasets = Counter()
    bad_count = 0

    with Pool(NUM_WORKERS) as pool:
        for i, (sample, valid, reason) in enumerate(pool.imap_unordered(check_sample, samples, chunksize=2000)):
            if valid:
                good.append(sample)
            else:
                bad_count += 1
                bad_reasons[reason.split(":")[0]] += 1
                bad_datasets[sample.get("dataset", "unknown")] += 1

            if (i + 1) % 500000 == 0:
                elapsed = time.time() - t0
                pct = (i + 1) / total * 100
                print(f"  Checked {i+1:,}/{total:,} ({pct:.1f}%) — "
                      f"removed {bad_count:,} so far — {elapsed:.0f}s")

    removed = total - len(good)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Valid: {len(good):,}")
    print(f"  Removed: {removed:,} ({removed/total*100:.4f}%)")

    if bad_reasons:
        print(f"\n  Removal reasons:")
        for r, c in bad_reasons.most_common():
            print(f"    {r}: {c:,}")
        print(f"\n  Affected datasets:")
        for d, c in bad_datasets.most_common():
            print(f"    {d}: {c:,}")

    # Rebuild type distribution
    type_dist = Counter(s.get("type", "unknown") for s in good)
    print(f"\n  Clean type distribution:")
    for t, c in type_dist.most_common():
        print(f"    {t}: {c:,}")

    # Write clean manifest
    data_out = {
        "description": data.get("description", "") + " [cleaned: removed broken paths]",
        "generated_by": "clean_manifest.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_total": total,
        "removed": removed,
        "total_samples": len(good),
        "total_datasets": len(set(s.get("dataset", "unknown") for s in good)),
        "samples": good,
    }

    print(f"\nWriting clean manifest: {MANIFEST_OUT}")
    with open(MANIFEST_OUT, "w") as f:
        json.dump(data_out, f, ensure_ascii=False)

    print(f"Done. {len(good):,} clean samples written.")
    return removed


if __name__ == "__main__":
    removed = main()
    sys.exit(0 if removed >= 0 else 1)

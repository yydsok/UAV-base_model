#!/usr/bin/env python3
"""
Fix broken paths in merged_train_all_v2.json and save as v3.

Most breakage comes from archive extraction creating duplicate nested dirs.
E.g. DroneRGBT/DroneRGBT/DroneRGBT/ → DroneRGBT/ on disk.

Strategy:
  1. Generic: remove consecutive duplicate directory components.
  2. Dataset-specific: handle AU-AIR, AeroScapes, SeaDronesSee edge cases.
  3. Verify every fixed path actually exists on disk.
"""

import json
import os
import sys
from collections import Counter

SRC = "/root/autodl-tmp/train1/manifests/merged_train_all_v2.json"
DST = "/root/autodl-tmp/train1/manifests/merged_train_all_v3.json"


def dedup_consecutive_dirs(path):
    """Remove consecutive duplicate directory components.

    E.g. /a/b/b/c → /a/b/c, /a/b/b/b/c → /a/b/c
    """
    parts = path.split('/')
    out = [parts[0]]
    for p in parts[1:]:
        if p != out[-1]:
            out.append(p)
    return '/'.join(out)


# Dataset-specific fixers (applied BEFORE generic dedup)
SPECIAL_FIXES = {
    'AU-AIR': lambda p: p.replace('/auair2019data/images/', '/auair2019data/'),
    'AeroScapes': lambda p: p.replace('/AeroScapes/aeroscapes/', '/AeroScapes/'),
}


def fix_seadronessee_objdet(path):
    """SeaDronesSee Object Detection: Uncompressed→Compressed, .png→.jpg"""
    path = path.replace('/Uncompressed Version/', '/Compressed Version/')
    if path.endswith('.png'):
        candidate = path[:-4] + '.jpg'
        if os.path.isfile(candidate):
            return candidate
    return path


def fix_path(path, dataset):
    if path is None:
        return path
    if os.path.isfile(path):
        return path

    # Apply dataset-specific fix first
    if dataset in SPECIAL_FIXES:
        path = SPECIAL_FIXES[dataset](path)
        if os.path.isfile(path):
            return path

    if dataset in ('SeaDronesSee',):
        path = fix_seadronessee_objdet(path)
        if os.path.isfile(path):
            return path

    # Generic: remove consecutive duplicate directory components
    fixed = dedup_consecutive_dirs(path)
    if os.path.isfile(fixed):
        return fixed

    # UAV123 special: after dedup, still has /data_seq/UAV123/ → /data_seq/
    if 'UAV123' in dataset:
        fixed2 = fixed.replace('/data_seq/UAV123/', '/data_seq/')
        if os.path.isfile(fixed2):
            return fixed2

    return path  # return original if nothing worked


def main():
    print(f"Loading: {SRC}")
    with open(SRC) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Total samples: {len(samples)}")

    fixed_count = Counter()  # dataset → count of paths fixed
    still_broken = Counter()  # dataset → count still broken
    broken_examples = {}  # dataset → first broken path (for debugging)

    for i, s in enumerate(samples):
        ds = s.get('dataset', '?')

        for key in ('rgb_path', 'ir_path'):
            orig = s.get(key)
            if orig is None:
                continue
            if os.path.isfile(orig):
                continue

            new_path = fix_path(orig, ds)
            if new_path != orig and os.path.isfile(new_path):
                s[key] = new_path
                fixed_count[ds] += 1
            else:
                still_broken[ds] += 1
                if ds not in broken_examples:
                    broken_examples[ds] = (i, key, orig)

        if (i + 1) % 200000 == 0:
            print(f"  processed {i+1}/{len(samples)}...")

    # Summary
    total_fixed = sum(fixed_count.values())
    total_broken = sum(still_broken.values())
    print(f"\n{'='*60}")
    print(f"Fixed {total_fixed} paths, {total_broken} still broken")
    print(f"{'='*60}")

    if fixed_count:
        print("\nFixed by dataset:")
        for ds, cnt in fixed_count.most_common():
            print(f"  {ds}: {cnt}")

    if still_broken:
        print("\nStill broken by dataset:")
        for ds, cnt in still_broken.most_common():
            idx, key, path = broken_examples[ds]
            print(f"  {ds}: {cnt}  (example idx={idx} {key}={path})")

    # Update metadata
    data['description'] = data.get('description', '') + ' [v3: paths fixed]'

    # Save
    print(f"\nSaving to: {DST}")
    with open(DST, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    print("Done.")


if __name__ == '__main__':
    main()

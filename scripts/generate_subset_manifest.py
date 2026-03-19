"""Generate a stratified subset manifest for quick validation.

Samples from each dataset with per-type ratio control.
Supports --max_rgb_ir_ratio to cap the only-RGB : only-IR ratio
(downsamples rgb_only datasets when needed).
"""

import json
import random
import argparse
from collections import defaultdict
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Full manifest JSON')
    parser.add_argument('--output', required=True, help='Output subset JSON')
    parser.add_argument('--ratio', default=0.1, type=float, help='Base sample ratio')
    parser.add_argument('--max_rgb_ir_ratio', default=1.5, type=float,
                        help='Max allowed rgb_only:ir_only ratio (default 1.5)')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Input: {len(samples)} samples")

    # Group by (dataset, type) for stratified sampling
    groups = defaultdict(list)
    for s in samples:
        key = (s.get('dataset', 'unknown'), s.get('type', 'unknown'))
        groups[key].append(s)

    random.seed(args.seed)

    # Phase 1: sample paired + ir_only + unknown at base ratio
    subset = []
    stats = defaultdict(lambda: defaultdict(int))

    ir_only_count = 0
    rgb_only_pool = []  # collect rgb_only groups for phase 2

    for (ds, typ), items in sorted(groups.items()):
        if typ == 'rgb_only':
            rgb_only_pool.append((ds, items))
            continue
        n = max(1, int(len(items) * args.ratio))
        sampled = random.sample(items, n)
        subset.extend(sampled)
        stats[ds][typ] = n
        if typ == 'ir_only':
            ir_only_count += n

    # Phase 2: sample rgb_only with cap
    rgb_only_budget = int(ir_only_count * args.max_rgb_ir_ratio)
    # Count total rgb_only at base ratio
    rgb_base_counts = {}
    rgb_base_total = 0
    for ds, items in rgb_only_pool:
        n = max(1, int(len(items) * args.ratio))
        rgb_base_counts[ds] = n
        rgb_base_total += n

    if rgb_base_total <= rgb_only_budget:
        # No downsampling needed
        scale = 1.0
    else:
        scale = rgb_only_budget / rgb_base_total

    rgb_only_total = 0
    for ds, items in rgb_only_pool:
        n_base = rgb_base_counts[ds]
        n = max(1, int(n_base * scale))
        n = min(n, len(items))
        sampled = random.sample(items, n)
        subset.extend(sampled)
        stats[ds]['rgb_only'] = n
        rgb_only_total += n

    random.shuffle(subset)

    # Count modality types
    type_counts = defaultdict(int)
    for s in subset:
        type_counts[s.get('type', 'unknown')] += 1

    ratio_str = (f"{type_counts.get('rgb_only',0)/max(1,type_counts.get('ir_only',1)):.2f}:1"
                 if type_counts.get('ir_only', 0) > 0 else "N/A")

    out = {
        'description': (f'Stratified subset (base {args.ratio*100:.0f}%, '
                        f'rgb:ir≤{args.max_rgb_ir_ratio}:1) for validation'),
        'generated_by': 'generate_subset_manifest.py',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_ratio': args.ratio,
        'max_rgb_ir_ratio': args.max_rgb_ir_ratio,
        'total_samples': len(subset),
        'total_datasets': len(set(s.get('dataset') for s in subset)),
        'samples': subset,
    }

    with open(args.output, 'w') as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"\nOutput: {len(subset)} samples ({len(subset)/len(samples)*100:.1f}%)")
    print(f"\nModality distribution:")
    for t in sorted(type_counts):
        print(f"  {t}: {type_counts[t]} ({type_counts[t]/len(subset)*100:.1f}%)")
    print(f"\nrgb_only:ir_only = {ratio_str}")
    if scale < 1.0:
        print(f"(rgb_only downsampled by {scale:.2f}x to meet ratio constraint)")
    print(f"\nPer-dataset breakdown:")
    for ds in sorted(stats):
        total = sum(stats[ds].values())
        types_str = ', '.join(f'{t}={v}' for t, v in sorted(stats[ds].items()))
        print(f"  {ds}: {total} ({types_str})")


if __name__ == '__main__':
    main()

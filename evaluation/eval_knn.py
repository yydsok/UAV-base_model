"""
KNN Classification Evaluation for DINO-MM.

Extracts CLS token features from DroneVehicle and performs weighted KNN classification.
Tests three modality conditions: RGB+IR, RGB-only, IR-only.

Reference: DINOv2 KNN evaluation protocol.

Usage:
    python eval_knn.py --checkpoint /path/to/checkpoint.pth
    python eval_knn.py --checkpoint /path/to/checkpoint.pth --k 10 20 --random_baseline
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, build_random_model, extract_features,
    DroneVehicleClassification, get_dataloader, get_num_classes,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


def knn_classify(train_features, train_labels, test_features, test_labels,
                 k=20, temperature=0.07, num_classes=None):
    """Weighted KNN classification.

    Args:
        train_features: [N_train, D] L2-normalized
        train_labels: [N_train]
        test_features: [N_test, D] L2-normalized
        test_labels: [N_test]
        k: number of neighbors
        temperature: softmax temperature for distance weighting
        num_classes: number of classes

    Returns:
        dict with top1 accuracy and per-class accuracy
    """
    if num_classes is None:
        num_classes = max(train_labels.max(), test_labels.max()).item() + 1

    # L2 normalize
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    # Compute all pairwise cosine similarities
    # Process in chunks to save memory
    chunk_size = 256
    n_test = test_features.shape[0]
    all_preds = []

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        chunk = test_features[start:end]  # [chunk, D]

        # Cosine similarity
        sim = chunk @ train_features.T  # [chunk, N_train]

        # Top-k
        topk_sim, topk_idx = sim.topk(k, dim=1)  # [chunk, k]
        topk_labels = train_labels[topk_idx]  # [chunk, k]

        # Weighted vote
        weights = (topk_sim / temperature).exp()  # [chunk, k]

        # Accumulate votes per class
        votes = torch.zeros(end - start, num_classes)
        for c in range(num_classes):
            mask = (topk_labels == c).float()  # [chunk, k]
            votes[:, c] = (weights * mask).sum(dim=1)

        preds = votes.argmax(dim=1)
        all_preds.append(preds)

    predictions = torch.cat(all_preds)

    # Metrics
    correct = (predictions == test_labels).float()
    top1 = correct.mean().item() * 100

    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = (test_labels == c)
        if mask.sum() > 0:
            per_class[c] = correct[mask].mean().item() * 100

    return {
        'top1': top1,
        'per_class': per_class,
        'predictions': predictions,
    }


def eval_knn(model, dataset_root=DRONEVEHICLE_ROOT, k_values=[10, 20],
             batch_size=64, device='cuda', modality_mode='both', num_workers=4):
    """Run KNN evaluation for one modality mode.

    Returns:
        dict with results for each k
    """
    # Load train and val datasets
    train_dataset = DroneVehicleClassification(dataset_root, split='train')
    val_dataset = DroneVehicleClassification(dataset_root, split='val')

    train_loader = get_dataloader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)

    print(f"\n  Extracting train features ({modality_mode})...")
    train_features, train_labels = extract_features(
        model, train_loader, modality_mode=modality_mode, device=device)

    print(f"  Extracting val features ({modality_mode})...")
    val_features, val_labels = extract_features(
        model, val_loader, modality_mode=modality_mode, device=device)

    num_classes = get_num_classes()

    results = {}
    for k in k_values:
        res = knn_classify(
            train_features, train_labels,
            val_features, val_labels,
            k=k, num_classes=num_classes,
        )
        results[k] = res
        print(f"  k={k}: Top-1 = {res['top1']:.2f}%")
        for c, acc in sorted(res['per_class'].items()):
            class_name = train_dataset.class_names.get(c, str(c))
            print(f"    {class_name}: {acc:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='KNN Classification Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', nargs='+', type=int, default=[10, 20])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random_baseline', action='store_true',
                        help='Also evaluate random initialization')
    args = parser.parse_args()

    model, train_args = build_model(args.checkpoint, device=args.device)

    modality_modes = ['both', 'rgb_only', 'ir_only']
    mode_labels = {'both': 'RGB+IR', 'rgb_only': 'RGB Only', 'ir_only': 'IR Only'}

    all_results = {}

    for mode in modality_modes:
        print(f"\n{'='*60}")
        print(f"KNN Evaluation — {mode_labels[mode]}")
        print(f"{'='*60}")
        all_results[mode] = eval_knn(
            model, args.dataset_root, args.k, args.batch_size,
            args.device, mode, args.num_workers,
        )

    # Random baseline
    if args.random_baseline:
        print(f"\n{'='*60}")
        print(f"Random Baseline")
        print(f"{'='*60}")
        random_model = build_random_model(device=args.device)
        all_results['random'] = eval_knn(
            random_model, args.dataset_root, args.k, args.batch_size,
            args.device, 'both', args.num_workers,
        )

    # Summary table
    print(f"\n{'='*60}")
    print(f"KNN Classification Summary")
    print(f"{'='*60}")
    header = f"{'Mode':<15}"
    for k in args.k:
        header += f" | k={k:>3} Top-1"
    print(header)
    print('-' * len(header))

    for mode_key in list(all_results.keys()):
        label = mode_labels.get(mode_key, mode_key)
        row = f"{label:<15}"
        for k in args.k:
            if k in all_results[mode_key]:
                row += f" | {all_results[mode_key][k]['top1']:>9.2f}%"
            else:
                row += f" |       N/A"
        print(row)


if __name__ == '__main__':
    main()

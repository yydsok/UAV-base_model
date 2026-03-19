"""
Cross-Modal Retrieval Evaluation for DINO-MM.

Given paired RGB+IR images, evaluates how well the model aligns RGB and IR
representations by performing RGB→IR and IR→RGB retrieval.

Metrics: Rank-1, Rank-5, Rank-10 accuracy, mAP.

Usage:
    python eval_crossmodal_retrieval.py --checkpoint /path/to/checkpoint.pth
    python eval_crossmodal_retrieval.py --checkpoint /path/to/checkpoint.pth --dataset llvip
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, build_random_model,
    LLVIPPaired, DroneVehiclePaired, get_dataloader, infer_modality_masks,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'
LLVIP_ROOT = '/root/autodl-tmp/data/LLVIP/registered/'


@torch.no_grad()
def extract_paired_features(model, dataloader, device='cuda', max_samples=None,
                             verbose=True, feature_type='patch_mean'):
    """Extract features for paired RGB and IR images.

    The dataloader yields (rgb_4ch, ir_4ch, idx) where rgb_4ch has IR zeroed
    and ir_4ch has RGB zeroed.

    Args:
        feature_type: 'cls' for CLS token, 'patch_mean' for mean of patch tokens.

    Returns:
        rgb_features: [N, D]
        ir_features: [N, D]
    """
    model.eval()
    all_rgb_feat = []
    all_ir_feat = []
    n_collected = 0

    for batch_idx, (rgb_tensor, ir_tensor, indices) in enumerate(dataloader):
        rgb_tensor = rgb_tensor.to(device)
        ir_tensor = ir_tensor.to(device)
        rgb_masks = infer_modality_masks(rgb_tensor)
        ir_masks = infer_modality_masks(ir_tensor)

        rgb_feat = model(rgb_tensor, return_all_tokens=True, modality_masks=rgb_masks)  # [B, N+1, D]
        ir_feat = model(ir_tensor, return_all_tokens=True, modality_masks=ir_masks)      # [B, N+1, D]

        if feature_type == 'patch_mean':
            rgb_feat = rgb_feat[:, 1:].mean(dim=1)  # [B, D]
            ir_feat = ir_feat[:, 1:].mean(dim=1)    # [B, D]
        else:  # cls
            rgb_feat = rgb_feat[:, 0]  # [B, D]
            ir_feat = ir_feat[:, 0]    # [B, D]

        all_rgb_feat.append(rgb_feat.cpu())
        all_ir_feat.append(ir_feat.cpu())
        n_collected += rgb_tensor.shape[0]

        if verbose and (batch_idx + 1) % 50 == 0:
            print(f"  Extracted {n_collected} pairs...")

        if max_samples and n_collected >= max_samples:
            break

    rgb_features = torch.cat(all_rgb_feat, dim=0)
    ir_features = torch.cat(all_ir_feat, dim=0)

    if max_samples:
        rgb_features = rgb_features[:max_samples]
        ir_features = ir_features[:max_samples]

    if verbose:
        print(f"  Total: {rgb_features.shape[0]} pairs, dim={rgb_features.shape[1]}")

    return rgb_features, ir_features


def compute_retrieval_metrics(query_features, gallery_features, ks=[1, 5, 10]):
    """Compute retrieval metrics.

    Ground truth: query[i] should match gallery[i] (paired data).

    Args:
        query_features: [N, D]
        gallery_features: [N, D]
        ks: list of k values for Rank-k accuracy

    Returns:
        dict with rank_k accuracies and mAP
    """
    # L2 normalize
    query_features = F.normalize(query_features, dim=1)
    gallery_features = F.normalize(gallery_features, dim=1)

    N = query_features.shape[0]

    # Compute similarity matrix
    sim = query_features @ gallery_features.T  # [N, N]

    # Rank gallery items for each query
    _, ranking = sim.sort(dim=1, descending=True)  # [N, N]

    # Ground truth: diagonal (query i matches gallery i)
    gt = torch.arange(N).unsqueeze(1)  # [N, 1]

    # Find rank of correct match for each query
    correct_ranks = (ranking == gt).nonzero(as_tuple=False)[:, 1]  # [N]

    metrics = {}

    # Rank-k accuracy
    for k in ks:
        rank_k = (correct_ranks < k).float().mean().item() * 100
        metrics[f'rank_{k}'] = rank_k

    # mAP (since there's exactly 1 relevant item per query, AP = 1/(rank+1))
    ap = 1.0 / (correct_ranks.float() + 1)
    metrics['mAP'] = ap.mean().item() * 100

    # Median rank
    metrics['median_rank'] = correct_ranks.float().median().item()

    return metrics


def visualize_retrieval(query_features, gallery_features,
                        query_dataset, gallery_dataset,
                        output_path, direction='rgb2ir',
                        num_queries=5, top_k=5):
    """Visualize top-k retrieval results.

    Note: This is a simplified version that just shows the retrieval ranking
    without loading actual images (to avoid heavy I/O).
    """
    query_features = F.normalize(query_features, dim=1)
    gallery_features = F.normalize(gallery_features, dim=1)

    N = query_features.shape[0]
    sim = query_features @ gallery_features.T

    _, ranking = sim.sort(dim=1, descending=True)

    # Print retrieval examples
    print(f"\n  Top-{top_k} retrieval examples ({direction}):")
    indices = np.random.choice(N, min(num_queries, N), replace=False)

    for qi in indices:
        retrieved = ranking[qi, :top_k].tolist()
        correct = qi in retrieved
        mark = 'O' if correct else 'X'
        print(f"    Query {qi} → Retrieved: {retrieved}  [{mark}]")


def eval_crossmodal_retrieval(ckpt_path, dataset_name='dronevehicle',
                              batch_size=64, device='cuda',
                              max_samples=None, output_dir=None,
                              feature_type='patch_mean'):
    """Run full cross-modal retrieval evaluation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'retrieval')
    os.makedirs(output_dir, exist_ok=True)

    model, args = build_model(ckpt_path, device=device)

    # Build dataset
    if dataset_name == 'llvip':
        dataset = LLVIPPaired(LLVIP_ROOT, split='test', max_samples=max_samples)
    elif dataset_name == 'dronevehicle':
        dataset = DroneVehiclePaired(DRONEVEHICLE_ROOT, split='val', max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    print("\nExtracting paired features...")
    rgb_features, ir_features = extract_paired_features(
        model, dataloader, device=device, max_samples=max_samples,
        feature_type=feature_type)

    # RGB → IR retrieval
    print("\n--- RGB → IR Retrieval ---")
    rgb2ir = compute_retrieval_metrics(rgb_features, ir_features)
    for k, v in sorted(rgb2ir.items()):
        print(f"  {k}: {v:.2f}")

    # IR → RGB retrieval
    print("\n--- IR → RGB Retrieval ---")
    ir2rgb = compute_retrieval_metrics(ir_features, rgb_features)
    for k, v in sorted(ir2rgb.items()):
        print(f"  {k}: {v:.2f}")

    # Visualize examples
    visualize_retrieval(rgb_features, ir_features, dataset, dataset,
                        os.path.join(output_dir, f'{dataset_name}_rgb2ir.png'),
                        direction='rgb2ir')
    visualize_retrieval(ir_features, rgb_features, dataset, dataset,
                        os.path.join(output_dir, f'{dataset_name}_ir2rgb.png'),
                        direction='ir2rgb')

    # Plot similarity matrix heatmap (subsample for visualization)
    n_vis = min(100, rgb_features.shape[0])
    rgb_sub = F.normalize(rgb_features[:n_vis], dim=1)
    ir_sub = F.normalize(ir_features[:n_vis], dim=1)
    sim_mat = (rgb_sub @ ir_sub.T).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(sim_mat, cmap='viridis', aspect='auto')
    ax.set_xlabel('IR Gallery Index', fontsize=12)
    ax.set_ylabel('RGB Query Index', fontsize=12)
    ax.set_title(f'Cross-Modal Similarity Matrix ({dataset_name})', fontsize=13)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_sim_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved similarity matrix to {output_dir}")

    return {'rgb2ir': rgb2ir, 'ir2rgb': ir2rgb}


def main():
    parser = argparse.ArgumentParser(description='Cross-Modal Retrieval Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dronevehicle',
                        choices=['dronevehicle', 'llvip'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of pairs (for speed)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--random_baseline', action='store_true')
    parser.add_argument('--feature_type', type=str, default='patch_mean',
                        choices=['cls', 'patch_mean'],
                        help='Feature extraction mode: cls token or patch token mean')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Cross-Modal Retrieval — {args.dataset}")
    print(f"{'='*60}")

    results = eval_crossmodal_retrieval(
        args.checkpoint, args.dataset, args.batch_size,
        args.device, args.max_samples, args.output_dir,
        feature_type=args.feature_type,
    )

    if args.random_baseline:
        print(f"\n{'='*60}")
        print(f"Random Baseline")
        print(f"{'='*60}")

        random_model = build_random_model(device=args.device)

        if args.dataset == 'llvip':
            dataset = LLVIPPaired(LLVIP_ROOT, split='test', max_samples=args.max_samples)
        else:
            dataset = DroneVehiclePaired(DRONEVEHICLE_ROOT, split='val', max_samples=args.max_samples)

        dataloader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=4)

        rgb_feat, ir_feat = extract_paired_features(
            random_model, dataloader, device=args.device, max_samples=args.max_samples)

        print("\n--- Random RGB → IR ---")
        rand_rgb2ir = compute_retrieval_metrics(rgb_feat, ir_feat)
        for k, v in sorted(rand_rgb2ir.items()):
            print(f"  {k}: {v:.2f}")

        print("\n--- Random IR → RGB ---")
        rand_ir2rgb = compute_retrieval_metrics(ir_feat, rgb_feat)
        for k, v in sorted(rand_ir2rgb.items()):
            print(f"  {k}: {v:.2f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Cross-Modal Retrieval Summary ({args.dataset})")
    print(f"{'='*60}")
    print(f"{'Direction':<15} | {'Rank-1':>8} | {'Rank-5':>8} | {'Rank-10':>8} | {'mAP':>8}")
    print(f"{'-'*15} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for direction, res in [('RGB → IR', results['rgb2ir']), ('IR → RGB', results['ir2rgb'])]:
        print(f"{direction:<15} | {res['rank_1']:>7.2f}% | {res['rank_5']:>7.2f}% | "
              f"{res['rank_10']:>7.2f}% | {res['mAP']:>7.2f}%")


if __name__ == '__main__':
    main()

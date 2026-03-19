"""
RankMe: Evaluate the effective rank of pretrained representations.

Computes the effective rank (exp of Shannon entropy of normalized singular values)
of the CLS token feature matrix. Higher rank → more expressive representations.

Reference: "RankMe: Assessing the downstream performance of pretrained
self-supervised representations by their rank" (Garrido et al., 2023)

Usage:
    python eval_rankme.py --checkpoint /path/to/checkpoint.pth [--num_samples 2000]
    python eval_rankme.py --checkpoints cp1.pth cp2.pth cp3.pth  # compare multiple
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, build_random_model, extract_features,
    DroneVehicleClassification, get_dataloader,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


def compute_rankme(features):
    """Compute the effective rank (RankMe) of a feature matrix.

    Args:
        features: [N, D] tensor or numpy array

    Returns:
        effective_rank: float
        singular_values: numpy array of singular values
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    # Center features
    features = features - features.mean(axis=0, keepdims=True)

    # SVD
    _, S, _ = np.linalg.svd(features, full_matrices=False)

    # Normalize singular values to form a probability distribution
    S_norm = S / S.sum()

    # Remove zeros to avoid log(0)
    S_norm = S_norm[S_norm > 0]

    # Shannon entropy
    entropy = -np.sum(S_norm * np.log(S_norm))

    # Effective rank
    effective_rank = np.exp(entropy)

    return effective_rank, S


def eval_rankme_single(ckpt_path, num_samples=2000, batch_size=64,
                       device='cuda', dataset_root=DRONEVEHICLE_ROOT):
    """Evaluate RankMe for a single checkpoint.

    Returns:
        dict with rankme, embed_dim, ratio, epoch
    """
    model, args = build_model(ckpt_path, device=device)
    embed_dim = model.embed_dim

    dataset = DroneVehicleClassification(dataset_root, split='train')
    dataloader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    print(f"\nExtracting features (max {num_samples} samples)...")
    features, _ = extract_features(
        model, dataloader, modality_mode='both',
        device=device, max_samples=num_samples,
    )

    rank, singular_values = compute_rankme(features)
    ratio = rank / embed_dim

    epoch = None
    if isinstance(ckpt_path, str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', None)

    print(f"\n{'='*50}")
    print(f"RankMe Results:")
    print(f"  Effective Rank: {rank:.1f}")
    print(f"  Embed Dim:      {embed_dim}")
    print(f"  Ratio:          {ratio:.4f}")
    if epoch is not None:
        print(f"  Epoch:          {epoch}")
    print(f"{'='*50}")

    return {
        'rankme': rank,
        'embed_dim': embed_dim,
        'ratio': ratio,
        'epoch': epoch,
        'singular_values': singular_values,
    }


def eval_rankme_random_baseline(num_samples=2000,
                                batch_size=64, device='cuda',
                                dataset_root=DRONEVEHICLE_ROOT):
    """Evaluate RankMe for a randomly initialized model."""
    print("\n--- Random baseline ---")
    model = build_random_model(device=device)
    embed_dim = model.embed_dim

    dataset = DroneVehicleClassification(dataset_root, split='train')
    dataloader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    features, _ = extract_features(
        model, dataloader, modality_mode='both',
        device=device, max_samples=num_samples,
    )

    rank, sv = compute_rankme(features)
    ratio = rank / embed_dim
    print(f"  Random RankMe: {rank:.1f} / {embed_dim} = {ratio:.4f}")
    return {'rankme': rank, 'embed_dim': embed_dim, 'ratio': ratio, 'epoch': -1}


def plot_rankme_curve(results, output_path):
    """Plot RankMe vs epoch for multiple checkpoints.

    Args:
        results: list of dicts from eval_rankme_single
        output_path: path to save the figure
    """
    epochs = [r['epoch'] for r in results if r['epoch'] is not None]
    ranks = [r['rankme'] for r in results if r['epoch'] is not None]

    if not epochs:
        print("No epoch information available, skipping plot")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    ax1.plot(epochs, ranks, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Effective Rank (RankMe)', fontsize=12)
    ax1.set_title('RankMe vs Training Epoch', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add ratio on right y-axis
    embed_dim = results[0]['embed_dim']
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Ratio (rank / {embed_dim})', fontsize=12)
    ax2.set_ylim(ax1.get_ylim()[0] / embed_dim, ax1.get_ylim()[1] / embed_dim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RankMe curve to {output_path}")


def plot_singular_values(singular_values, output_path, title='Singular Value Spectrum'):
    """Plot the singular value spectrum."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(singular_values, 'b-', linewidth=1)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title(f'{title} (linear)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(singular_values, 'b-', linewidth=1)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Singular Value (log)', fontsize=12)
    ax2.set_title(f'{title} (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved singular value spectrum to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RankMe evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint')
    parser.add_argument('--checkpoints', nargs='+', type=str, default=None,
                        help='Paths to multiple checkpoints (for comparison)')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of samples for feature extraction')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'outputs', 'rankme'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random_baseline', action='store_true',
                        help='Also evaluate random initialization baseline')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint:
        result = eval_rankme_single(
            args.checkpoint, args.num_samples, args.batch_size,
            args.device, args.dataset_root,
        )
        # Plot singular values
        plot_singular_values(
            result['singular_values'],
            os.path.join(args.output_dir, 'singular_values.png'),
        )

        if args.random_baseline:
            baseline = eval_rankme_random_baseline(
                device=args.device, dataset_root=args.dataset_root,
                num_samples=args.num_samples, batch_size=args.batch_size,
            )
            print(f"\n  Pretrained / Random ratio: {result['rankme'] / baseline['rankme']:.2f}x")

    elif args.checkpoints:
        results = []
        for ckpt in args.checkpoints:
            print(f"\n{'='*60}")
            print(f"Processing: {ckpt}")
            r = eval_rankme_single(
                ckpt, args.num_samples, args.batch_size,
                args.device, args.dataset_root,
            )
            results.append(r)

        # Plot curve
        plot_rankme_curve(results, os.path.join(args.output_dir, 'rankme_curve.png'))

        # Summary table
        print(f"\n{'='*60}")
        print(f"{'Epoch':>6} | {'RankMe':>10} | {'Ratio':>8}")
        print(f"{'-'*6} | {'-'*10} | {'-'*8}")
        for r in results:
            ep = r['epoch'] if r['epoch'] is not None else '?'
            print(f"{ep:>6} | {r['rankme']:>10.1f} | {r['ratio']:>8.4f}")
    else:
        parser.error("Must specify --checkpoint or --checkpoints")


if __name__ == '__main__':
    main()

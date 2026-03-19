"""
t-SNE Visualization for DINO-MM.

Extracts CLS features from DroneVehicle and visualizes them in 2D using t-SNE.
Produces two plots:
  1. Colored by class (car/truck) — good model: clusters by class
  2. Colored by modality (RGB+IR / RGB-only / IR-only) — good model: no modality clusters

Usage:
    python eval_tsne.py --checkpoint /path/to/checkpoint.pth
    python eval_tsne.py --checkpoint /path/to/checkpoint.pth --num_samples 1000
"""

import os
import sys
import argparse

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, build_random_model, extract_features,
    DroneVehicleClassification, get_dataloader,
    DRONEVEHICLE_CLASS_MAP,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


def run_tsne(features, perplexity=30, n_iter=1000, random_state=42):
    """Run t-SNE on features.

    Args:
        features: [N, D] numpy array
        perplexity: t-SNE perplexity
        n_iter: number of iterations

    Returns:
        embeddings: [N, 2] numpy array
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    embeddings = tsne.fit_transform(features)
    return embeddings


def plot_tsne_by_class(embeddings, labels, class_names, output_path, title=''):
    """Plot t-SNE colored by class label."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    unique_labels = sorted(set(labels.tolist()))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names.get(label, f'Class {label}')
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[colors[i]], label=name,
            alpha=0.6, s=15, edgecolors='none',
        )

    ax.legend(fontsize=11, markerscale=2)
    ax.set_title(f'{title}t-SNE by Class', fontsize=14)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tsne_by_modality(embeddings_dict, output_path, title=''):
    """Plot t-SNE colored by modality condition.

    Args:
        embeddings_dict: {'both': [N,2], 'rgb_only': [N,2], 'ir_only': [N,2]}
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    mode_colors = {
        'both': '#2196F3',      # blue
        'rgb_only': '#F44336',  # red
        'ir_only': '#4CAF50',   # green
    }
    mode_labels = {
        'both': 'RGB+IR',
        'rgb_only': 'RGB Only',
        'ir_only': 'IR Only',
    }

    for mode, emb in embeddings_dict.items():
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=mode_colors[mode], label=mode_labels[mode],
            alpha=0.5, s=15, edgecolors='none',
        )

    ax.legend(fontsize=11, markerscale=2)
    ax.set_title(f'{title}t-SNE by Modality', fontsize=14)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tsne_combined(embeddings_dict, labels_dict, class_names, output_path, title=''):
    """Plot 2x3 grid: top row by class, bottom row by modality, columns = modality modes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    modes = ['both', 'rgb_only', 'ir_only']
    mode_labels = {'both': 'RGB+IR', 'rgb_only': 'RGB Only', 'ir_only': 'IR Only'}

    unique_labels = sorted(set(labels_dict['both'].tolist()))
    class_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))
    mode_colors = {'both': '#2196F3', 'rgb_only': '#F44336', 'ir_only': '#4CAF50'}

    for col, mode in enumerate(modes):
        emb = embeddings_dict[mode]
        labels = labels_dict[mode]

        # Top row: by class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names.get(label, f'Class {label}')
            axes[0, col].scatter(
                emb[mask, 0], emb[mask, 1],
                c=[class_colors[i]], label=name,
                alpha=0.6, s=12, edgecolors='none',
            )
        axes[0, col].set_title(f'{mode_labels[mode]} — by Class', fontsize=11)
        axes[0, col].legend(fontsize=9, markerscale=2)
        axes[0, col].grid(True, alpha=0.2)

        # Bottom row: this modality vs others
        axes[1, col].scatter(
            emb[:, 0], emb[:, 1],
            c=mode_colors[mode], label=mode_labels[mode],
            alpha=0.5, s=12, edgecolors='none',
        )
        axes[1, col].set_title(f'{mode_labels[mode]} — Features', fontsize=11)
        axes[1, col].legend(fontsize=9, markerscale=2)
        axes[1, col].grid(True, alpha=0.2)

    fig.suptitle(f'{title}t-SNE Feature Visualization', fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def eval_tsne(ckpt_path, num_samples=1000, batch_size=64, device='cuda',
              dataset_root=DRONEVEHICLE_ROOT, output_dir=None, perplexity=30):
    """Run full t-SNE evaluation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'tsne')
    os.makedirs(output_dir, exist_ok=True)

    model, args = build_model(ckpt_path, device=device)

    dataset = DroneVehicleClassification(dataset_root, split='train')
    dataloader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    class_names = {v: k for k, v in DRONEVEHICLE_CLASS_MAP.items()}

    modality_modes = ['both', 'rgb_only', 'ir_only']
    features_dict = {}
    labels_dict = {}

    for mode in modality_modes:
        print(f"\nExtracting features ({mode})...")
        feats, labels = extract_features(
            model, dataloader, modality_mode=mode,
            device=device, max_samples=num_samples,
        )
        features_dict[mode] = feats
        labels_dict[mode] = labels

    # Run t-SNE for each modality mode separately
    embeddings_dict = {}
    for mode in modality_modes:
        print(f"\nRunning t-SNE ({mode})...")
        emb = run_tsne(features_dict[mode], perplexity=perplexity)
        embeddings_dict[mode] = emb

    # Plot per-mode by class
    for mode in modality_modes:
        plot_tsne_by_class(
            embeddings_dict[mode], labels_dict[mode], class_names,
            os.path.join(output_dir, f'tsne_{mode}_by_class.png'),
            title=f'[{mode}] ',
        )

    # Joint t-SNE: combine all features and run once
    print("\nRunning joint t-SNE (all modalities together)...")
    all_features = torch.cat([features_dict[m] for m in modality_modes], dim=0)
    all_labels = torch.cat([labels_dict[m] for m in modality_modes], dim=0)
    all_embeddings = run_tsne(all_features, perplexity=perplexity)

    n_per_mode = features_dict['both'].shape[0]
    modality_ids = np.concatenate([
        np.full(n_per_mode, i) for i, _ in enumerate(modality_modes)
    ])

    # Plot by class (joint)
    plot_tsne_by_class(
        all_embeddings, all_labels, class_names,
        os.path.join(output_dir, 'tsne_joint_by_class.png'),
        title='[Joint] ',
    )

    # Plot by modality (joint)
    joint_by_modality = {}
    for i, mode in enumerate(modality_modes):
        mask = modality_ids == i
        joint_by_modality[mode] = all_embeddings[mask]

    plot_tsne_by_modality(
        joint_by_modality,
        os.path.join(output_dir, 'tsne_joint_by_modality.png'),
        title='[Joint] ',
    )

    # Combined 2x3 grid
    plot_tsne_combined(
        embeddings_dict, labels_dict, class_names,
        os.path.join(output_dir, 'tsne_combined.png'),
    )

    print(f"\nAll t-SNE plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    eval_tsne(
        args.checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        perplexity=args.perplexity,
    )


if __name__ == '__main__':
    main()

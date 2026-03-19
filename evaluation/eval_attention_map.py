"""
Attention Map Visualization for DINO-MM.

Visualizes CLS→patch attention maps from the last transformer block.
Shows what the model attends to in RGB+IR, RGB-only, and IR-only conditions.

Usage:
    python eval_attention_map.py --checkpoint /path/to/checkpoint.pth
    python eval_attention_map.py --checkpoint /path/to/checkpoint.pth --image_paths img1.jpg img2.jpg
"""

import os
import sys
import argparse
import random

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, extract_attention_weights,
    DroneVehicleClassification, get_eval_transform,
    RGBIR_MEAN, RGBIR_STD,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


def load_single_image(rgb_path, ir_path=None, img_size=224):
    """Load a single RGB+IR image pair and return 4-channel tensor.

    Args:
        rgb_path: path to RGB image
        ir_path: path to IR image (optional)
        img_size: resize to this size

    Returns:
        tensor: [1, 4, H, W]
        rgb_display: [H, W, 3] uint8 for visualization
        ir_display: [H, W] uint8 for visualization
    """
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise IOError(f"Cannot read: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if ir_path and os.path.exists(ir_path):
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
    else:
        ir = np.zeros(rgb.shape[:2], dtype=np.uint8)

    # Resize for display
    rgb_display = cv2.resize(rgb, (img_size, img_size))
    ir_display = cv2.resize(ir, (img_size, img_size))

    # Build 4ch
    if ir.ndim == 2:
        ir = ir[:, :, np.newaxis]
    img_4ch = np.concatenate([rgb, ir], axis=2)

    transform = get_eval_transform(img_size=img_size, crop_size=img_size)
    tensor = transform(img_4ch).unsqueeze(0)

    return tensor, rgb_display, ir_display


def visualize_attention(attn_weights, rgb_display, ir_display, patch_size,
                        output_path, title_prefix='', max_heads=6):
    """Visualize CLS→patch attention maps.

    Args:
        attn_weights: [1, num_heads, N+1, N+1] attention weights
        rgb_display: [H, W, 3] uint8
        ir_display: [H, W] uint8
        patch_size: int
        output_path: path to save figure
        title_prefix: prefix for figure title
        max_heads: maximum number of heads to display
    """
    attn = attn_weights[0]  # [num_heads, N+1, N+1]
    num_heads = attn.shape[0]

    # CLS token attention to patch tokens
    cls_attn = attn[:, 0, 1:]  # [num_heads, N]

    # Reshape to spatial grid
    h = w = int(cls_attn.shape[1] ** 0.5)
    cls_attn = cls_attn.reshape(num_heads, h, w)

    # Select heads to display
    n_display = min(num_heads, max_heads)

    img_h, img_w = rgb_display.shape[:2]

    # Layout: 2 rows (RGB + IR display, then attention heads)
    fig, axes = plt.subplots(2, n_display + 1, figsize=(3 * (n_display + 1), 7))

    # First row: original images + mean attention
    axes[0, 0].imshow(rgb_display)
    axes[0, 0].set_title('RGB', fontsize=10)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(ir_display, cmap='inferno')
    axes[1, 0].set_title('IR', fontsize=10)
    axes[1, 0].axis('off')

    # Mean attention
    mean_attn = cls_attn.mean(0).numpy()
    mean_attn_resized = cv2.resize(mean_attn, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    mean_attn_resized = (mean_attn_resized - mean_attn_resized.min()) / (mean_attn_resized.max() - mean_attn_resized.min() + 1e-8)

    # Overlay on RGB
    axes[0, 1].imshow(rgb_display)
    axes[0, 1].imshow(mean_attn_resized, alpha=0.6, cmap='jet')
    axes[0, 1].set_title('Mean Attn\n(on RGB)', fontsize=9)
    axes[0, 1].axis('off')

    # Overlay on IR
    axes[1, 1].imshow(ir_display, cmap='gray')
    axes[1, 1].imshow(mean_attn_resized, alpha=0.6, cmap='jet')
    axes[1, 1].set_title('Mean Attn\n(on IR)', fontsize=9)
    axes[1, 1].axis('off')

    # Per-head attention
    for i in range(min(n_display - 1, num_heads)):
        head_attn = cls_attn[i].numpy()
        head_attn_resized = cv2.resize(head_attn, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        head_attn_resized = (head_attn_resized - head_attn_resized.min()) / (head_attn_resized.max() - head_attn_resized.min() + 1e-8)

        axes[0, i + 2].imshow(rgb_display)
        axes[0, i + 2].imshow(head_attn_resized, alpha=0.6, cmap='jet')
        axes[0, i + 2].set_title(f'Head {i}', fontsize=9)
        axes[0, i + 2].axis('off')

        axes[1, i + 2].imshow(head_attn_resized, cmap='viridis')
        axes[1, i + 2].set_title(f'Head {i}', fontsize=9)
        axes[1, i + 2].axis('off')

    # Hide unused axes
    for row in range(2):
        for col in range(min(n_display - 1, num_heads) + 2, n_display + 1):
            axes[row, col].axis('off')

    fig.suptitle(f'{title_prefix}CLS→Patch Attention', fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def eval_attention_maps(ckpt_path, num_images=5, dataset_root=DRONEVEHICLE_ROOT,
                        output_dir=None, device='cuda', img_size=224):
    """Generate attention map visualizations for multiple images and modality modes.

    Args:
        ckpt_path: path to checkpoint
        num_images: number of images to visualize
        dataset_root: DroneVehicle root
        output_dir: output directory
        device: torch device
        img_size: image size for visualization
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'attention_maps')
    os.makedirs(output_dir, exist_ok=True)

    model, args = build_model(ckpt_path, device=device)
    patch_size = model.patch_embed.patch_size

    # Load dataset to get image paths
    dataset = DroneVehicleClassification(dataset_root, split='val')

    # Randomly select images
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    modality_modes = ['both', 'rgb_only', 'ir_only']
    mode_labels = {'both': 'RGB+IR', 'rgb_only': 'RGB Only', 'ir_only': 'IR Only'}

    for img_idx in indices:
        sample = dataset.samples[img_idx]
        stem = sample['stem']

        # Load raw images for display
        rgb_path = os.path.join(dataset.rgb_dir, f'{stem}.jpg')
        ir_path = os.path.join(dataset.ir_dir, f'{stem}.jpg')

        tensor, rgb_display, ir_display = load_single_image(
            rgb_path, ir_path, img_size=img_size)

        for mode in modality_modes:
            # Extract attention
            attn = extract_attention_weights(model, tensor, modality_mode=mode, device=device)

            out_path = os.path.join(output_dir, f'{stem}_{mode}.png')
            visualize_attention(
                attn, rgb_display, ir_display, patch_size,
                output_path=out_path,
                title_prefix=f'[{mode_labels[mode]}] ',
            )

    # Also create a comparison figure for one image
    if indices:
        sample = dataset.samples[indices[0]]
        stem = sample['stem']
        rgb_path = os.path.join(dataset.rgb_dir, f'{stem}.jpg')
        ir_path = os.path.join(dataset.ir_dir, f'{stem}.jpg')
        tensor, rgb_display, ir_display = load_single_image(rgb_path, ir_path, img_size=img_size)

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for row, mode in enumerate(modality_modes):
            attn = extract_attention_weights(model, tensor, modality_mode=mode, device=device)
            cls_attn = attn[0, :, 0, 1:]  # [heads, N]
            h = w = int(cls_attn.shape[1] ** 0.5)

            # Mean attention map
            mean_attn = cls_attn.mean(0).reshape(h, w).numpy()
            mean_attn = cv2.resize(mean_attn, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min() + 1e-8)

            # Column 0: RGB
            axes[row, 0].imshow(rgb_display)
            axes[row, 0].set_ylabel(mode_labels[mode], fontsize=12, fontweight='bold')
            if row == 0:
                axes[row, 0].set_title('RGB Input', fontsize=11)
            axes[row, 0].axis('off')

            # Column 1: Attn overlay on RGB
            axes[row, 1].imshow(rgb_display)
            axes[row, 1].imshow(mean_attn, alpha=0.6, cmap='jet')
            if row == 0:
                axes[row, 1].set_title('Attention on RGB', fontsize=11)
            axes[row, 1].axis('off')

            # Column 2: Attn overlay on IR
            axes[row, 2].imshow(ir_display, cmap='gray')
            axes[row, 2].imshow(mean_attn, alpha=0.6, cmap='jet')
            if row == 0:
                axes[row, 2].set_title('Attention on IR', fontsize=11)
            axes[row, 2].axis('off')

        fig.suptitle(f'Attention Map Comparison — {stem}', fontsize=14, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{stem}_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Attention Map Visualization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--image_paths', nargs='+', type=str, default=None,
                        help='Specific image paths (RGB). IR is auto-detected.')
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    if args.image_paths:
        # Custom images mode
        model, train_args = build_model(args.checkpoint, device=args.device)
        patch_size = model.patch_embed.patch_size
        out_dir = args.output_dir or os.path.join(
            os.path.dirname(__file__), 'outputs', 'attention_maps')
        os.makedirs(out_dir, exist_ok=True)

        for rgb_path in args.image_paths:
            # Try to find matching IR
            ir_path = rgb_path.replace('/img/', '/imgr/').replace('/visible/', '/infrared/')
            if not os.path.exists(ir_path):
                ir_path = None

            tensor, rgb_disp, ir_disp = load_single_image(
                rgb_path, ir_path, img_size=args.img_size)

            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            for mode in ['both', 'rgb_only', 'ir_only']:
                attn = extract_attention_weights(
                    model, tensor, modality_mode=mode, device=args.device)
                out_path = os.path.join(out_dir, f'{stem}_{mode}.png')
                visualize_attention(
                    attn, rgb_disp, ir_disp, patch_size,
                    output_path=out_path,
                    title_prefix=f'[{mode.replace("_", " ").title()}] ',
                )
    else:
        eval_attention_maps(
            args.checkpoint,
            num_images=args.num_images,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            device=args.device,
            img_size=args.img_size,
        )


if __name__ == '__main__':
    main()

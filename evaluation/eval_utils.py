"""
Shared utilities for DINO-MM evaluation suite.
- Checkpoint loading and model building
- Feature extraction with modality control
- Dataset classes: DroneVehicleClassification, LLVIPPaired
- Standard eval transforms
"""

import os
import sys
import math
import xml.etree.ElementTree as ET
from functools import partial
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add project root to path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.vision_transformer_rgbir import (
    VisionTransformer, vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant,
)

# Architecture registry
ARCH_REGISTRY = {
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_large': vit_large,
    'vit_huge': vit_huge,
    'vit_giant': vit_giant,
}

# DroneVehicle class names (observed in annotations)
DRONEVEHICLE_CLASSES = ['car', 'truck', 'bus', 'feright car', 'van']
# Simplified: map to a smaller set for binary or multi-class
DRONEVEHICLE_CLASS_MAP = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'feright car': 3,
    'van': 4,
}

# Standard normalization (ImageNet stats for RGB, replicate mean for IR)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IR_MEAN = [0.5]
IR_STD = [0.5]
RGBIR_MEAN = IMAGENET_MEAN + IR_MEAN  # [4]
RGBIR_STD = IMAGENET_STD + IR_STD     # [4]


# =============================================================================
# Checkpoint loading & model building
# =============================================================================

def load_checkpoint(ckpt_path):
    """Load checkpoint and return the full state dict.

    Returns:
        dict with keys like 'student', 'teacher', 'args', 'epoch', etc.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    if 'epoch' in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if 'args' in ckpt:
        args = ckpt['args']
        if isinstance(args, dict):
            if 'arch' in args:
                print(f"  Arch: {args.get('arch')}, Patch: {args.get('patch_size')}, Fusion: {args.get('fusion', 'N/A')}")
        elif hasattr(args, 'arch'):
            print(f"  Arch: {args.arch}, Patch: {args.patch_size}, Fusion: {getattr(args, 'fusion', 'N/A')}")
    return ckpt


def build_model(ckpt_or_path, key='teacher', device='cuda'):
    """Build ViT model from checkpoint, load weights, set to eval mode.

    Args:
        ckpt_or_path: checkpoint dict or path to .pth file
        key: 'teacher' or 'student' (teacher is better per DINO convention)
        device: target device

    Returns:
        model: VisionTransformer in eval mode on device
        args: the training args from checkpoint (or None)
    """
    if isinstance(ckpt_or_path, str):
        ckpt = load_checkpoint(ckpt_or_path)
    else:
        ckpt = ckpt_or_path

    # Extract training args
    args = ckpt.get('args', None)

    # Checkpoint args may be argparse.Namespace or a plain dict.
    if isinstance(args, dict):
        arch = args.get('arch', 'vit_large')
        patch_size = args.get('patch_size', 14)
        in_chans = args.get('in_chans', 4)
        fusion = args.get('fusion', 'gated_cross_attn')
    elif args is not None:
        arch = getattr(args, 'arch', 'vit_large')
        patch_size = getattr(args, 'patch_size', 14)
        in_chans = getattr(args, 'in_chans', 4)
        fusion = getattr(args, 'fusion', 'gated_cross_attn')
    else:
        # Fallback: try to infer from checkpoint keys
        arch = 'vit_large'
        patch_size = 14
        in_chans = 4
        fusion = 'gated_cross_attn'
        print("Warning: No args in checkpoint, using default architecture config")

    # Build model
    build_fn = ARCH_REGISTRY.get(arch)
    if build_fn is None:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(ARCH_REGISTRY.keys())}")

    model = build_fn(
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        num_classes=0,
        fusion_start_block=args.get('fusion_start_block', 0) if isinstance(args, dict)
            else getattr(args, 'fusion_start_block', 0) if args is not None else 0,
    )

    # Load weights
    state_dict = ckpt.get(key, ckpt.get('model', ckpt.get('state_dict', {})))

    # Strip MultiCropWrapper prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        # Remove 'backbone.' prefix from MultiCropWrapper
        if k.startswith('backbone.'):
            cleaned[k[len('backbone.'):]] = v
        elif k.startswith('module.backbone.'):
            cleaned[k[len('module.backbone.'):]] = v
        elif k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            # Skip head weights (from DINOHead)
            if not k.startswith('head.') and not k.startswith('head0.') and not k.startswith('head1.'):
                cleaned[k] = v

    msg = model.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  Missing keys: {msg.missing_keys[:5]}{'...' if len(msg.missing_keys) > 5 else ''}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")

    model.eval()
    model.to(device)
    print(f"  Model built: {arch} patch={patch_size} fusion={fusion} embed_dim={model.embed_dim}")
    print(f"  Loaded '{key}' weights, moved to {device}")
    return model, args


def build_random_model(arch='vit_small', patch_size=16, fusion='gated_cross_attn', device='cuda'):
    """Build a randomly initialized model (for baseline comparison)."""
    build_fn = ARCH_REGISTRY[arch]
    model = build_fn(patch_size=patch_size, in_chans=4, fusion=fusion, num_classes=0)
    model.eval()
    model.to(device)
    return model


def build_dino_baseline(weights_path, device='cuda'):
    """Build original DINO ViT-S/16 (3-channel, ImageNet pretrained) as baseline.

    Args:
        weights_path: path to dino_deitsmall16_pretrain.pth
        device: target device

    Returns:
        model: VisionTransformer (3ch) in eval mode
    """
    model = vit_small(patch_size=16, in_chans=3, fusion='concat', num_classes=0)

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    # Facebook DINO checkpoints store weights directly (no 'teacher'/'student' wrapper)
    if 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # Strip 'backbone.' or 'module.' prefixes if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            cleaned[k[len('backbone.'):]] = v
        elif k.startswith('module.backbone.'):
            cleaned[k[len('module.backbone.'):]] = v
        elif k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            if not k.startswith('head.') and not k.startswith('head0.') and not k.startswith('head1.'):
                cleaned[k] = v

    msg = model.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  DINO baseline missing keys: {msg.missing_keys[:5]}{'...' if len(msg.missing_keys) > 5 else ''}")
    if msg.unexpected_keys:
        print(f"  DINO baseline unexpected keys: {msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")

    model.eval()
    model.to(device)
    print(f"  DINO baseline built: vit_small patch=16 in_chans=3 embed_dim={model.embed_dim}")
    return model


# =============================================================================
# Feature extraction
# =============================================================================

def prepare_input(images_4ch, modality_mode='both'):
    """Apply modality masking to a batch of 4-channel images.

    Args:
        images_4ch: [B, 4, H, W] tensor
        modality_mode: 'both', 'rgb_only', 'ir_only'

    Returns:
        masked_images: [B, 4, H, W] with appropriate channels zeroed
    """
    x = images_4ch.clone()
    if modality_mode == 'rgb_only':
        x[:, 3:, :, :] = 0.0  # zero out IR
    elif modality_mode == 'ir_only':
        x[:, :3, :, :] = 0.0  # zero out RGB
    return x


def infer_modality_masks(images_4ch):
    """Infer [RGB, IR] availability masks from a batch of 4-channel tensors."""
    rgb_avail = images_4ch[:, :3].abs().sum(dim=(1, 2, 3)) > 0
    ir_avail = images_4ch[:, 3:].abs().sum(dim=(1, 2, 3)) > 0
    return torch.stack([rgb_avail.float(), ir_avail.float()], dim=1)


@torch.no_grad()
def extract_features(model, dataloader, modality_mode='both', n_last_blocks=1,
                     device='cuda', max_samples=None, verbose=True):
    """Extract CLS features from a dataloader.

    Args:
        model: VisionTransformer in eval mode
        dataloader: yields (images_4ch, labels) or (images_4ch,)
        modality_mode: 'both', 'rgb_only', 'ir_only'
        n_last_blocks: number of last blocks to concatenate CLS tokens from
        device: torch device
        max_samples: limit total samples (None = all)
        verbose: print progress

    Returns:
        features: [N, feat_dim] tensor
        labels: [N] tensor or None
    """
    model.eval()
    all_features = []
    all_labels = []
    n_collected = 0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                imgs, labs = batch[0], batch[1]
                all_labels.append(labs)
            else:
                imgs = batch[0]
        else:
            imgs = batch

        imgs = imgs.to(device)
        imgs = prepare_input(imgs, modality_mode)
        modality_masks = infer_modality_masks(imgs)

        if n_last_blocks > 1:
            intermediate = model.get_intermediate_layers(
                imgs, n=n_last_blocks, modality_masks=modality_masks)
            feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)
        else:
            feat = model(imgs, return_all_tokens=False, modality_masks=modality_masks)

        all_features.append(feat.cpu())
        n_collected += imgs.shape[0]

        if verbose and (batch_idx + 1) % 50 == 0:
            print(f"  Extracted {n_collected} samples...")

        if max_samples is not None and n_collected >= max_samples:
            break

    features = torch.cat(all_features, dim=0)
    if max_samples is not None:
        features = features[:max_samples]

    labels = None
    if all_labels:
        labels = torch.cat(all_labels, dim=0)
        if max_samples is not None:
            labels = labels[:max_samples]

    if verbose:
        print(f"  Total: {features.shape[0]} samples, feature dim={features.shape[1]}")
    return features, labels


@torch.no_grad()
def extract_features_3ch(model, dataloader, n_last_blocks=1,
                         device='cuda', max_samples=None, verbose=True):
    """Extract CLS features from a 3-channel model (DINO baseline).

    Simplified version of extract_features() without modality masking.
    Expects dataloader to yield (images_3ch, labels).
    """
    model.eval()
    all_features = []
    all_labels = []
    n_collected = 0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                imgs, labs = batch[0], batch[1]
                all_labels.append(labs)
            else:
                imgs = batch[0]
        else:
            imgs = batch

        imgs = imgs.to(device)

        if n_last_blocks > 1:
            intermediate = model.get_intermediate_layers(imgs, n=n_last_blocks)
            feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)
        else:
            feat = model(imgs, return_all_tokens=False)

        all_features.append(feat.cpu())
        n_collected += imgs.shape[0]

        if verbose and (batch_idx + 1) % 50 == 0:
            print(f"  Extracted {n_collected} samples...")

        if max_samples is not None and n_collected >= max_samples:
            break

    features = torch.cat(all_features, dim=0)
    if max_samples is not None:
        features = features[:max_samples]

    labels = None
    if all_labels:
        labels = torch.cat(all_labels, dim=0)
        if max_samples is not None:
            labels = labels[:max_samples]

    if verbose:
        print(f"  Total: {features.shape[0]} samples, feature dim={features.shape[1]}")
    return features, labels


@torch.no_grad()
def extract_attention_weights(model, images_4ch, modality_mode='both', device='cuda'):
    """Extract attention weights from the last transformer block.

    Uses a forward hook on the last block's attention module.

    Returns:
        attn_weights: [B, num_heads, N+1, N+1] (softmax attention)
    """
    model.eval()
    imgs = images_4ch.to(device)
    imgs = prepare_input(imgs, modality_mode)
    modality_masks = infer_modality_masks(imgs)

    attn_output = {}

    def hook_fn(module, input, output):
        # The Attention.forward returns x after projection
        # We need to capture the attention weights before projection
        # Re-compute them from the same input
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn_output['attn'] = attn.detach().cpu()

    # Register hook on the last block's attention
    last_block = model.blocks[-1]
    handle = last_block.attn.register_forward_hook(hook_fn)

    try:
        _ = model(imgs, return_all_tokens=True, modality_masks=modality_masks)
    finally:
        handle.remove()

    return attn_output['attn']


# =============================================================================
# Eval transforms
# =============================================================================

class ToTensor4Ch:
    """Convert [H, W, 4] uint8 numpy to [4, H, W] float tensor normalized to [0,1]."""
    def __call__(self, img):
        # img: [H, W, 4] uint8
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [4, H, W]
        return img


class Normalize4Ch:
    """Normalize 4-channel tensor with per-channel mean/std."""
    def __init__(self, mean=RGBIR_MEAN, std=RGBIR_STD):
        self.mean = torch.tensor(mean).view(4, 1, 1)
        self.std = torch.tensor(std).view(4, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def get_eval_transform(img_size=224, crop_size=224):
    """Standard eval transform: resize + center crop + to_tensor + normalize.

    Works on [H, W, 4] uint8 numpy arrays (RGB+IR concatenated).
    """
    def transform(img_4ch):
        # img_4ch: [H, W, 4] uint8 numpy
        h, w = img_4ch.shape[:2]
        # Resize (keep aspect ratio, resize shorter side to img_size)
        if h < w:
            new_h = img_size
            new_w = int(w * img_size / h)
        else:
            new_w = img_size
            new_h = int(h * img_size / w)
        img_resized = cv2.resize(img_4ch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Center crop
        start_h = (new_h - crop_size) // 2
        start_w = (new_w - crop_size) // 2
        img_cropped = img_resized[start_h:start_h + crop_size, start_w:start_w + crop_size]

        # To tensor + normalize
        tensor = ToTensor4Ch()(img_cropped)
        tensor = Normalize4Ch()(tensor)
        return tensor

    return transform


def get_eval_transform_3ch(img_size=224, crop_size=224):
    """Standard 3-channel RGB eval transform for DINO baseline.

    Works on [H, W, 3] uint8 numpy arrays (RGB only).
    Uses ImageNet normalization.
    """
    def transform(img_rgb):
        # img_rgb: [H, W, 3] uint8 numpy
        h, w = img_rgb.shape[:2]
        if h < w:
            new_h = img_size
            new_w = int(w * img_size / h)
        else:
            new_w = img_size
            new_h = int(h * img_size / w)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        start_h = (new_h - crop_size) // 2
        start_w = (new_w - crop_size) // 2
        img_cropped = img_resized[start_h:start_h + crop_size, start_w:start_w + crop_size]

        # To float tensor [3, H, W]
        tensor = torch.from_numpy(img_cropped.astype(np.float32) / 255.0).permute(2, 0, 1)
        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    return transform


# =============================================================================
# Dataset: DroneVehicleClassification
# =============================================================================

def parse_voc_xml(xml_path, class_map=None):
    """Parse a VOC-format XML annotation file.

    DroneVehicle uses <polygon> (x1,y1,x2,y2,x3,y3,x4,y4) instead of <bndbox>.
    We compute axis-aligned bounding boxes from the polygon corners.

    Returns:
        list of dicts: [{'class': str, 'class_id': int, 'bbox': [x1,y1,x2,y2]}]
    """
    if class_map is None:
        class_map = DRONEVEHICLE_CLASS_MAP

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        if name not in class_map:
            continue
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue

        # Try bndbox first, then polygon
        bndbox = obj.find('bndbox')
        polygon = obj.find('polygon')

        if bndbox is not None:
            x1 = int(float(bndbox.find('xmin').text))
            y1 = int(float(bndbox.find('ymin').text))
            x2 = int(float(bndbox.find('xmax').text))
            y2 = int(float(bndbox.find('ymax').text))
        elif polygon is not None:
            xs = []
            ys = []
            for i in range(1, 5):
                xi = polygon.find(f'x{i}')
                yi = polygon.find(f'y{i}')
                if xi is not None and yi is not None:
                    xs.append(int(float(xi.text)))
                    ys.append(int(float(yi.text)))
            if not xs:
                continue
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
        else:
            continue

        objects.append({
            'class': name,
            'class_id': class_map[name],
            'bbox': [x1, y1, x2, y2],
        })

    return objects


class DroneVehicleClassification(Dataset):
    """DroneVehicle dataset for image-level classification.

    Assigns each image the dominant class label (most frequent object class).
    Loads RGB+IR paired images as 4-channel input.

    Directory structure:
        {root}/{split}/{split}img/     — RGB images
        {root}/{split}/{split}imgr/    — IR images
        {root}/{split}/{split}label/   — RGB VOC XML annotations
    """

    def __init__(self, root, split='train', transform=None, class_map=None,
                 min_objects=1):
        """
        Args:
            root: DroneVehicle root directory
            split: 'train' or 'val'
            transform: callable on [H,W,4] uint8 array → tensor
            class_map: dict mapping class name → int
            min_objects: minimum objects in image to include
        """
        self.root = root
        self.split = split
        self.transform = transform or get_eval_transform()
        self.class_map = class_map or DRONEVEHICLE_CLASS_MAP

        # Build paths
        self.rgb_dir = os.path.join(root, split, f'{split}img')
        self.ir_dir = os.path.join(root, split, f'{split}imgr')
        self.label_dir = os.path.join(root, split, f'{split}label')

        # Scan and filter samples
        self.samples = []
        label_files = sorted(os.listdir(self.label_dir))
        for lf in label_files:
            if not lf.endswith('.xml'):
                continue
            xml_path = os.path.join(self.label_dir, lf)
            objects = parse_voc_xml(xml_path, self.class_map)
            if len(objects) < min_objects:
                continue

            # Dominant class
            class_counts = defaultdict(int)
            for obj in objects:
                class_counts[obj['class_id']] += 1
            label = max(class_counts, key=class_counts.get)

            stem = lf.replace('.xml', '')
            self.samples.append({
                'stem': stem,
                'label': label,
                'objects': objects,
            })

        # Build class distribution
        self.class_names = {v: k for k, v in self.class_map.items()}
        class_dist = defaultdict(int)
        for s in self.samples:
            class_dist[s['label']] += 1
        print(f"DroneVehicleClassification [{split}]: {len(self.samples)} images")
        for cid, cnt in sorted(class_dist.items()):
            print(f"  {self.class_names.get(cid, cid)}: {cnt}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        stem = sample['stem']

        # Load RGB
        rgb_path = os.path.join(self.rgb_dir, f'{stem}.jpg')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            rgb_path = os.path.join(self.rgb_dir, f'{stem}.png')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"Cannot read RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load IR
        ir_path = os.path.join(self.ir_dir, f'{stem}.jpg')
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            ir_path = os.path.join(self.ir_dir, f'{stem}.png')
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            ir = np.zeros(rgb.shape[:2], dtype=np.uint8)

        # Ensure same size
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]

        # Concatenate: [H, W, 4]
        img_4ch = np.concatenate([rgb, ir], axis=2)

        # Apply transform
        tensor = self.transform(img_4ch)

        label = sample['label']
        return tensor, label


class DroneVehicleClassification3Ch(Dataset):
    """DroneVehicle dataset returning 3-channel RGB tensors for DINO baseline.

    Reuses the same sample scanning logic as DroneVehicleClassification but
    only loads RGB images (no IR), producing [3, H, W] tensors compatible
    with the original DINO ViT-S/16 (in_chans=3).
    """

    def __init__(self, root, split='train', transform=None, class_map=None,
                 min_objects=1):
        self.root = root
        self.split = split
        self.transform = transform or get_eval_transform_3ch()
        self.class_map = class_map or DRONEVEHICLE_CLASS_MAP

        self.rgb_dir = os.path.join(root, split, f'{split}img')
        self.label_dir = os.path.join(root, split, f'{split}label')

        self.samples = []
        label_files = sorted(os.listdir(self.label_dir))
        for lf in label_files:
            if not lf.endswith('.xml'):
                continue
            xml_path = os.path.join(self.label_dir, lf)
            objects = parse_voc_xml(xml_path, self.class_map)
            if len(objects) < min_objects:
                continue

            class_counts = defaultdict(int)
            for obj in objects:
                class_counts[obj['class_id']] += 1
            label = max(class_counts, key=class_counts.get)

            stem = lf.replace('.xml', '')
            self.samples.append({'stem': stem, 'label': label})

        self.class_names = {v: k for k, v in self.class_map.items()}
        class_dist = defaultdict(int)
        for s in self.samples:
            class_dist[s['label']] += 1
        print(f"DroneVehicleClassification3Ch [{split}]: {len(self.samples)} images")
        for cid, cnt in sorted(class_dist.items()):
            print(f"  {self.class_names.get(cid, cid)}: {cnt}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        stem = sample['stem']

        rgb_path = os.path.join(self.rgb_dir, f'{stem}.jpg')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            rgb_path = os.path.join(self.rgb_dir, f'{stem}.png')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"Cannot read RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        tensor = self.transform(rgb)
        return tensor, sample['label']


class DroneVehicleCropDataset(Dataset):
    """DroneVehicle dataset that crops individual object regions.

    Used for downstream detection evaluation (crop-based classification).
    Each sample is a cropped region of an object, labeled by its class.
    """

    def __init__(self, root, split='train', transform=None, class_map=None,
                 min_crop_size=32, context_factor=1.2):
        """
        Args:
            root: DroneVehicle root directory
            split: 'train' or 'val'
            transform: callable on [H,W,4] uint8 → tensor
            class_map: class name → int mapping
            min_crop_size: minimum crop width/height in pixels
            context_factor: expand bbox by this factor for context
        """
        self.root = root
        self.split = split
        self.transform = transform or get_eval_transform(img_size=128, crop_size=128)
        self.class_map = class_map or DRONEVEHICLE_CLASS_MAP
        self.context_factor = context_factor

        rgb_dir = os.path.join(root, split, f'{split}img')
        ir_dir = os.path.join(root, split, f'{split}imgr')
        label_dir = os.path.join(root, split, f'{split}label')

        self.crops = []
        label_files = sorted(os.listdir(label_dir))
        for lf in label_files:
            if not lf.endswith('.xml'):
                continue
            xml_path = os.path.join(label_dir, lf)
            objects = parse_voc_xml(xml_path, self.class_map)
            stem = lf.replace('.xml', '')

            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                w, h = x2 - x1, y2 - y1
                if w < min_crop_size or h < min_crop_size:
                    continue
                self.crops.append({
                    'stem': stem,
                    'bbox': obj['bbox'],
                    'class_id': obj['class_id'],
                    'rgb_dir': rgb_dir,
                    'ir_dir': ir_dir,
                })

        class_dist = defaultdict(int)
        for c in self.crops:
            class_dist[c['class_id']] += 1
        print(f"DroneVehicleCropDataset [{split}]: {len(self.crops)} crops")
        class_names = {v: k for k, v in self.class_map.items()}
        for cid, cnt in sorted(class_dist.items()):
            print(f"  {class_names.get(cid, cid)}: {cnt}")

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop_info = self.crops[idx]
        stem = crop_info['stem']

        # Load images
        rgb_path = os.path.join(crop_info['rgb_dir'], f'{stem}.jpg')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            rgb_path = os.path.join(crop_info['rgb_dir'], f'{stem}.png')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ir_path = os.path.join(crop_info['ir_dir'], f'{stem}.jpg')
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            ir_path = os.path.join(crop_info['ir_dir'], f'{stem}.png')
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            ir = np.zeros(rgb.shape[:2], dtype=np.uint8)
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]

        img_4ch = np.concatenate([rgb, ir], axis=2)

        # Crop with context
        H, W = img_4ch.shape[:2]
        x1, y1, x2, y2 = crop_info['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        bw *= self.context_factor
        bh *= self.context_factor
        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(W, int(cx + bw / 2))
        y2 = min(H, int(cy + bh / 2))

        crop = img_4ch[y1:y2, x1:x2]
        tensor = self.transform(crop)
        return tensor, crop_info['class_id']


# =============================================================================
# Dataset: LLVIPPaired
# =============================================================================

class LLVIPPaired(Dataset):
    """LLVIP dataset for cross-modal retrieval evaluation.

    Returns paired RGB and IR images as separate 4-channel tensors.
    For RGB input: IR channel is zeroed.
    For IR input: RGB channels are zeroed.
    """

    def __init__(self, root, split='test', transform=None, max_samples=None):
        """
        Args:
            root: LLVIP registered root (e.g., /root/autodl-tmp/data/LLVIP/registered/)
            split: 'train' or 'test'
            transform: callable on [H,W,4] uint8 → tensor
            max_samples: limit total pairs
        """
        self.root = root
        self.split = split
        self.transform = transform or get_eval_transform()

        rgb_dir = os.path.join(root, 'visible', split)
        ir_dir = os.path.join(root, 'infrared', split)

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.jpg', '.png'))])

        # Match by filename stem
        rgb_stems = {os.path.splitext(f)[0]: f for f in rgb_files}
        ir_stems = {os.path.splitext(f)[0]: f for f in ir_files}
        common_stems = sorted(set(rgb_stems.keys()) & set(ir_stems.keys()))

        self.pairs = []
        for stem in common_stems:
            self.pairs.append({
                'rgb_path': os.path.join(rgb_dir, rgb_stems[stem]),
                'ir_path': os.path.join(ir_dir, ir_stems[stem]),
                'stem': stem,
            })

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        print(f"LLVIPPaired [{split}]: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load RGB
        rgb = cv2.imread(pair['rgb_path'], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load IR
        ir = cv2.imread(pair['ir_path'], cv2.IMREAD_GRAYSCALE)
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]

        # Build two 4-channel images
        # RGB input: [RGB, zeros]
        rgb_4ch = np.concatenate([rgb, np.zeros_like(ir)], axis=2)
        # IR input: [zeros, IR]
        ir_4ch = np.concatenate([np.zeros_like(rgb), ir], axis=2)

        rgb_tensor = self.transform(rgb_4ch)
        ir_tensor = self.transform(ir_4ch)

        return rgb_tensor, ir_tensor, idx


class DroneVehiclePaired(Dataset):
    """DroneVehicle dataset for cross-modal retrieval.

    Returns paired RGB and IR images as separate 4-channel tensors.
    """

    def __init__(self, root, split='val', transform=None, max_samples=None):
        self.root = root
        self.split = split
        self.transform = transform or get_eval_transform()

        rgb_dir = os.path.join(root, split, f'{split}img')
        ir_dir = os.path.join(root, split, f'{split}imgr')

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])

        self.pairs = []
        for f in rgb_files:
            stem = os.path.splitext(f)[0]
            rgb_path = os.path.join(rgb_dir, f)
            ir_path = os.path.join(ir_dir, f)
            if not os.path.exists(ir_path):
                ir_path = os.path.join(ir_dir, stem + '.png')
            if os.path.exists(ir_path):
                self.pairs.append({'rgb_path': rgb_path, 'ir_path': ir_path, 'stem': stem})

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        print(f"DroneVehiclePaired [{split}]: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        rgb = cv2.imread(pair['rgb_path'], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ir = cv2.imread(pair['ir_path'], cv2.IMREAD_GRAYSCALE)
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]

        rgb_4ch = np.concatenate([rgb, np.zeros_like(ir)], axis=2)
        ir_4ch = np.concatenate([np.zeros_like(rgb), ir], axis=2)

        rgb_tensor = self.transform(rgb_4ch)
        ir_tensor = self.transform(ir_4ch)

        return rgb_tensor, ir_tensor, idx


# =============================================================================
# Helper functions
# =============================================================================

def get_dataloader(dataset, batch_size=64, num_workers=4, shuffle=False):
    """Create a standard DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_num_classes(class_map=None):
    """Return the number of classes."""
    if class_map is None:
        class_map = DRONEVEHICLE_CLASS_MAP
    return len(set(class_map.values()))


if __name__ == '__main__':
    # Quick sanity check
    print("=== Testing DroneVehicleClassification ===")
    ds = DroneVehicleClassification('/root/autodl-tmp/data/DroneVehicle', split='val')
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        img, label = ds[0]
        print(f"Sample shape: {img.shape}, label: {label}")

    print("\n=== Testing LLVIPPaired ===")
    ds2 = LLVIPPaired('/root/autodl-tmp/data/LLVIP/registered/', split='test', max_samples=10)
    print(f"Dataset size: {len(ds2)}")
    if len(ds2) > 0:
        rgb_t, ir_t, idx = ds2[0]
        print(f"RGB tensor: {rgb_t.shape}, IR tensor: {ir_t.shape}")

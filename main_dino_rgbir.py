"""
Main training script for RGB+IR DINO pretraining.

Usage:
    # Single-GPU training
    python main_dino_rgbir.py \
        --arch vit_small --patch_size 8 --in_chans 4 \
        --data_path /root/autodl-tmp/data/RGBT-Tiny/manifest_train.json \
        --data_mode file --output_dir ./checkpoints/demo_rgbir --epochs 50

    # Multi-GPU training
    torchrun --nproc_per_node=N main_dino_rgbir.py \
        --arch vit_small --patch_size 8 --in_chans 4 \
        --data_path /root/autodl-tmp/data/RGBT-Tiny/manifest_train.json \
        --data_mode file --output_dir ./checkpoints/demo_rgbir --epochs 50
"""

import argparse
import os
import sys
import time
import math
import json
import datetime
import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

from models.vision_transformer_rgbir import (
    vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant
)
from models.dino_head import DINOHead, MultiCropWrapper
from models.multi_granularity import MultiGranularityFeatures
from models.temporal_module import TemporalAttention
from models.transforms_rgbir import DataAugmentationDINO_RGBIR
from models.view_bridge import ViewDomainBridge
from models.cross_modal_projector import CrossModalProjector
from models.pcgrad import pcgrad_backward
from datasets.multimodal_drone import (
    MultiModalDroneDataset, collate_multimodal, TypeBalancedDistributedSampler)
from datasets.video_dataset import build_video_dataset, collate_video
from dino_loss_rgbir import (
    DINOLoss, MGCLLoss, TCLLoss, PATCLoss,
    ViewInvarianceLoss, ViewBridgeLoss, CrossModalPairLoss,
    CrossModalPairLossWithQueue, GrayscaleBridgeLoss,
    CrossModalPatchLoss, CrossModalMGCLLoss, PretrainingLoss)


# ============================================================================
# Utilities
# ============================================================================

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine annealing schedule with optional linear warmup."""
    total_iters = max(0, epochs * niter_per_ep)
    warmup_iters = min(max(0, warmup_epochs * niter_per_ep), total_iters)

    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    remaining_iters = total_iters - warmup_iters
    if remaining_iters > 0:
        iters = np.arange(remaining_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / remaining_iters)
        )
    else:
        schedule = np.array([])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def ramp_scheduler(base_value, epochs, niter_per_ep,
                   ramp_epochs=0, start_value=None):
    """Linear ramp to `base_value`, then stay constant."""
    total_iters = epochs * niter_per_ep
    ramp_iters = min(max(0, ramp_epochs * niter_per_ep), total_iters)
    if start_value is None:
        start_value = base_value

    if ramp_iters > 0:
        ramp = np.linspace(start_value, base_value, ramp_iters)
        steady = np.ones(max(0, total_iters - ramp_iters)) * base_value
        schedule = np.concatenate((ramp, steady))
    else:
        schedule = np.ones(total_iters) * base_value

    assert len(schedule) == total_iters
    return schedule


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        if not self.deque:
            return 0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def normalize_type_sampling_weights(raw_weights):
    if raw_weights is None:
        return None
    sample_types = ('paired', 'rgb_only', 'ir_only')
    weights = [max(0.0, float(w)) for w in raw_weights]
    total = sum(weights)
    if total <= 0:
        return None
    return {
        sample_type: weight / total
        for sample_type, weight in zip(sample_types, weights)
    }
    return dist.get_world_size()


def get_available_cpu_count():
    """Return the CPU quota visible to this process."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def resolve_num_workers(requested):
    """Clamp DataLoader workers to the available CPU quota."""
    available = get_available_cpu_count()
    if requested is None:
        return min(16, available)
    if requested < 0:
        return max(1, available - 2)
    return max(0, min(int(requested), available))


def dataloader_worker_init_fn(_worker_id):
    """Keep each worker single-threaded to avoid CPU oversubscription."""
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except ImportError:
        pass


@torch.no_grad()
def build_modality_grouped_bridge_teacher_z(teacher_z_list, mask_list):
    """Average teacher bridge features only across views with the same modality.

    Each sample is grouped by its per-view modality mask, so RGB-only views are
    no longer forced to share the same bridge target as IR-only views.
    """
    if not teacher_z_list:
        return []
    if len(teacher_z_list) != len(mask_list):
        raise ValueError(
            f"len(teacher_z_list)={len(teacher_z_list)} != len(mask_list)={len(mask_list)}"
        )

    batch_size = teacher_z_list[0].shape[0]
    grouped = [torch.empty_like(z) for z in teacher_z_list]
    for sample_idx in range(batch_size):
        groups = {}
        for view_idx, mask in enumerate(mask_list):
            key = tuple((mask[sample_idx] > 0.5).to(dtype=torch.int32).tolist())
            groups.setdefault(key, []).append(view_idx)
        for view_indices in groups.values():
            merged = F.normalize(
                torch.stack(
                    [teacher_z_list[view_idx][sample_idx] for view_idx in view_indices],
                    dim=0,
                ).mean(dim=0),
                dim=-1,
            )
            for view_idx in view_indices:
                grouped[view_idx][sample_idx] = merged
    return grouped


def save_checkpoint(state, filename):
    if is_main_process():
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """Freeze the last layer of DINO head for the first N epochs."""
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


# ============================================================================
# Argument parser
# ============================================================================

def str2bool(v):
    """Robust bool parser for argparse (accepts true/false/1/0/yes/no)."""
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ('yes', 'true', 't', '1', 'y'):
        return True
    if val in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def get_args_parser():
    parser = argparse.ArgumentParser('DINO-MM RGB+IR Pretraining')

    # Model
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'vit_huge', 'vit_giant'],
                        help='ViT architecture')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--in_chans', default=4, type=int,
                        help='Input channels (3 RGB + 1 IR)')
    parser.add_argument('--out_dim', default=65536, type=int,
                        help='DINO head output dimension')
    parser.add_argument('--norm_last_layer', default=True, type=str2bool,
                        help='Normalize last layer of DINO head')
    parser.add_argument('--fusion', default='gated_cross_attn', type=str,
                        choices=['concat', 'add', 'cross_attn', 'gated_cross_attn'],
                        help=(
                            'RGB-IR patch embedding fusion mode. '
                            '"concat": cat+Linear+LN (~d^2 extra params); '
                            '"add": element-wise sum (no extra params); '
                            '"cross_attn": RGB cross-attends to IR context; '
                            '"gated_cross_attn": bidirectional gated cross-attention (recommended).'))
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
                        help='Stochastic depth rate for student (0 for teacher)')
    parser.add_argument('--use_gradient_checkpointing', default=False, type=str2bool,
                        help='Enable gradient checkpointing to save activation memory (slower)')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int,
                        help='Gradient accumulation steps (effective_batch = batch_size * accum * gpus)')

    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size_per_gpu', default=32, type=int)
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Base learning rate (linearly scaled with batch size)')
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    parser.add_argument('--clip_grad', default=3.0, type=float)
    parser.add_argument('--freeze_last_layer', default=1, type=int)

    # EMA
    parser.add_argument('--momentum_teacher', default=0.996, type=float)

    # Temperature
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int)

    # Multi-crop
    parser.add_argument('--global_crop_size', default=224, type=int)
    parser.add_argument('--local_crop_size', default=96, type=int)
    parser.add_argument('--global_crops_scale', default=[0.4, 1.0],
                        nargs=2, type=float)
    parser.add_argument('--local_crops_scale', default=[0.05, 0.4],
                        nargs=2, type=float)
    parser.add_argument('--local_crops_number', default=6, type=int)

    # Multi-granularity
    parser.add_argument('--num_clusters', default=8, type=int,
                        help='Number of clusters for object-level features')
    parser.add_argument('--proj_dim', default=256, type=int,
                        help='Projection dimension for MGCL')

    # Loss weights
    parser.add_argument('--w_mgcl', default=1.0, type=float)
    parser.add_argument('--w_view', default=0.5, type=float)
    parser.add_argument('--w_bridge', default=0.5, type=float)
    parser.add_argument('--w_align_rgbir', default=0.0, type=float,
                        help='Weight for explicit paired RGB-only <-> IR-only alignment loss')
    parser.add_argument('--w_align_rgbir_patch', default=0.0, type=float,
                        help='Weight for teacher-guided paired RGB patch <-> IR patch correspondence loss')
    parser.add_argument('--w_align_mgcl', default=0.0, type=float,
                        help='Weight for 3-level cross-modal MGCL alignment (student single-modal → teacher fused)')
    parser.add_argument('--align_mgcl_temperature', default=0.1, type=float,
                        help='Temperature for cross-modal MGCL distillation')
    parser.add_argument('--w_dino_video', default=0.2, type=float,
                        help='Weight for video DINO loss (treated as video auxiliary loss)')
    parser.add_argument('--align_rgbir_temperature', default=0.07, type=float,
                        help='Temperature for explicit paired RGB-IR InfoNCE loss')
    parser.add_argument('--align_rgbir_patch_temperature', default=0.1, type=float,
                        help='Temperature for teacher-guided paired RGB-IR patch correspondence loss')
    parser.add_argument('--align_rgbir_patch_sigma', default=1.5, type=float,
                        help='Gaussian patch-radius prior for cross-modal patch correspondence, in patch-grid units')
    parser.add_argument('--align_rgbir_patch_geometry_blend', default=0.7, type=float,
                        help='Blend ratio between identity prior and teacher-guided geometry prior for cross-modal patch correspondence')
    parser.add_argument('--align_rgbir_patch_confidence_power', default=1.5, type=float,
                        help='Entropy-based confidence sharpening exponent for cross-modal patch correspondence')
    parser.add_argument('--align_rgbir_patch_min_confidence', default=0.05, type=float,
                        help='Minimum teacher confidence retained per patch for cross-modal patch correspondence')
    # Cross-modal alignment improvements
    parser.add_argument('--align_rgbir_queue_size', default=4096, type=int,
                        help='Momentum queue size for cross-modal alignment')
    parser.add_argument('--align_hard_neg_topk', default=128, type=int,
                        help='Top-K hardest negatives to upweight')
    parser.add_argument('--align_hard_neg_weight', default=2.0, type=float,
                        help='Weight multiplier for hard negatives')
    parser.add_argument('--align_temp_start', default=0.2, type=float,
                        help='Starting temperature for curriculum schedule')
    parser.add_argument('--align_temp_end', default=0.05, type=float,
                        help='Ending temperature for curriculum schedule')
    parser.add_argument('--align_temp_warmup_epochs', default=10, type=int,
                        help='Epochs over which temperature decays')
    parser.add_argument('--align_ramp_epochs', default=5, type=int,
                        help='Linearly ramp alignment loss weights over N epochs')
    parser.add_argument('--use_grayscale_bridge', default=False, type=str2bool,
                        help='Enable grayscale bridge loss (RGB→Gray→IR)')
    parser.add_argument('--w_gray_bridge', default=0.01, type=float,
                        help='Weight for grayscale bridge loss')
    parser.add_argument('--intermediate_cross_attn_layers', default=[], type=int, nargs='*',
                        help='Transformer layer indices for dual-stream intermediate cross-attention')
    # Adaptive loss weighting
    parser.add_argument('--adaptive_weighting', default=False, type=str2bool,
                        help='Use Kendall uncertainty weighting for auxiliary losses')
    parser.add_argument('--aux_loss_ramp_epochs', default=0, type=int,
                        help='Linearly ramp image auxiliary loss scales to 1 over N epochs')
    parser.add_argument('--aux_loss_start_scale', default=1.0, type=float,
                        help='Starting scale for image auxiliary losses during ramp')
    parser.add_argument('--video_aux_ramp_epochs', default=0, type=int,
                        help='Linearly ramp video auxiliary loss scales to 1 over N epochs')
    parser.add_argument('--video_aux_start_scale', default=1.0, type=float,
                        help='Starting scale for video auxiliary losses during ramp')

    # View-domain bridge
    parser.add_argument('--bridge_proj_dim', default=256, type=int)
    parser.add_argument('--bridge_num_prototypes', default=64, type=int)
    parser.add_argument('--bridge_sinkhorn_iters', default=3, type=int,
                        help='Sinkhorn-Knopp iterations for OT assignment')
    parser.add_argument('--bridge_sinkhorn_temp', default=0.1, type=float,
                        help='Sinkhorn temperature for assignment sharpness')
    parser.add_argument('--bridge_loss_temp', default=0.1, type=float,
                        help='Student temperature for bridge soft cross-entropy')
    parser.add_argument('--bridge_prototype_ema', default=0.999, type=float,
                        help='EMA decay for prototype updates')

    # View-domain bridge
    # Temporal (not used in phase 1)
    parser.add_argument('--use_temporal', default=False, action='store_true')
    parser.add_argument('--num_frames', default=1, type=int)
    parser.add_argument('--temporal_layers', default=2, type=int)
    parser.add_argument('--temporal_loss_type', default='patc', type=str,
                        choices=['tcl', 'patc'],
                        help='Temporal correspondence objective: vanilla TCL or PATC')
    parser.add_argument('--w_tcl', default=0.1, type=float,
                        help='Weight for temporal correspondence loss')
    parser.add_argument('--w_tcl_patch', default=0.5, type=float,
                        help='Weight for patch-level temporal correspondence loss')
    parser.add_argument('--tcl_temperature', default=0.07, type=float,
                        help='Temperature for temporal global InfoNCE loss')
    parser.add_argument('--tcl_patch_temperature', default=0.1, type=float,
                        help='Temperature for patch-level temporal correspondence')
    parser.add_argument('--tcl_spatial_sigma_base', default=2.0, type=float,
                        help='Base sigma for patch correspondence prior')
    parser.add_argument('--tcl_spatial_sigma_scale', default=0.5, type=float,
                        help='Sigma scaling factor per frame temporal distance')
    parser.add_argument('--patc_geometry_blend', default=0.7, type=float,
                        help='Blend ratio between identity prior and projected geometry prior in PATC')
    parser.add_argument('--patc_confidence_power', default=1.5, type=float,
                        help='Entropy-based confidence sharpening exponent in PATC')
    parser.add_argument('--patc_min_confidence', default=0.05, type=float,
                        help='Minimum teacher confidence retained per patch in PATC')
    parser.add_argument('--patc_gap_weight_scale', default=0.5, type=float,
                        help='Extra weighting applied to wider temporal/view gaps in PATC')

    # Video data (phase 2)
    parser.add_argument('--video_manifest', default='', type=str,
                        help='Path to video manifest JSON (empty=no video training)')
    parser.add_argument('--video_batch_size', default=8, type=int,
                        help='Video batch size per GPU')
    parser.add_argument('--video_num_workers', default=8, type=int,
                        help='Number of DataLoader workers for video')
    parser.add_argument('--video_prefetch_factor', default=2, type=int,
                        help='Video batches prefetched per worker')
    parser.add_argument('--video_persistent_workers', default=True, type=str2bool,
                        help='Keep video DataLoader workers alive across epochs')
    parser.add_argument('--video_target_size', default=224, type=int,
                        help='Target spatial resolution for video frames')
    parser.add_argument('--video_step_interval', default=0, type=int,
                        help='Consume one video batch every N image steps (0=auto-match image/video epoch lengths)')

    # Viewpoint augmentation
    parser.add_argument('--use_view_aug', default=True, type=str2bool)
    parser.add_argument('--alpha_range', default=[0.05, 0.35],
                        nargs=2, type=float)

    # Data
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to JSON manifest or LMDB database')
    parser.add_argument('--data_mode', default='file', type=str,
                        choices=['lmdb', 'file'])
    parser.add_argument('--strict_loading', default=False, type=str2bool,
                        help='Raise on any sample load failure (for precheck)')
    parser.add_argument('--max_load_fail_ratio', default=0.001, type=float,
                        help='Abort if failure ratio exceeds this (0=no limit)')
    parser.add_argument('--dry_run_samples', default=0, type=int,
                        help='Only load first N samples (0=all, for precheck)')
    parser.add_argument('--type_sampling_weights', default=None, nargs=3, type=float,
                        metavar=('PAIRED', 'RGB_ONLY', 'IR_ONLY'),
                        help='Optional image sampling weights for paired/rgb_only/ir_only. Example: --type_sampling_weights 0.7 0.25 0.05')
    parser.add_argument('--type_sampling_seed', default=0, type=int,
                        help='Seed offset for optional type-balanced image sampler')
    parser.add_argument('--alignment_offsets', default='', type=str,
                        help='Path to alignment_offsets.json for RGB-IR registration')

    # CLIP cross-modal alignment
    parser.add_argument('--fusion_start_block', default=0, type=int,
                        help='Block index where Stage2 (fusion) starts. 0=legacy single-stream, 4=recommended two-stage')
    parser.add_argument('--w_clip_align', default=0.0, type=float,
                        help='Weight for CLIP cross-modal alignment loss (image)')
    parser.add_argument('--w_clip_video', default=0.0, type=float,
                        help='Weight for CLIP cross-modal alignment loss (video)')
    parser.add_argument('--clip_proj_dim', default=128, type=int,
                        help='CLIP projector output dimension')
    parser.add_argument('--clip_queue_size', default=16384, type=int,
                        help='CLIP momentum queue size')

    # Pretrained weights
    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to pretrained weights (optional)')

    # Output
    parser.add_argument('--output_dir', default='./checkpoints/demo_rgbir', type=str)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--log_every', default=50, type=int)

    # Mixed precision
    parser.add_argument('--use_fp16', default=True, type=str2bool)

    # Distributed
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Image batches prefetched per worker')
    parser.add_argument('--persistent_workers', default=True, type=str2bool,
                        help='Keep image DataLoader workers alive across epochs')

    # Resume
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')

    # PCGrad
    parser.add_argument('--use_pcgrad', default=True, type=str2bool,
                        help='Use PCGrad gradient projection to resolve loss conflicts')

    # Wandb logging
    parser.add_argument('--wandb_key', default='', type=str,
                        help='Wandb API key (empty=disabled)')
    parser.add_argument('--wandb_project', default='DINO-MM', type=str,
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', default='', type=str,
                        help='Wandb run name (auto-generated if empty)')

    return parser


# ============================================================================
# Model building
# ============================================================================

def init_distributed(allow_cpu=False):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        print("Not using distributed mode, single GPU training")
        return 0, 1, 0
    elif allow_cpu:
        print("CUDA not available, running in CPU mode")
        return 0, 1, 0
    else:
        print("CUDA not available, exiting")
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=60),
    )
    dist.barrier()
    return rank, world_size, local_rank


def load_pretrained_weights(model, pretrained_path):
    """Load pretrained ViT weights into DualModalPatchEmbed backbone.

    Remaps 'patch_embed.proj.*' → 'patch_embed.rgb_proj.*' so that 3-channel
    pretrained weights (DINOv2, MAE, EVA, etc.) load directly into the RGB
    branch with no channel dimension hacks. The IR branch (ir_proj) stays
    randomly initialized and is trained from scratch on drone data.
    """
    from collections import OrderedDict

    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Try different keys
    for key in ['model', 'state_dict', 'teacher']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            if key == 'teacher':
                state_dict = {k.replace('backbone.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('backbone.')}
            break
    else:
        state_dict = checkpoint

    # Clean prefixes
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Conditionally remap patch_embed weights for DualModalPatchEmbed.
    #
    # Pretrained 3-channel checkpoints (DINOv2, MAE, etc.) store patch embed as:
    #   'patch_embed.proj.weight'  [d, 3, P, P]
    #   'patch_embed.proj.bias'    [d]
    #
    # DualModalPatchEmbed splits this into rgb_proj and ir_proj. If the current
    # model uses dual-modal patch embedding, load pretrained patch weights into
    # rgb_proj and keep ir_proj randomly initialized. For single-branch PatchEmbed,
    # keep original patch_embed.proj.* keys unchanged.
    is_dual_modal = hasattr(model.patch_embed, 'rgb_proj') and hasattr(model.patch_embed, 'ir_proj')
    if is_dual_modal:
        for suffix in ('weight', 'bias'):
            old_key = f'patch_embed.proj.{suffix}'
            new_key = f'patch_embed.rgb_proj.{suffix}'
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
                print(f"  Remapped {old_key} → {new_key} (IR branch stays random init)")

    # Handle position embedding size mismatch
    pos_embed_key = 'pos_embed'
    if pos_embed_key in state_dict:
        pretrained_pos = state_dict[pos_embed_key]
        model_pos = model.state_dict()[pos_embed_key]
        if pretrained_pos.shape != model_pos.shape:
            print(f"Interpolating pos_embed: {pretrained_pos.shape} -> {model_pos.shape}")
            cls_token_pos = pretrained_pos[:, :1, :]
            patch_pos = pretrained_pos[:, 1:, :]
            dim = patch_pos.shape[-1]
            old_size = int(math.sqrt(patch_pos.shape[1]))
            new_size = int(math.sqrt(model_pos.shape[1] - 1))

            patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
            patch_pos = nn.functional.interpolate(
                patch_pos, size=(new_size, new_size),
                mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
            state_dict[pos_embed_key] = torch.cat([cls_token_pos, patch_pos], dim=1)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_path}")
    if msg.missing_keys:
        print(f"  Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys}")
    return model


def build_model(args):
    """Build student and teacher models."""
    arch_dict = {
        'vit_tiny':  vit_tiny,
        'vit_small': vit_small,
        'vit_base':  vit_base,
        'vit_large': vit_large,
        'vit_huge':  vit_huge,
        'vit_giant': vit_giant,
    }
    backbone_fn = arch_dict[args.arch]

    # Build student backbone
    student_backbone = backbone_fn(
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        drop_path_rate=args.drop_path_rate,
        fusion=args.fusion,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        intermediate_cross_attn_layers=getattr(args, 'intermediate_cross_attn_layers', []),
        fusion_start_block=getattr(args, 'fusion_start_block', 0),
    )
    embed_dim = student_backbone.embed_dim

    # Build teacher backbone (no drop_path, no checkpointing — runs in no_grad)
    teacher_backbone = backbone_fn(
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        drop_path_rate=0.0,
        fusion=args.fusion,
        use_gradient_checkpointing=False,
        intermediate_cross_attn_layers=getattr(args, 'intermediate_cross_attn_layers', []),
        fusion_start_block=getattr(args, 'fusion_start_block', 0),
    )

    # DINO heads
    student_head = DINOHead(
        in_dim=embed_dim, out_dim=args.out_dim,
        use_bn=False, norm_last_layer=args.norm_last_layer)
    teacher_head = DINOHead(
        in_dim=embed_dim, out_dim=args.out_dim, use_bn=False)

    # Wrap with MultiCropWrapper
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)

    # Load pretrained weights if provided
    if args.pretrained:
        load_pretrained_weights(student.backbone, args.pretrained)

    # Copy student weights to teacher
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Multi-granularity feature extractor
    mg_student = MultiGranularityFeatures(
        embed_dim=embed_dim, proj_dim=args.proj_dim,
        num_clusters=args.num_clusters)
    mg_teacher = MultiGranularityFeatures(
        embed_dim=embed_dim, proj_dim=args.proj_dim,
        num_clusters=args.num_clusters)
    mg_teacher.load_state_dict(mg_student.state_dict())
    for p in mg_teacher.parameters():
        p.requires_grad = False

    # View-domain bridge
    view_bridge = ViewDomainBridge(
        in_dim=embed_dim,
        proj_dim=args.bridge_proj_dim,
        num_prototypes=args.bridge_num_prototypes,
        sinkhorn_iters=getattr(args, 'bridge_sinkhorn_iters', 3),
        sinkhorn_temp=getattr(args, 'bridge_sinkhorn_temp', 0.1),
        prototype_ema=getattr(args, 'bridge_prototype_ema', 0.999),
    )

    # Temporal attention (phase 2)
    temporal_attn = None
    if args.use_temporal:
        temporal_attn = TemporalAttention(
            embed_dim=embed_dim, num_heads=8,
            num_layers=args.temporal_layers)

    # CLIP cross-modal projector
    clip_projector = None
    if getattr(args, 'w_clip_align', 0) > 0 or getattr(args, 'w_clip_video', 0) > 0:
        clip_projector = CrossModalProjector(
            in_dim=embed_dim,
            hidden_dim=embed_dim,
            out_dim=getattr(args, 'clip_proj_dim', 128),
            queue_size=getattr(args, 'clip_queue_size', 16384),
        )

    return (student, teacher, mg_student, mg_teacher,
            view_bridge, temporal_attn, clip_projector, embed_dim)


def build_loss(args, embed_dim):
    """Build combined loss function."""
    ncrops = 2 + args.local_crops_number

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=ncrops,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
    )

    mgcl_loss = MGCLLoss(
        proj_dim=args.proj_dim,
        temperature=0.1,
    )

    view_loss = ViewInvarianceLoss(temperature=0.07)
    align_rgbir_loss = None
    if getattr(args, 'w_align_rgbir', 0.0) > 0:
        align_rgbir_loss = CrossModalPairLossWithQueue(
            embed_dim=embed_dim,
            temperature=getattr(args, 'align_rgbir_temperature', 0.07),
            queue_size=getattr(args, 'align_rgbir_queue_size', 4096),
            hard_neg_topk=getattr(args, 'align_hard_neg_topk', 128),
            hard_neg_weight=getattr(args, 'align_hard_neg_weight', 2.0),
            temp_start=getattr(args, 'align_temp_start', 0.2),
            temp_end=getattr(args, 'align_temp_end', 0.05),
            temp_warmup_epochs=getattr(args, 'align_temp_warmup_epochs', 10),
        )
    align_rgbir_patch_loss = None
    if getattr(args, 'w_align_rgbir_patch', 0.0) > 0:
        align_rgbir_patch_loss = CrossModalPatchLoss(
            patch_temperature=getattr(args, 'align_rgbir_patch_temperature', 0.1),
            spatial_sigma=getattr(args, 'align_rgbir_patch_sigma', 1.5),
            geometry_blend=getattr(args, 'align_rgbir_patch_geometry_blend', 0.7),
            confidence_power=getattr(args, 'align_rgbir_patch_confidence_power', 1.5),
            min_confidence=getattr(args, 'align_rgbir_patch_min_confidence', 0.05),
        )
    bridge_loss = ViewBridgeLoss(temperature=args.bridge_loss_temp)

    cross_modal_mgcl_loss = None
    if getattr(args, 'w_align_mgcl', 0.0) > 0:
        cross_modal_mgcl_loss = CrossModalMGCLLoss(
            temperature=getattr(args, 'align_mgcl_temperature', 0.1),
        )

    gray_bridge_loss = None
    if getattr(args, 'use_grayscale_bridge', False):
        gray_bridge_loss = GrayscaleBridgeLoss(
            embed_dim=embed_dim,
            temperature=getattr(args, 'align_rgbir_temperature', 0.07),
            queue_size=getattr(args, 'align_rgbir_queue_size', 4096),
            hard_neg_topk=getattr(args, 'align_hard_neg_topk', 128),
            hard_neg_weight=getattr(args, 'align_hard_neg_weight', 2.0),
            temp_start=getattr(args, 'align_temp_start', 0.2),
            temp_end=getattr(args, 'align_temp_end', 0.05),
            temp_warmup_epochs=getattr(args, 'align_temp_warmup_epochs', 10),
        )

    tcl_loss = None
    if args.use_temporal:
        temporal_kwargs = dict(
            temperature=args.tcl_temperature,
            patch_temperature=getattr(args, 'tcl_patch_temperature', 0.1),
            spatial_sigma_base=getattr(args, 'tcl_spatial_sigma_base', 2.0),
            spatial_sigma_scale=getattr(args, 'tcl_spatial_sigma_scale', 0.5),
        )
        if getattr(args, 'temporal_loss_type', 'patc') == 'patc':
            tcl_loss = PATCLoss(
                **temporal_kwargs,
                geometry_blend=getattr(args, 'patc_geometry_blend', 0.7),
                confidence_power=getattr(args, 'patc_confidence_power', 1.5),
                min_confidence=getattr(args, 'patc_min_confidence', 0.05),
                gap_weight_scale=getattr(args, 'patc_gap_weight_scale', 0.5),
            )
        else:
            tcl_loss = TCLLoss(**temporal_kwargs)

    combined = PretrainingLoss(
        dino_loss=dino_loss,
        mgcl_loss=mgcl_loss,
        view_loss=view_loss,
        cross_modal_pair_loss=align_rgbir_loss,
        cross_modal_patch_loss=align_rgbir_patch_loss,
        cross_modal_mgcl_loss=cross_modal_mgcl_loss,
        view_bridge_loss=bridge_loss,
        tcl_loss=tcl_loss,
        gray_bridge_loss=gray_bridge_loss,
        w_dino_video=getattr(args, 'w_dino_video', 0.2),
        w_mgcl=args.w_mgcl,
        w_view=args.w_view,
        w_bridge=args.w_bridge,
        w_align_rgbir=getattr(args, 'w_align_rgbir', 0.0),
        w_align_rgbir_patch=getattr(args, 'w_align_rgbir_patch', 0.0),
        w_align_mgcl=getattr(args, 'w_align_mgcl', 0.0),
        w_gray_bridge=getattr(args, 'w_gray_bridge', 0.0),
        align_ramp_epochs=getattr(args, 'align_ramp_epochs', 0),
        total_epochs=args.epochs,
        w_tcl=args.w_tcl,
        w_tcl_patch=getattr(args, 'w_tcl_patch', 0.5),
        adaptive_weighting=getattr(args, 'adaptive_weighting', False),
    )

    return combined


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(student, teacher, mg_student, mg_teacher,
                    view_bridge, temporal_attn, clip_projector,
                    loss_fn, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule,
                    aux_loss_schedule, video_aux_loss_schedule,
                    epoch, fp16_scaler, args,
                    video_data_loader=None):
    """Train for one epoch with unified single forward pass.

    All crops (2 global + N local + optional view crop) are concatenated
    into ONE batch and processed through a SINGLE student forward pass
    and a SINGLE teacher forward pass. Outputs are sliced and routed
    to their respective loss functions.
    """
    student.train()
    mg_student.train()
    view_bridge.train()
    if temporal_attn is not None:
        temporal_attn.train()
    if clip_projector is not None:
        clip_projector.train()

    metric_logger = {k: SmoothedValue(window_size=20) for k in [
        'total', 'dino', 'mgcl', 'view', 'bridge', 'bridge_entropy', 'bridge_kl',
        'align_rgbir', 'align_rgbir_patch',
        'align_mgcl', 'align_mgcl_token', 'align_mgcl_object', 'align_mgcl_image',
        'clip_align', 'clip_video',
        'dino_video', 'tcl', 'tcl_patch', 'mgcl_video', 'bridge_video',
        'bridge_video_entropy', 'bridge_video_kl',
        'lr', 'grad_cos_sim',
        'ew_dino_video', 'ew_mgcl', 'ew_view', 'ew_bridge',
        'ew_align_rgbir', 'ew_align_rgbir_patch', 'ew_align_mgcl',
        'ew_tcl', 'ew_tcl_patch', 'ew_mgcl_video', 'ew_bridge_video',
        'pcgrad_mgcl', 'pcgrad_view', 'pcgrad_bridge',
        'pcgrad_align_rgbir', 'pcgrad_align_rgbir_patch', 'pcgrad_align_mgcl',
        'pcgrad_dino_video', 'pcgrad_tcl', 'pcgrad_tcl_patch',
        'pcgrad_mgcl_video', 'pcgrad_bridge_video']}
    temporal_loss_label = getattr(getattr(loss_fn, 'tcl_loss', None), 'loss_name', 'tcl')
    temporal_patch_label = getattr(getattr(loss_fn, 'tcl_loss', None), 'patch_loss_name', 'tcl_patch')

    # Video iterator: wraps around when exhausted
    video_iter = None
    video_step_interval = 1
    if video_data_loader is not None:
        video_iter = iter(video_data_loader)
        if getattr(args, 'video_step_interval', 0) > 0:
            video_step_interval = args.video_step_interval
        else:
            video_step_interval = max(
                1, math.ceil(len(data_loader) / max(1, len(video_data_loader))))

    accum_steps = getattr(args, 'accumulate_grad_batches', 1) or 1

    for it, (crops, view_crop, crop_masks, view_mask,
             pair_anchor, base_masks) in enumerate(data_loader):
        global_it = len(data_loader) * epoch + it
        is_accum_step = ((it + 1) % accum_steps == 0) or (it + 1 == len(data_loader))

        # Safely get backbone
        if isinstance(student, DDP):
            student_backbone = student.module.backbone
            student_head = student.module.head
        else:
            student_backbone = student.backbone
            student_head = student.head

        if isinstance(teacher, DDP):
            teacher_model = teacher.module
        else:
            teacher_model = teacher

        # Update learning rate and weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            lr_val = lr_schedule[global_it]
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr_val * scale
            if i == 0:
                param_group["weight_decay"] = wd_schedule[global_it]
        aux_weight_scale = float(aux_loss_schedule[global_it])
        video_aux_weight_scale = float(video_aux_loss_schedule[global_it])

        # Move to GPU
        crops = [c.cuda(non_blocking=True) for c in crops]
        crop_masks = [m.cuda(non_blocking=True) for m in crop_masks]
        if view_crop is not None:
            view_crop = view_crop.cuda(non_blocking=True)
        if view_mask is not None:
            view_mask = view_mask.cuda(non_blocking=True)
        if pair_anchor is not None:
            pair_anchor = pair_anchor.cuda(non_blocking=True)
        base_masks = base_masks.cuda(non_blocking=True)

        B = crops[0].shape[0]
        n_global = 2
        n_local = len(crops) - n_global
        has_view = view_crop is not None

        # --- Build unified batch: [global_1, global_2, local_1..N, view_crop] ---
        # Group by resolution for efficient batching
        all_crops = list(crops)
        if has_view:
            all_crops.append(view_crop)

        # Compute number of crops per resolution group
        # Global crops: global_crop_size, Local crops: local_crop_size
        # View crop: same as global_crop_size
        n_global_group = n_global + (1 if has_view else 0)  # globals + view share same resolution
        n_local_group = n_local

        # --- UNIFIED STUDENT FORWARD PASS (single backbone call per resolution group) ---
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            # Group crops by resolution and process in batched groups
            # Global-resolution group: global_1, global_2, [view_crop]
            global_group_crops = crops[:n_global]
            if has_view:
                global_group_crops = global_group_crops + [view_crop]

            global_batch = torch.cat(global_group_crops, dim=0)  # [B*(2+view), 4, H, W]
            global_mask_groups = crop_masks[:n_global]
            if has_view:
                global_mask_groups = global_mask_groups + [view_mask]
            global_masks = torch.cat(global_mask_groups, dim=0)

            student_global_all = student_backbone(
                global_batch, return_all_tokens=True,
                modality_masks=global_masks)  # [B*(2+view), N+1, d]

            # Snapshot Stage1 features before any subsequent forward overwrites them
            _cached_stage1_rgb = None
            _cached_stage1_ir = None
            if (hasattr(student_backbone, '_use_two_stage')
                    and student_backbone._use_two_stage
                    and student_backbone._stage1_rgb is not None):
                # Clone but keep grad graph alive for CLIP backward
                _cached_stage1_rgb = student_backbone._stage1_rgb.clone()
                _cached_stage1_ir = student_backbone._stage1_ir.clone()

            student_global_cls = student_global_all[:, 0]    # [B*(2+view), d]
            student_global_patches = student_global_all[:, 1:]  # [B*(2+view), N, d]

            # Local-resolution group (if any)
            student_local_cls = None
            if n_local > 0:
                local_batch = torch.cat(crops[n_global:], dim=0)  # [B*n_local, 4, h, w]
                local_masks = torch.cat(crop_masks[n_global:], dim=0)

                student_local_all = student_backbone(
                    local_batch, return_all_tokens=True,
                    modality_masks=local_masks)  # [B*n_local, M+1, d]
                student_local_cls = student_local_all[:, 0]  # [B*n_local, d]

            # --- Project CLS through DINO head ---
            # Concatenate all CLS tokens: [B*n_global, d] ++ [B*n_local, d]
            all_student_cls = student_global_cls[:B * n_global]  # exclude view crop CLS from DINO
            if student_local_cls is not None:
                all_student_cls = torch.cat([all_student_cls, student_local_cls], dim=0)

            student_out = student_head(all_student_cls)  # [B*(n_global+n_local), out_dim]

            # --- UNIFIED TEACHER FORWARD PASS (global crops only) ---
            with torch.no_grad():
                teacher_global_batch = torch.cat(crops[:n_global], dim=0)  # [B*2, 4, H, W]
                teacher_masks = torch.cat(crop_masks[:n_global], dim=0)

                teacher_global_all = teacher_model.backbone(
                    teacher_global_batch, return_all_tokens=True,
                    modality_masks=teacher_masks)  # [B*2, N+1, d]

                teacher_cls = teacher_global_all[:, 0]     # [B*2, d]
                teacher_patches = teacher_global_all[:, 1:]  # [B*2, N, d]

                teacher_out = teacher_model.head(teacher_cls)  # [B*2, out_dim]

            # --- Slice student outputs for each loss ---

            # Student global patch tokens (first 2 global crops only, no view)
            student_global_patch_2 = student_global_patches[:B * n_global]  # [B*2, N, d]

            # Multi-granularity features (global crops only)
            with torch.no_grad():
                teacher_assignments = mg_teacher.sinkhorn(
                    teacher_patches, update_centers=False
                )
                teacher_mg = mg_teacher(teacher_patches, assignments=teacher_assignments)
            student_mg = mg_student(
                student_global_patch_2, assignments=teacher_assignments.detach())

            # --- Viewpoint invariance ---
            feat_original = student_global_patches[:B].mean(dim=1)  # [B, d] from global_1
            feat_global2 = student_global_patches[B:B * 2].mean(dim=1)  # [B, d]

            feat_view = None
            feat_original_view = None
            feat_view_loss = None
            view_pair_mask = None
            align_rgb_feat = None
            align_ir_feat = None
            align_rgb_patches = None
            align_ir_patches = None
            align_teacher_rgb_patches = None
            align_teacher_ir_patches = None
            align_patch_grid_size = None
            if has_view:
                feat_view = student_global_patches[B * n_global:B * n_global_group].mean(dim=1)  # [B, d]
                view_pair_mask = (crop_masks[0] == view_mask).all(dim=1)
                if view_pair_mask.any():
                    feat_original_view = feat_original[view_pair_mask]
                    feat_view_loss = feat_view[view_pair_mask]

            if (args.w_align_rgbir > 0 or args.w_align_rgbir_patch > 0
                    or getattr(args, 'w_align_mgcl', 0.0) > 0) and pair_anchor is not None:
                paired_mask = (base_masks[:, 0] > 0.5) & (base_masks[:, 1] > 0.5)
                if paired_mask.any():
                    paired_anchor = pair_anchor[paired_mask]
                    n_paired = paired_anchor.shape[0]
                    rgb_only = paired_anchor.clone()
                    rgb_only[:, 3:] = 0
                    ir_only = paired_anchor.clone()
                    ir_only[:, :3] = 0
                    align_batch = torch.cat([rgb_only, ir_only], dim=0)
                    rgb_mask = base_masks.new_tensor([1.0, 0.0]).expand(n_paired, -1)
                    ir_mask = base_masks.new_tensor([0.0, 1.0]).expand(n_paired, -1)
                    align_masks = torch.cat([rgb_mask, ir_mask], dim=0)
                    align_tokens = student_backbone(
                        align_batch, return_all_tokens=True,
                        modality_masks=align_masks)
                    align_patch_tokens = align_tokens[:, 1:]
                    align_features = align_patch_tokens.mean(dim=1)
                    align_rgb_feat = align_features[:n_paired]
                    align_ir_feat = align_features[n_paired:]
                    if args.w_align_rgbir_patch > 0:
                        align_rgb_patches = align_patch_tokens[:n_paired]
                        align_ir_patches = align_patch_tokens[n_paired:]
                        n_patches = align_rgb_patches.shape[1]
                        align_patch_grid_size = int(round(math.sqrt(n_patches)))
                        if align_patch_grid_size * align_patch_grid_size != n_patches:
                            raise ValueError(
                                f"Cross-modal patch alignment expects square patch grids, got N={n_patches}"
                            )
                        with torch.no_grad():
                            teacher_align_tokens = teacher_model.backbone(
                                align_batch, return_all_tokens=True,
                                modality_masks=align_masks)
                            teacher_align_patch_tokens = teacher_align_tokens[:, 1:]
                            align_teacher_rgb_patches = teacher_align_patch_tokens[:n_paired]
                            align_teacher_ir_patches = teacher_align_patch_tokens[n_paired:]

            # --- CLIP cross-modal alignment (image) ---
            clip_align_loss = None
            if (clip_projector is not None
                    and getattr(args, 'w_clip_align', 0) > 0
                    and pair_anchor is not None):
                # In two-stage mode: use SNAPSHOT of Stage1 features from the
                # main global forward (before any subsequent forward overwrites)
                if _cached_stage1_rgb is not None:
                    paired_mask_clip = (base_masks[:, 0] > 0.5) & (base_masks[:, 1] > 0.5)
                    if paired_mask_clip.any():
                        # Snapshot is from global_batch; first B = first global crop
                        s1_rgb = _cached_stage1_rgb[:B]  # [B, N+1, d]
                        s1_ir = _cached_stage1_ir[:B]
                        s1_rgb_paired = s1_rgb[paired_mask_clip]
                        s1_ir_paired = s1_ir[paired_mask_clip]
                        # Use CLS token (index 0) — already aggregates global info
                        # Cast to float32 for projector stability (snapshot may be fp16 under autocast)
                        clip_rgb_feat = s1_rgb_paired[:, 0].float()  # [n, d]
                        clip_ir_feat = s1_ir_paired[:, 0].float()
                        if clip_rgb_feat.shape[0] > 1:
                            clip_align_loss = clip_projector(clip_rgb_feat, clip_ir_feat)
                else:
                    # Legacy single-stream mode: use align_rgb_feat/ir_feat
                    if align_rgb_feat is not None and align_ir_feat is not None:
                        if align_rgb_feat.shape[0] > 1:
                            clip_align_loss = clip_projector(
                                align_rgb_feat.float(), align_ir_feat.float())
            align_rgb_mg = None
            align_ir_mg = None
            align_teacher_fused_mg = None
            if (getattr(args, 'w_align_mgcl', 0.0) > 0
                    and pair_anchor is not None):
                paired_mask = (base_masks[:, 0] > 0.5) & (base_masks[:, 1] > 0.5)
                if paired_mask.any():
                    paired_anchor_mgcl = pair_anchor[paired_mask]
                    n_paired_mg = paired_anchor_mgcl.shape[0]

                    # Student single-modal patches (reuse if already computed)
                    if align_rgb_feat is not None:
                        # align_patch_tokens already computed above
                        s_rgb_patches = align_patch_tokens[:n_paired_mg]
                        s_ir_patches = align_patch_tokens[n_paired_mg:]
                    else:
                        rgb_only = paired_anchor_mgcl.clone()
                        rgb_only[:, 3:] = 0
                        ir_only = paired_anchor_mgcl.clone()
                        ir_only[:, :3] = 0
                        align_batch_mg = torch.cat([rgb_only, ir_only], dim=0)
                        rgb_mask = base_masks.new_tensor([1.0, 0.0]).expand(n_paired_mg, -1)
                        ir_mask = base_masks.new_tensor([0.0, 1.0]).expand(n_paired_mg, -1)
                        align_masks_mg = torch.cat([rgb_mask, ir_mask], dim=0)
                        align_tokens_mg = student_backbone(
                            align_batch_mg, return_all_tokens=True,
                            modality_masks=align_masks_mg)
                        s_rgb_patches = align_tokens_mg[:, 1:][:n_paired_mg]
                        s_ir_patches = align_tokens_mg[:, 1:][n_paired_mg:]

                    # Teacher fused forward (RGB+IR together)
                    with torch.no_grad():
                        fused_mask = base_masks.new_tensor([1.0, 1.0]).expand(n_paired_mg, -1)
                        teacher_fused_tokens = teacher_model.backbone(
                            paired_anchor_mgcl, return_all_tokens=True,
                            modality_masks=fused_mask)
                        teacher_fused_patches = teacher_fused_tokens[:, 1:]

                        # Teacher Sinkhorn assignments (anchor for all 3 views)
                        fused_assignments = mg_teacher.sinkhorn(
                            teacher_fused_patches, update_centers=False)

                        # Teacher fused MG features
                        align_teacher_fused_mg = mg_teacher(
                            teacher_fused_patches, assignments=fused_assignments)

                    # Student single-modal MG features (using teacher's fused assignments)
                    align_rgb_mg = mg_student(
                        s_rgb_patches, assignments=fused_assignments.detach())
                    align_ir_mg = mg_student(
                        s_ir_patches, assignments=fused_assignments.detach())

            # --- Grayscale bridge features ---
            align_gray_feat = None
            if (getattr(args, 'use_grayscale_bridge', False)
                    and align_rgb_feat is not None and align_ir_feat is not None
                    and pair_anchor is not None):
                paired_mask = (base_masks[:, 0] > 0.5) & (base_masks[:, 1] > 0.5)
                if paired_mask.any():
                    paired_anchor_gray = pair_anchor[paired_mask]
                    n_paired_g = paired_anchor_gray.shape[0]
                    # Generate grayscale: 0.299*R + 0.587*G + 0.114*B
                    gray_ch = (0.299 * paired_anchor_gray[:, 0:1]
                               + 0.587 * paired_anchor_gray[:, 1:2]
                               + 0.114 * paired_anchor_gray[:, 2:3])
                    # Place grayscale in IR channel slot, zero RGB
                    gray_input = torch.zeros_like(paired_anchor_gray)
                    gray_input[:, 3:] = gray_ch
                    gray_mask = base_masks.new_tensor([0.0, 1.0]).expand(n_paired_g, -1)
                    gray_tokens = student_backbone(
                        gray_input, return_all_tokens=True,
                        modality_masks=gray_mask)
                    align_gray_feat = gray_tokens[:, 1:].mean(dim=1)

            # Skip the entire ViewBridge path when its weight is disabled.
            bridge_logits_list = None
            vb_module = None
            bridge_student_logits_list = None
            bridge_teacher_z_list = None
            video_bridge_student_logits = None
            video_bridge_teacher_z_list = None
            if args.w_bridge > 0:
                vb_module = (view_bridge.module
                             if isinstance(view_bridge, DDP) else view_bridge)

                # Student logits
                _, bridge_logits_g1 = view_bridge(feat_original)
                _, bridge_logits_g2 = view_bridge(feat_global2)
                bridge_student_logits_list = [bridge_logits_g1, bridge_logits_g2]
                bridge_mask_list = [crop_masks[0], crop_masks[1]]

                if has_view:
                    _, bridge_logits_view = view_bridge(feat_view)
                    bridge_student_logits_list.append(bridge_logits_view)
                    bridge_mask_list.append(view_mask)

                # Teacher projected features. Build shared targets only within
                # each modality group so bridge focuses on cross-view/domain
                # consistency instead of forcing RGB-only and IR-only views
                # onto the same image-level target.
                with torch.no_grad():
                    teacher_feat_g1 = teacher_patches[:B].mean(dim=1)
                    teacher_feat_g2 = teacher_patches[B:B * 2].mean(dim=1)

                    teacher_z_g1, _ = vb_module.teacher_forward(teacher_feat_g1)
                    teacher_z_g2, _ = vb_module.teacher_forward(teacher_feat_g2)
                    teacher_z_list = [teacher_z_g1, teacher_z_g2]
                    if has_view:
                        teacher_view_all = teacher_model.backbone(
                            view_crop, return_all_tokens=True,
                            modality_masks=view_mask)
                        teacher_feat_view = teacher_view_all[:, 1:].mean(dim=1)
                        teacher_z_view, _ = vb_module.teacher_forward(teacher_feat_view)
                        teacher_z_list.append(teacher_z_view)
                    bridge_teacher_z_list = build_modality_grouped_bridge_teacher_z(
                        teacher_z_list,
                        bridge_mask_list,
                    )

            # --- VIDEO: unified forward (DINO + TCL + MGCL + Bridge) ---
            tcl_features = None
            tcl_num_frames = 1
            video_student_out = None
            video_teacher_out = None
            video_ncrops = None
            video_teacher_indices = None
            video_student_patches_for_tcl = None
            video_teacher_patches_for_tcl = None
            video_timestamps_for_tcl = None
            video_student_mg = None
            video_teacher_mg = None
            video_bridge_logits_list = None
            video_grid_size = 16
            clip_video_loss = None

            use_video_batch = (
                args.use_temporal
                and temporal_attn is not None
                and video_iter is not None
                and (it % video_step_interval == 0)
            )
            if use_video_batch:
                try:
                    video_batch = next(video_iter)
                except StopIteration:
                    video_iter = iter(video_data_loader)
                    video_batch = next(video_iter)

                video_frames, video_masks, timestamps = video_batch
                video_frames = video_frames.cuda(non_blocking=True)  # [Bv, T, 4, H, W]
                video_masks = video_masks.cuda(non_blocking=True)     # [Bv, 2]
                timestamps = timestamps.cuda(non_blocking=True)       # [Bv, T]

                B_v, T, C_v, H_v, W_v = video_frames.shape
                flat_frames = video_frames.reshape(B_v * T, C_v, H_v, W_v)
                flat_masks = video_masks.unsqueeze(1).expand(-1, T, -1).reshape(B_v * T, 2)

                # --- Student forward (same backbone as image) ---
                student_video_all = student_backbone(
                    flat_frames, return_all_tokens=True,
                    modality_masks=flat_masks)  # [Bv*T, N+1, d]

                student_video_patches = student_video_all[:, 1:]  # [Bv*T, N, d]
                N_v = student_video_patches.shape[1]
                video_grid_size = int(math.sqrt(N_v))

                # --- CLIP cross-modal alignment (video) ---
                clip_video_loss = None
                if (clip_projector is not None
                        and getattr(args, 'w_clip_video', 0) > 0):
                    video_is_paired = (video_masks[:, 0] > 0.5) & (video_masks[:, 1] > 0.5)
                    if (video_is_paired.any()
                            and hasattr(student_backbone, '_use_two_stage')
                            and student_backbone._use_two_stage
                            and student_backbone._stage1_rgb is not None):
                        # Stage1 outputs from video forward: [Bv*T, N+1, d]
                        v_s1_rgb = student_backbone._stage1_rgb
                        v_s1_ir = student_backbone._stage1_ir
                        # Expand paired mask to per-frame: [Bv] → [Bv*T]
                        v_paired_flat = video_is_paired.unsqueeze(1).expand(-1, T).reshape(-1)
                        if v_paired_flat.any():
                            # Use CLS token, cast to float32
                            v_rgb_feat = v_s1_rgb[v_paired_flat][:, 0].float()
                            v_ir_feat = v_s1_ir[v_paired_flat][:, 0].float()
                            if v_rgb_feat.shape[0] > 1:
                                clip_video_loss = clip_projector(v_rgb_feat, v_ir_feat)

                # Apply TemporalAttention on patch tokens
                student_video_patches = temporal_attn(
                    student_video_patches, num_frames=T, timestamps=timestamps)
                student_video_frame_repr = student_video_patches.mean(dim=1)
                student_video_frame_repr_bt = student_video_frame_repr.reshape(B_v, T, -1)
                student_video_patches_bt = student_video_patches.reshape(B_v, T, N_v, -1)

                # --- Teacher forward: 2 anchor frames (first + middle) ---
                anchor_indices = [0, T // 2]
                video_teacher_indices = list(range(len(anchor_indices)))
                teacher_anchor_frames = video_frames[:, anchor_indices]  # [Bv, 2, C, H, W]
                # DINO expects teacher crops grouped by anchor view, not interleaved by sample.
                teacher_anchor_flat = teacher_anchor_frames.permute(1, 0, 2, 3, 4).reshape(
                    2 * B_v, C_v, H_v, W_v
                )
                teacher_anchor_masks = video_masks.unsqueeze(0).expand(2, -1, -1).reshape(2 * B_v, 2)

                with torch.no_grad():
                    teacher_video_all = teacher_model.backbone(
                        teacher_anchor_flat, return_all_tokens=True,
                        modality_masks=teacher_anchor_masks)  # [Bv*2, N+1, d]
                    teacher_video_all = teacher_video_all.reshape(2, B_v, N_v + 1, -1).permute(1, 0, 2, 3)
                    teacher_video_cls = teacher_video_all[:, :, 0]              # [Bv, 2, d]
                    teacher_video_patches_bt = teacher_video_all[:, :, 1:]      # [Bv, 2, N, d]
                    teacher_video_patches_anchor = teacher_video_patches_bt.permute(1, 0, 2, 3).reshape(
                        2 * B_v, N_v, -1
                    )
                    video_teacher_out = teacher_model.head(
                        teacher_video_cls.permute(1, 0, 2).reshape(2 * B_v, -1)
                    )  # [2*Bv, out_dim], grouped by anchor view

                # --- Video DINO: match teacher anchor views with temporal student anchor representations ---
                student_video_anchor_repr = student_video_frame_repr_bt[:, anchor_indices]
                video_student_out = student_head(
                    student_video_anchor_repr.permute(1, 0, 2).reshape(B_v * len(anchor_indices), -1)
                )  # [2*Bv, out_dim], grouped by anchor view
                video_ncrops = len(anchor_indices)

                # --- Global TCL features ---
                tcl_features = student_video_frame_repr  # [Bv*T, d]
                tcl_num_frames = T

                # --- Patch-level TCL: need teacher patches for all T frames ---
                with torch.no_grad():
                    teacher_video_all_t = teacher_model.backbone(
                        flat_frames, return_all_tokens=True,
                        modality_masks=flat_masks)  # [Bv*T, N+1, d]
                    video_teacher_patches_for_tcl = teacher_video_all_t[:, 1:]
                video_student_patches_for_tcl = student_video_patches
                video_timestamps_for_tcl = timestamps

                # --- Video MGCL (on anchor frames only) ---
                student_video_patch_anchor = (
                    student_video_patches_bt[:, anchor_indices]
                    .permute(1, 0, 2, 3)
                    .reshape(B_v * 2, N_v, -1)
                )
                with torch.no_grad():
                    teacher_video_assignments = mg_teacher.sinkhorn(
                        teacher_video_patches_anchor, update_centers=False
                    )
                    video_teacher_mg = mg_teacher(
                        teacher_video_patches_anchor,
                        assignments=teacher_video_assignments)
                video_student_mg = mg_student(
                    student_video_patch_anchor,
                    assignments=teacher_video_assignments.detach())

                # --- Video Bridge (on anchor frames) ---
                if args.w_bridge > 0:
                    s_vid_feat_g1 = student_video_frame_repr_bt[:, anchor_indices[0]]   # [Bv, d]
                    s_vid_feat_g2 = student_video_frame_repr_bt[:, anchor_indices[1]]   # [Bv, d]

                    _, v_bridge_logits_g1 = view_bridge(s_vid_feat_g1)
                    _, v_bridge_logits_g2 = view_bridge(s_vid_feat_g2)
                    video_bridge_student_logits = [v_bridge_logits_g1, v_bridge_logits_g2]
                    video_bridge_mask_list = [video_masks, video_masks]

                    with torch.no_grad():
                        t_vid_feat_g1 = teacher_video_patches_bt[:, 0].mean(dim=1)
                        t_vid_feat_g2 = teacher_video_patches_bt[:, 1].mean(dim=1)
                        vt_z_g1, _ = vb_module.teacher_forward(t_vid_feat_g1)
                        vt_z_g2, _ = vb_module.teacher_forward(t_vid_feat_g2)
                        video_bridge_teacher_z_list = build_modality_grouped_bridge_teacher_z(
                            [vt_z_g1, vt_z_g2],
                            video_bridge_mask_list,
                        )

            if args.w_bridge > 0 and vb_module is not None:
                with torch.no_grad():
                    all_bridge_teacher_z = []
                    if bridge_teacher_z_list is not None:
                        all_bridge_teacher_z.extend(bridge_teacher_z_list)
                    if video_bridge_teacher_z_list is not None:
                        all_bridge_teacher_z.extend(video_bridge_teacher_z_list)
                    if all_bridge_teacher_z:
                        # Compute one shared target snapshot for image + video
                        # so small video batches do not run a degenerate
                        # balanced Sinkhorn on their own.
                        all_teacher_z = torch.cat(all_bridge_teacher_z, dim=0)
                        all_targets = vb_module.sinkhorn_assignments(all_teacher_z)

                        offset = 0
                        bridge_teacher_targets = None
                        if bridge_teacher_z_list is not None:
                            bridge_teacher_targets = []
                            for teacher_z in bridge_teacher_z_list:
                                next_offset = offset + teacher_z.shape[0]
                                bridge_teacher_targets.append(all_targets[offset:next_offset])
                                offset = next_offset

                        video_bridge_targets = None
                        if video_bridge_teacher_z_list is not None:
                            video_bridge_targets = []
                            for teacher_z in video_bridge_teacher_z_list:
                                next_offset = offset + teacher_z.shape[0]
                                video_bridge_targets.append(all_targets[offset:next_offset])
                                offset = next_offset

                        vb_module.update_prototypes_ema(all_teacher_z, assignments=all_targets)

                        if bridge_student_logits_list is not None and bridge_teacher_targets is not None:
                            bridge_logits_list = (
                                bridge_student_logits_list, bridge_teacher_targets
                            )
                        if video_bridge_student_logits is not None and video_bridge_targets is not None:
                            video_bridge_logits_list = (
                                video_bridge_student_logits,
                                video_bridge_targets,
                            )

            with torch.no_grad():
                mg_center_features = [teacher_patches]
                mg_center_assignments = [teacher_assignments]
                if video_teacher_mg is not None:
                    mg_center_features.append(teacher_video_patches_anchor)
                    mg_center_assignments.append(teacher_video_assignments)
                mg_teacher.sinkhorn.update_centers_multi(
                    mg_center_features, mg_center_assignments
                )

            # --- Combined loss ---
            total_loss, loss_dict, individual_losses = loss_fn(
                student_out=student_out,
                teacher_out=teacher_out,
                epoch=epoch,
                student_mg=student_mg,
                teacher_mg=teacher_mg,
                feat_original=feat_original_view,
                feat_view=feat_view_loss,
                align_rgb_feat=align_rgb_feat,
                align_ir_feat=align_ir_feat,
                align_gray_feat=align_gray_feat,
                align_rgb_patches=align_rgb_patches,
                align_ir_patches=align_ir_patches,
                align_teacher_rgb_patches=align_teacher_rgb_patches,
                align_teacher_ir_patches=align_teacher_ir_patches,
                align_patch_grid_size=align_patch_grid_size,
                align_rgb_mg=align_rgb_mg,
                align_ir_mg=align_ir_mg,
                align_teacher_fused_mg=align_teacher_fused_mg,
                bridge_logits_list=bridge_logits_list,
                tcl_features=tcl_features,
                num_frames=tcl_num_frames,
                aux_weight_scale=aux_weight_scale,
                video_aux_weight_scale=video_aux_weight_scale,
                video_student_out=video_student_out,
                video_teacher_out=video_teacher_out,
                video_ncrops=video_ncrops,
                video_teacher_indices=video_teacher_indices,
                video_student_patches=video_student_patches_for_tcl,
                video_teacher_patches=video_teacher_patches_for_tcl,
                video_timestamps=video_timestamps_for_tcl,
                video_student_mg=video_student_mg,
                video_teacher_mg=video_teacher_mg,
                video_bridge_logits_list=video_bridge_logits_list,
                grid_size=video_grid_size,
            )

        # --- Add CLIP losses to total and individual_losses ---
        w_clip = getattr(args, 'w_clip_align', 0.0)
        w_clip_v = getattr(args, 'w_clip_video', 0.0)
        if clip_align_loss is not None and w_clip > 0:
            total_loss = total_loss + w_clip * clip_align_loss
            individual_losses['clip_align'] = w_clip * clip_align_loss
            loss_dict['clip_align'] = clip_align_loss.item()
        else:
            loss_dict['clip_align'] = 0.0
        if clip_video_loss is not None and w_clip_v > 0:
            total_loss = total_loss + w_clip_v * clip_video_loss
            individual_losses['clip_video'] = w_clip_v * clip_video_loss
            loss_dict['clip_video'] = clip_video_loss.item()
        else:
            loss_dict['clip_video'] = 0.0

        # --- Backward pass ---
        scaled_total = total_loss / accum_steps

        if it % accum_steps == 0:
            optimizer.zero_grad()

        # Build shared param list for gradient operations
        param_list = (
            list(student.parameters()) +
            list(mg_student.parameters()) +
            list(view_bridge.parameters()) +
            list(loss_fn.parameters()))
        if temporal_attn is not None:
            param_list += list(temporal_attn.parameters())
        if clip_projector is not None:
            param_list += list(clip_projector.parameters())
        shared_params = [p for p in param_list if p.requires_grad]

        use_pcgrad = args.use_pcgrad
        conflict_info = {}

        if use_pcgrad and len(individual_losses) > 1:
            # PCGrad: image DINO is the anchor objective; video losses are auxiliaries.
            dino_loss_val = individual_losses.get('dino', torch.tensor(0.0, device=total_loss.device)) / accum_steps

            aux_losses_for_pcgrad = {}
            for key, (weight, loss_tensor) in loss_fn.get_pcgrad_aux_terms(
                    individual_losses,
                    aux_weight_scale=aux_weight_scale,
                    video_aux_weight_scale=video_aux_weight_scale).items():
                aux_losses_for_pcgrad[key] = (weight / accum_steps, loss_tensor)

            if aux_losses_for_pcgrad:
                conflict_info = pcgrad_backward(
                    dino_loss_val, aux_losses_for_pcgrad, shared_params, fp16_scaler)
            else:
                if fp16_scaler is not None:
                    fp16_scaler.scale(scaled_total).backward()
                else:
                    scaled_total.backward()
        else:
            # Standard backward (no PCGrad)
            if fp16_scaler is not None:
                fp16_scaler.scale(scaled_total).backward()
            else:
                scaled_total.backward()

        # Sync uncertainty weighting gradients across DDP (loss_fn not in DDP)
        if getattr(args, 'adaptive_weighting', False) and dist.is_initialized():
            for p in loss_fn.uncertainty.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                dist.all_reduce(p.grad)
                p.grad.div_(dist.get_world_size())

        # Gradient clipping and optimizer step (only on accumulation boundary)
        if is_accum_step:
            if fp16_scaler is not None:
                if args.clip_grad > 0:
                    fp16_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(shared_params, args.clip_grad)
                cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(shared_params, args.clip_grad)
                cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                optimizer.step()

        # --- EMA update teacher (only on accumulation boundary) ---
        if is_accum_step:
            with torch.no_grad():
                m = momentum_schedule[global_it]
                for ps, pt in zip(student.parameters(),
                                  teacher.parameters()):
                    pt.data.mul_(m).add_((1 - m) * ps.detach().data)
                for ps, pt in zip(mg_student.parameters(),
                                  mg_teacher.parameters()):
                    pt.data.mul_(m).add_((1 - m) * ps.detach().data)
                vb_module = (view_bridge.module
                             if isinstance(view_bridge, DDP) else view_bridge)
                vb_module.update_teacher_projector(m)

        # --- Logging ---
        for k, v in loss_dict.items():
            if k in metric_logger:
                metric_logger[k].update(v)
        metric_logger['lr'].update(optimizer.param_groups[0]['lr'])

        # Log PCGrad conflict info
        for name, cos_sim in conflict_info.items():
            key = f'pcgrad_{name}'
            if key in metric_logger:
                metric_logger[key].update(cos_sim)

        if it % args.log_every == 0 and is_main_process():
            log_str = (
                f"Epoch [{epoch}][{it}/{len(data_loader)}]  "
                f"loss: {metric_logger['total'].avg:.4f}  "
                f"dino: {metric_logger['dino'].avg:.4f}  "
                f"mgcl: {metric_logger['mgcl'].avg:.4f}  "
                f"align: {metric_logger['align_rgbir'].avg:.4f}  "
                f"align_p: {metric_logger['align_rgbir_patch'].avg:.4f}  "
                f"align_mg: {metric_logger['align_mgcl'].avg:.4f}  "
                f"view: {metric_logger['view'].avg:.4f}  "
                f"bridge: {metric_logger['bridge'].avg:.4f}  "
                f"bridge_h: {metric_logger['bridge_entropy'].avg:.4f}  "
                f"bridge_kl: {metric_logger['bridge_kl'].avg:.4f}  "
                f"lr: {metric_logger['lr'].avg:.6f}"
            )
            if args.use_temporal:
                log_str += (
                    f"  {temporal_loss_label}: {loss_dict.get('tcl', 0.0):.4f}"
                    f"  {temporal_patch_label}: {loss_dict.get('tcl_patch', 0.0):.4f}"
                    f"  dino_v: {loss_dict.get('dino_video', 0.0):.4f}"
                    f"  bridge_v: {loss_dict.get('bridge_video', 0.0):.4f}"
                )
            if getattr(args, 'w_clip_align', 0) > 0 or getattr(args, 'w_clip_video', 0) > 0:
                log_str += (
                    f"  clip: {loss_dict.get('clip_align', 0.0):.4f}"
                    f"  clip_v: {loss_dict.get('clip_video', 0.0):.4f}"
                )
            if args.use_pcgrad and conflict_info:
                cos_strs = [f"{k}={v:.3f}" for k, v in conflict_info.items()]
                log_str += f"  pcgrad: [{', '.join(cos_strs)}]"
            if getattr(args, 'adaptive_weighting', False):
                log_str += (
                    f"  ew: dino_v={metric_logger['ew_dino_video'].avg:.3f}"
                    f" mgcl={metric_logger['ew_mgcl'].avg:.3f}"
                    f" align={metric_logger['ew_align_rgbir'].avg:.3f}"
                    f" align_p={metric_logger['ew_align_rgbir_patch'].avg:.3f}"
                    f" view={metric_logger['ew_view'].avg:.3f}"
                    f" bridge={metric_logger['ew_bridge'].avg:.3f}"
                )
            print(log_str)

            # Wandb per-iteration logging
            if _WANDB_AVAILABLE and args.wandb_key and wandb.run is not None:
                log_data = {
                    'train/total_loss': metric_logger['total'].avg,
                    'train/dino': metric_logger['dino'].avg,
                    'train/mgcl': metric_logger['mgcl'].avg,
                    'train/align_rgbir': metric_logger['align_rgbir'].avg,
                    'train/align_rgbir_patch': metric_logger['align_rgbir_patch'].avg,
                    'train/align_mgcl': metric_logger['align_mgcl'].avg,
                    'train/align_mgcl_token': metric_logger['align_mgcl_token'].avg,
                    'train/align_mgcl_object': metric_logger['align_mgcl_object'].avg,
                    'train/align_mgcl_image': metric_logger['align_mgcl_image'].avg,
                    'train/view': metric_logger['view'].avg,
                    'train/bridge': metric_logger['bridge'].avg,
                    'train/bridge_entropy': metric_logger['bridge_entropy'].avg,
                    'train/bridge_kl': metric_logger['bridge_kl'].avg,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                }
                # Video loss keys
                for vk in ('dino_video', 'tcl', 'tcl_patch',
                            'mgcl_video', 'bridge_video'):
                    if metric_logger[vk].count > 0:
                        log_data[f'train/{vk}'] = metric_logger[vk].avg
                # CLIP loss keys
                for ck in ('clip_align', 'clip_video'):
                    if metric_logger[ck].count > 0:
                        log_data[f'train/{ck}'] = metric_logger[ck].avg
                if args.use_temporal:
                    if metric_logger['tcl'].count > 0 and temporal_loss_label != 'tcl':
                        log_data[f'train/{temporal_loss_label}'] = metric_logger['tcl'].avg
                    if metric_logger['tcl_patch'].count > 0 and temporal_patch_label != 'tcl_patch':
                        log_data[f'train/{temporal_patch_label}'] = metric_logger['tcl_patch'].avg
                if getattr(args, 'adaptive_weighting', False):
                    for ew_key in (
                            'ew_dino_video', 'ew_mgcl', 'ew_align_rgbir', 'ew_align_rgbir_patch',
                            'ew_align_mgcl', 'ew_view', 'ew_bridge',
                            'ew_tcl', 'ew_tcl_patch', 'ew_mgcl_video', 'ew_bridge_video'):
                        if metric_logger[ew_key].count > 0:
                            log_data[f'train/{ew_key}'] = metric_logger[ew_key].avg
                for name, cos_sim in conflict_info.items():
                    log_data[f'train/pcgrad_{name}'] = cos_sim
                wandb.log(log_data, step=global_it)

    return {k: v.global_avg for k, v in metric_logger.items()}


# ============================================================================
# Main
# ============================================================================

def main(args):
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed(
        allow_cpu=args.dry_run_samples > 0)

    # Fix random seeds
    torch.manual_seed(42 + rank)
    cudnn.benchmark = True

    # Output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Build data
    # =========================================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    )

    transform = DataAugmentationDINO_RGBIR(
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        local_crops_number=args.local_crops_number,
        use_view_augmentation=args.use_view_aug,
        return_pair_anchor=(
            getattr(args, 'w_align_rgbir', 0.0) > 0
            or getattr(args, 'w_align_rgbir_patch', 0.0) > 0
            or getattr(args, 'w_align_mgcl', 0.0) > 0
        ),
    )

    dataset = MultiModalDroneDataset(
        data_source=args.data_path,
        transform=transform,
        mode=args.data_mode,
        strict_loading=args.strict_loading,
        max_load_fail_ratio=args.max_load_fail_ratio,
        dry_run_samples=args.dry_run_samples,
        alignment_offsets_path=args.alignment_offsets or None,
    )

    type_sampling_probs = normalize_type_sampling_weights(
        getattr(args, 'type_sampling_weights', None))
    if type_sampling_probs is not None:
        try:
            sampler = TypeBalancedDistributedSampler(
                dataset,
                type_weights=type_sampling_probs,
                seed=getattr(args, 'type_sampling_seed', 0),
            )
        except RuntimeError as exc:
            logger.warning(
                "Type-balanced image sampler unavailable (%s). Falling back to default sampler.",
                exc,
            )
            type_sampling_probs = None
            if dist.is_initialized():
                sampler = torch.utils.data.DistributedSampler(
                    dataset, shuffle=True)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)
    elif dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    image_num_workers = resolve_num_workers(args.num_workers)
    image_loader_kwargs = {
        'sampler': sampler,
        'batch_size': args.batch_size_per_gpu,
        'num_workers': image_num_workers,
        'pin_memory': True,
        'drop_last': True,
        'collate_fn': collate_multimodal,
        'worker_init_fn': dataloader_worker_init_fn if image_num_workers > 0 else None,
    }
    if image_num_workers > 0:
        image_loader_kwargs['persistent_workers'] = args.persistent_workers
        image_loader_kwargs['prefetch_factor'] = max(1, int(args.prefetch_factor))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        **image_loader_kwargs,
    )

    print(f"Dataset: {len(dataset)} samples, "
          f"DataLoader: {len(data_loader)} batches/epoch")

    # --- Video DataLoader (optional, phase 2) ---
    video_data_loader = None
    video_sampler = None
    if args.video_manifest:
        # Auto-enable temporal mode when video manifest is provided
        args.use_temporal = True
        if args.num_frames <= 1:
            args.num_frames = 4

        video_dataset = build_video_dataset(
            manifest_path=args.video_manifest,
            num_frames=args.num_frames,
            target_size=args.video_target_size,
            alignment_offsets_path=args.alignment_offsets or None,
        )

        if dist.is_initialized():
            video_sampler = torch.utils.data.DistributedSampler(
                video_dataset, shuffle=True)
        else:
            video_sampler = torch.utils.data.RandomSampler(video_dataset)

        video_num_workers = resolve_num_workers(args.video_num_workers)
        video_loader_kwargs = {
            'sampler': video_sampler,
            'batch_size': args.video_batch_size,
            'num_workers': video_num_workers,
            'pin_memory': True,
            'drop_last': True,
            'collate_fn': collate_video,
            'worker_init_fn': dataloader_worker_init_fn if video_num_workers > 0 else None,
        }
        if video_num_workers > 0:
            video_loader_kwargs['persistent_workers'] = args.video_persistent_workers
            video_loader_kwargs['prefetch_factor'] = max(1, int(args.video_prefetch_factor))

        video_data_loader = torch.utils.data.DataLoader(
            video_dataset,
            **video_loader_kwargs,
        )

        if is_main_process():
            print(f"\n--- Video Data ---")
            print(f"  Video manifest: {args.video_manifest}")
            print(f"  Video samples: {len(video_dataset)}")
            print(f"  Video batch size: {args.video_batch_size}")
            print(f"  Video num_frames: {args.num_frames}")
            print(f"  Video target_size: {args.video_target_size}")
            print(f"  Video workers: {video_num_workers}, "
                  f"persistent: {args.video_persistent_workers}, "
                  f"prefetch: {max(1, int(args.video_prefetch_factor)) if video_num_workers > 0 else 'n/a'}")
            print(f"  Video batches/epoch: {len(video_data_loader)}")
            if args.video_step_interval > 0:
                video_step_interval = args.video_step_interval
            else:
                video_step_interval = max(
                    1, math.ceil(len(data_loader) / max(1, len(video_data_loader))))
            print(f"  Video step interval: {video_step_interval} image steps/video batch")
            print(f"  Video steps/epoch used: {math.ceil(len(data_loader) / video_step_interval)}")
            print(f"------------------\n")

    # Save the effective runtime config after any auto-adjustments
    # (for example, video training auto-enables temporal mode).
    if is_main_process():
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Print data config summary
    if is_main_process():
        print(f"\n--- Data Config ---")
        print(f"  Manifest: {args.data_path}")
        print(f"  Mode: {args.data_mode}")
        print(f"  Samples: {len(dataset)}")
        if hasattr(dataset, 'get_type_distribution'):
            try:
                print(f"  Type distribution: {dataset.get_type_distribution()}")
            except RuntimeError:
                pass
        if type_sampling_probs is not None:
            print(f"  Type sampling probs: {type_sampling_probs}")
        print(f"  Image workers: {image_num_workers}, "
              f"persistent: {args.persistent_workers}, "
              f"prefetch: {max(1, int(args.prefetch_factor)) if image_num_workers > 0 else 'n/a'}")
        print(f"  Strict loading: {args.strict_loading}")
        print(f"  Max fail ratio: {args.max_load_fail_ratio}")
        if args.dry_run_samples > 0:
            print(f"  DRY-RUN: limited to {args.dry_run_samples} samples")
        print(f"-------------------\n")

    # Dry-run mode: load a few batches to validate then exit
    if args.dry_run_samples > 0:
        if is_main_process():
            print("=== DRY-RUN: validating data loading ===")
        n_batches = min(len(data_loader), 10)
        for i, (crops, view_crop, crop_masks, view_mask,
                pair_anchor, base_masks) in enumerate(data_loader):
            if i >= n_batches:
                break
            if is_main_process() and i == 0:
                print(f"  Batch shape: crops[0]={crops[0].shape}, "
                      f"crop_masks[0]={crop_masks[0].shape}, "
                      f"view_mask={'None' if view_mask is None else tuple(view_mask.shape)}, "
                      f"pair_anchor={'None' if pair_anchor is None else tuple(pair_anchor.shape)}, "
                      f"base_masks={tuple(base_masks.shape)}")
        if is_main_process():
            print(f"  Loaded {n_batches} batches OK.")
            print(f"  {dataset.load_stats.get_summary()}")
        if video_data_loader is not None:
            n_video_batches = min(len(video_data_loader), 10)
            for i, (frames, video_masks, timestamps) in enumerate(video_data_loader):
                if i >= n_video_batches:
                    break
                if is_main_process() and i == 0:
                    print(f"  Video batch shape: frames={tuple(frames.shape)}, "
                          f"video_masks={tuple(video_masks.shape)}, "
                          f"timestamps={tuple(timestamps.shape)}")
            if is_main_process():
                print(f"  Loaded {n_video_batches} video batches OK.")
        if is_main_process():
            print("=== DRY-RUN complete. Exiting. ===")
        sys.exit(0)

    # =========================================================================
    # Build model
    # =========================================================================
    (student, teacher, mg_student, mg_teacher,
     view_bridge, temporal_attn, clip_projector, embed_dim) = build_model(args)

    # Move to GPU
    student = student.cuda()
    teacher = teacher.cuda()
    mg_student = mg_student.cuda()
    mg_teacher = mg_teacher.cuda()
    view_bridge = view_bridge.cuda()
    if temporal_attn is not None:
        temporal_attn = temporal_attn.cuda()
        if dist.is_initialized():
            temporal_attn = DDP(temporal_attn, device_ids=[local_rank],
                                find_unused_parameters=True)
    if clip_projector is not None:
        clip_projector = clip_projector.cuda()
        if dist.is_initialized():
            clip_projector = DDP(clip_projector, device_ids=[local_rank],
                                 find_unused_parameters=True)

    # DDP wrapping
    if dist.is_initialized():
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = DDP(student, device_ids=[local_rank],
                      find_unused_parameters=True)
        mg_student_ddp = DDP(mg_student, device_ids=[local_rank],
                             find_unused_parameters=True)
        view_bridge_ddp = DDP(view_bridge, device_ids=[local_rank],
                              find_unused_parameters=True)
    else:
        mg_student_ddp = mg_student
        view_bridge_ddp = view_bridge

    # =========================================================================
    # Build loss and optimizer
    # =========================================================================
    loss_fn = build_loss(args, embed_dim).cuda()

    # Separate uncertainty params from regular loss_fn params
    loss_fn_regular_params = []
    uncertainty_params = []
    for name, p in loss_fn.named_parameters():
        if not p.requires_grad:
            continue
        if 'uncertainty' in name:
            uncertainty_params.append(p)
        else:
            loss_fn_regular_params.append(p)

    params_groups = [
        {"params": [p for p in student.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for p in mg_student.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for p in view_bridge.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": loss_fn_regular_params,
         "weight_decay": args.weight_decay},
    ]
    if uncertainty_params:
        params_groups.append({
            "params": uncertainty_params,
            "weight_decay": 0.0,
            "lr_scale": 0.01,
        })
    if temporal_attn is not None:
        params_groups.append(
            {"params": [p for p in temporal_attn.parameters() if p.requires_grad],
             "weight_decay": args.weight_decay})
    if clip_projector is not None:
        params_groups.append(
            {"params": [p for p in clip_projector.parameters() if p.requires_grad],
             "weight_decay": args.weight_decay})

    optimizer = torch.optim.AdamW(params_groups)

    # FP16 scaler
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # =========================================================================
    # Schedules
    # =========================================================================
    effective_batch_size = (args.batch_size_per_gpu
                            * get_world_size()
                            * getattr(args, 'accumulate_grad_batches', 1))
    base_lr = args.lr * effective_batch_size / 256

    lr_schedule = cosine_scheduler(
        base_value=base_lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
    )

    momentum_schedule = cosine_scheduler(
        base_value=args.momentum_teacher,
        final_value=1.0,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
    )
    aux_loss_schedule = ramp_scheduler(
        base_value=1.0,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        ramp_epochs=getattr(args, 'aux_loss_ramp_epochs', 0),
        start_value=getattr(args, 'aux_loss_start_scale', 1.0),
    )
    video_aux_loss_schedule = ramp_scheduler(
        base_value=1.0,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        ramp_epochs=getattr(args, 'video_aux_ramp_epochs', 0),
        start_value=getattr(args, 'video_aux_start_scale', 1.0),
    )

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        student_state = checkpoint.get('student', {})
        if dist.is_initialized():
            student.module.load_state_dict(student_state, strict=False)
        else:
            student.load_state_dict(student_state, strict=False)
        teacher.load_state_dict(checkpoint.get('teacher', {}), strict=False)
        mg_student.load_state_dict(checkpoint.get('mg_student', {}), strict=False)
        mg_teacher.load_state_dict(checkpoint.get('mg_teacher', {}), strict=False)
        view_bridge.load_state_dict(checkpoint.get('view_bridge', {}), strict=False)
        if 'loss_fn' in checkpoint:
            loss_fn.load_state_dict(checkpoint['loss_fn'], strict=False)
        if temporal_attn is not None and 'temporal_attn' in checkpoint:
            ta_module = (temporal_attn.module
                         if isinstance(temporal_attn, DDP) else temporal_attn)
            ta_module.load_state_dict(checkpoint['temporal_attn'], strict=False)
        if clip_projector is not None and 'clip_projector' in checkpoint:
            cp_module = (clip_projector.module
                         if isinstance(clip_projector, DDP) else clip_projector)
            cp_module.load_state_dict(checkpoint['clip_projector'], strict=False)
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                print(f"  WARNING: Could not load optimizer state ({e}), "
                      "re-initializing optimizer (LR schedule will restart)")
                # epoch / lr_schedule still restored correctly
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

        # Validate critical architecture args match
        saved_args = checkpoint.get('args', {})
        if isinstance(saved_args, dict):
            for key in ['fusion_start_block', 'fusion', 'arch', 'patch_size', 'in_chans']:
                saved_val = saved_args.get(key)
                current_val = getattr(args, key, None)
                if saved_val is not None and current_val is not None and saved_val != current_val:
                    print(f"  WARNING: {key} mismatch: checkpoint={saved_val}, current={current_val}")

    # =========================================================================
    # Print config
    # =========================================================================
    if is_main_process():
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        mg_params = sum(p.numel() for p in mg_student.parameters())
        vb_params = sum(p.numel() for p in view_bridge.parameters())
        ta_params = sum(p.numel() for p in temporal_attn.parameters()) if temporal_attn is not None else 0

        print(f"\n{'='*60}")
        print(f"DINO-MM RGB+IR Pretraining")
        print(f"{'='*60}")
        print(f"  Arch: {args.arch}, Patch size: {args.patch_size}")
        print(f"  Input channels: {args.in_chans}, Fusion: {args.fusion}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Student params: {total_params/1e6:.1f}M (trainable: {trainable_params/1e6:.1f}M)")
        print(f"  MG params: {mg_params/1e6:.1f}M")
        print(f"  VB params: {vb_params/1e6:.1f}M")
        if temporal_attn is not None:
            print(f"  TA params: {ta_params/1e6:.1f}M")
        print(f"  Batch size: {effective_batch_size} "
              f"({args.batch_size_per_gpu} x {get_world_size()} GPUs"
              + (f" x {args.accumulate_grad_batches} accum" if args.accumulate_grad_batches > 1 else "")
              + ")")
        print(f"  Base LR: {base_lr:.6f}")
        print(f"  Epochs: {args.epochs} (start: {start_epoch})")
        print(f"  Gradient checkpointing: {'ON' if args.use_gradient_checkpointing else 'OFF'}")
        print(f"  PCGrad: {'ON' if args.use_pcgrad else 'OFF'}")
        print(f"  Drop path rate: {args.drop_path_rate}")
        print(f"  Loss weights: mgcl={args.w_mgcl}, "
              f"align_rgbir={getattr(args, 'w_align_rgbir', 0.0)}, "
              f"align_rgbir_patch={getattr(args, 'w_align_rgbir_patch', 0.0)}, "
              f"view={args.w_view}, bridge={args.w_bridge}"
              + (f", dino_video={getattr(args, 'w_dino_video', 0.2)}, "
                 f"{args.temporal_loss_type}={args.w_tcl}, "
                 f"{args.temporal_loss_type}_patch={getattr(args, 'w_tcl_patch', 0.5)}"
                 if args.use_temporal else ""))
        if getattr(args, 'adaptive_weighting', False):
            print(f"  Adaptive weighting: ON (Kendall uncertainty, lr_scale=0.01)")
        if getattr(args, 'aux_loss_ramp_epochs', 0) > 0:
            print(f"  Aux ramp: image {args.aux_loss_start_scale:.2f} -> 1.00 over {args.aux_loss_ramp_epochs} epochs")
        if getattr(args, 'video_aux_ramp_epochs', 0) > 0:
            print(f"  Video aux ramp: {args.video_aux_start_scale:.2f} -> 1.00 over {args.video_aux_ramp_epochs} epochs")
        print(f"  Temporal: {'ON' if args.use_temporal else 'OFF'}")
        if args.use_temporal:
            print(f"  Temporal loss: {args.temporal_loss_type.upper()}")
        if args.video_manifest:
            print(f"  Video: ON (manifest={args.video_manifest})")
            print(f"  Video batch: {args.video_batch_size}, frames: {args.num_frames}")
            print(f"  Video step interval: {args.video_step_interval if args.video_step_interval > 0 else 'auto'}")
        print(f"  View aug: {'ON' if args.use_view_aug else 'OFF'}")
        print(f"{'='*60}\n")

    # --- Wandb initialization ---
    use_wandb = _WANDB_AVAILABLE and args.wandb_key and is_main_process()
    if use_wandb:
        wandb.login(key=args.wandb_key)
        run_name = args.wandb_run_name or (
            f"{args.arch}_p{args.patch_size}_{args.fusion}_"
            f"bs{effective_batch_size}_ep{args.epochs}")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            resume='allow',
        )
        wandb.config.update({
            'effective_batch_size': effective_batch_size,
            'base_lr': base_lr,
            'total_samples': len(dataset),
            'iters_per_epoch': len(data_loader),
        }, allow_val_change=True)
        print(f"  Wandb: ON (project={args.wandb_project}, run={run_name})")

    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(start_epoch, args.epochs):
        if dist.is_initialized():
            data_loader.sampler.set_epoch(epoch)
            if video_sampler is not None and hasattr(video_sampler, 'set_epoch'):
                video_sampler.set_epoch(epoch)

        epoch_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            mg_student=mg_student_ddp if dist.is_initialized() else mg_student,
            mg_teacher=mg_teacher,
            view_bridge=view_bridge_ddp if dist.is_initialized() else view_bridge,
            temporal_attn=temporal_attn,
            clip_projector=clip_projector,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            aux_loss_schedule=aux_loss_schedule,
            video_aux_loss_schedule=video_aux_loss_schedule,
            epoch=epoch,
            fp16_scaler=fp16_scaler,
            args=args,
            video_data_loader=video_data_loader,
        )

        # Save checkpoint
        student_state = (student.module.state_dict()
                         if dist.is_initialized() else student.state_dict())

        # Always save checkpoint_latest for resumability
        save_checkpoint({
            'epoch': epoch,
            'student': student_state,
            'teacher': teacher.state_dict(),
            'mg_student': mg_student.state_dict(),
            'mg_teacher': mg_teacher.state_dict(),
            'view_bridge': view_bridge.state_dict(),
            'loss_fn': loss_fn.state_dict(),
            'temporal_attn': (temporal_attn.module.state_dict()
                              if isinstance(temporal_attn, DDP)
                              else temporal_attn.state_dict()) if temporal_attn is not None else {},
            'clip_projector': (clip_projector.module.state_dict()
                               if isinstance(clip_projector, DDP)
                               else clip_projector.state_dict()) if clip_projector is not None else {},
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'epoch_stats': epoch_stats,
        }, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        # Save numbered checkpoint periodically
        # Hot-reload: check for override file (echo 3 > output_dir/.save_every)
        _save_every_file = os.path.join(args.output_dir, '.save_every')
        _save_every = args.save_every
        if os.path.isfile(_save_every_file):
            try:
                _save_every = int(open(_save_every_file).read().strip())
            except (ValueError, OSError):
                pass
        if epoch % _save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch,
                'student': student_state,
                'teacher': teacher.state_dict(),
                'mg_student': mg_student.state_dict(),
                'mg_teacher': mg_teacher.state_dict(),
                'view_bridge': view_bridge.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'temporal_attn': (temporal_attn.module.state_dict()
                              if isinstance(temporal_attn, DDP)
                              else temporal_attn.state_dict()) if temporal_attn is not None else {},
                'clip_projector': (clip_projector.module.state_dict()
                              if isinstance(clip_projector, DDP)
                              else clip_projector.state_dict()) if clip_projector is not None else {},
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'epoch_stats': epoch_stats,
            }, os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth'))

        if is_main_process():
            print(f"\nEpoch {epoch} completed. Stats: {epoch_stats}")
            print(f"  {dataset.load_stats.get_summary()}\n")

            # Wandb epoch summary
            if use_wandb:
                epoch_log = {f'epoch/{k}': v for k, v in epoch_stats.items()}
                epoch_log['epoch/epoch'] = epoch
                wandb.log(epoch_log)

    print("Training completed!")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

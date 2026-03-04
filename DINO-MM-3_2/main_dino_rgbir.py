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
import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.vision_transformer_rgbir import (
    vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant
)
from models.dino_head import DINOHead, MultiCropWrapper
from models.multi_granularity import MultiGranularityFeatures
from models.modality_completion import ModalityCompletion
from models.temporal_module import TemporalAttention
from models.transforms_rgbir import DataAugmentationDINO_RGBIR
from models.view_bridge import ViewDomainBridge
from datasets.multimodal_drone import (
    MultiModalDroneDataset, collate_multimodal)
from dino_loss_rgbir import (
    DINOLoss, MGCLLoss, CrossModalAlignLoss, TCLLoss,
    ViewInvarianceLoss, ViewBridgeLoss, PretrainingLoss)


# ============================================================================
# Utilities
# ============================================================================

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine annealing schedule with optional linear warmup."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
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
    parser.add_argument('--fusion', default='concat', type=str,
                        choices=['concat', 'add', 'cross_attn'],
                        help=(
                            'RGB-IR patch embedding fusion mode. '
                            '"concat": cat+Linear+LN (recommended, ~d^2 extra params); '
                            '"add": element-wise sum (no extra params); '
                            '"cross_attn": RGB cross-attends to IR context.'))

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
    parser.add_argument('--w_align', default=1.0, type=float)
    parser.add_argument('--w_view', default=0.5, type=float)
    parser.add_argument('--w_bridge', default=0.5, type=float)
    parser.add_argument('--w_latent', default=1.0, type=float)
    parser.add_argument('--w_rec', default=1.0, type=float)

    # View-domain bridge
    parser.add_argument('--bridge_proj_dim', default=256, type=int)
    parser.add_argument('--bridge_num_prototypes', default=64, type=int)
    parser.add_argument('--bridge_temp', default=1.0, type=float,
                        help='ViewBridge softmax temperature (higher=softer, must be >0.3 to avoid near-one-hot)')
    parser.add_argument('--bridge_lambda_sharp', default=0.0, type=float,
                        help='ViewBridge sharpness weight (0=disabled, avoids conflict with high temp)')
    parser.add_argument('--bridge_lambda_balance', default=0.02, type=float,
                        help='ViewBridge balance weight (encourages uniform prototype usage)')

    # Modality completion
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--use_codebook', default=True, action='store_true')

    # Temporal (not used in phase 1)
    parser.add_argument('--use_temporal', default=False, action='store_true')
    parser.add_argument('--num_frames', default=1, type=int)
    parser.add_argument('--temporal_layers', default=2, type=int)
    parser.add_argument('--w_tcl', default=0.1, type=float,
                        help='Weight for TCL (time-contrastive learning) loss')
    parser.add_argument('--tcl_temperature', default=0.07, type=float,
                        help='Temperature for TCL InfoNCE loss')

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
    parser.add_argument('--num_workers', default=8, type=int)

    # Resume
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')

    return parser


# ============================================================================
# Model building
# ============================================================================

def init_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        print("Not using distributed mode, single GPU training")
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
        drop_path_rate=0.1,
        fusion=args.fusion,
    )
    embed_dim = student_backbone.embed_dim

    # Build teacher backbone
    teacher_backbone = backbone_fn(
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        drop_path_rate=0.0,
        fusion=args.fusion,
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

    # Modality completion
    mod_completion = ModalityCompletion(
        feature_dim=embed_dim, latent_dim=args.latent_dim,
        num_modalities=2, use_codebook=args.use_codebook)

    # View-domain bridge
    view_bridge = ViewDomainBridge(
        in_dim=embed_dim,
        proj_dim=args.bridge_proj_dim,
        num_prototypes=args.bridge_num_prototypes,
    )

    # Temporal attention (phase 2)
    temporal_attn = None
    if args.use_temporal:
        temporal_attn = TemporalAttention(
            embed_dim=embed_dim, num_heads=8,
            num_layers=args.temporal_layers)

    return (student, teacher, mg_student, mg_teacher,
            mod_completion, view_bridge, temporal_attn, embed_dim)


def build_loss(args):
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

    align_loss = CrossModalAlignLoss(temperature=0.07)
    view_loss = ViewInvarianceLoss(temperature=0.07)
    bridge_loss = ViewBridgeLoss(
        temperature=args.bridge_temp,
        lambda_sharp=args.bridge_lambda_sharp,
        lambda_balance=args.bridge_lambda_balance,
    )

    tcl_loss = TCLLoss(temperature=args.tcl_temperature) if args.use_temporal else None

    combined = PretrainingLoss(
        dino_loss=dino_loss,
        mgcl_loss=mgcl_loss,
        align_loss=align_loss,
        view_loss=view_loss,
        view_bridge_loss=bridge_loss,
        tcl_loss=tcl_loss,
        w_mgcl=args.w_mgcl,
        w_align=args.w_align,
        w_view=args.w_view,
        w_bridge=args.w_bridge,
        w_latent=args.w_latent,
        w_rec=args.w_rec,
        w_tcl=args.w_tcl,
    )

    return combined


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(student, teacher, mg_student, mg_teacher,
                    mod_completion, view_bridge, temporal_attn,
                    loss_fn, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args):
    """Train for one epoch."""
    student.train()
    mg_student.train()
    mod_completion.train()
    view_bridge.train()

    metric_logger = {k: SmoothedValue(window_size=20) for k in [
        'total', 'dino', 'mgcl', 'align', 'view', 'bridge', 'latent', 'rec', 'tcl', 'lr']}

    for it, (crops, view_crop, modality_mask) in enumerate(data_loader):
        global_it = len(data_loader) * epoch + it

        # Safely get backbone
        if isinstance(student, DDP):
            student_backbone = student.module.backbone
        else:
            student_backbone = student.backbone

        if isinstance(teacher, DDP):
            teacher_model = teacher.module
        else:
            teacher_model = teacher

        # Update learning rate and weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[global_it]

        # Move to GPU
        crops = [c.cuda(non_blocking=True) for c in crops]
        modality_mask = modality_mask.cuda(non_blocking=True)
        if view_crop is not None:
            view_crop = view_crop.cuda(non_blocking=True)

        # --- Forward pass with mixed precision ---
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            # Teacher forward: only 2 global crops
            teacher_out = teacher_model(crops[:2])  # [B*2, out_dim]

            # Student forward: all crops
            student_out, student_tokens = student(
                crops, return_backbone_feat=True)

            # --- Multi-granularity features (on global crops only) ---
            B = crops[0].shape[0]
            global_student_tokens = student_tokens[:B*2]  # [B*2, N, d]

            with torch.no_grad():
                teacher_tokens_list = []
                for gc in crops[:2]:
                    t_out = teacher_model.backbone(gc, return_all_tokens=True)
                    teacher_tokens_list.append(t_out[:, 1:])
                global_teacher_tokens = torch.cat(teacher_tokens_list, dim=0)

            # Multi-granularity features
            student_mg = mg_student(global_student_tokens)
            with torch.no_grad():
                teacher_mg = mg_teacher(global_teacher_tokens)

            # --- Modality-specific features for alignment/completion ---
            gc1 = crops[0]

            rgb_only = gc1.clone()
            rgb_only[:, 3:, :, :] = 0
            ir_only = gc1.clone()
            ir_only[:, :3, :, :] = 0

            rgb_tokens = student_backbone(rgb_only, return_all_tokens=True)[:, 1:]
            ir_tokens = student_backbone(ir_only, return_all_tokens=True)[:, 1:]

            # Stop-gradient on IR tokens for cross-modal alignment.
            # Without this, the shared backbone can trivially satisfy the
            # alignment loss by mapping both modalities to the same point,
            # making align_loss/latent_loss/rec_loss collapse to ~0.
            # Detaching IR makes alignment only update the RGB pathway,
            # forcing genuine cross-modal feature learning.
            ir_tokens_sg = ir_tokens.detach()

            # Combine dataset mask with per-crop channel check after SensorDrop
            rgb_present = (gc1[:, :3].abs().sum(dim=(1, 2, 3)) > 0).float()
            ir_present = (gc1[:, 3:].abs().sum(dim=(1, 2, 3)) > 0).float()
            aug_modality_mask = torch.stack([
                modality_mask[:, 0] * rgb_present,
                modality_mask[:, 1] * ir_present,
            ], dim=1)

            feat_rgb = rgb_tokens
            feat_ir = ir_tokens_sg  # stop-gradient for alignment loss

            # --- Modality completion ---
            # Use ir_tokens (with gradient) for completion so decoder can backprop
            completion_feats = [rgb_tokens, ir_tokens]
            _, completion_losses = mod_completion(completion_feats, aug_modality_mask)

            # --- Viewpoint invariance + bridge features ---
            feat_original = global_student_tokens[:B].mean(dim=1)  # [B, d]
            feat_global2 = global_student_tokens[B:B * 2].mean(dim=1)  # [B, d]
            feat_view = None
            bridge_logits_list = []

            _, bridge_logits_g1 = view_bridge(feat_original)
            _, bridge_logits_g2 = view_bridge(feat_global2)
            bridge_logits_list.extend([bridge_logits_g1, bridge_logits_g2])

            if view_crop is not None:
                view_out = student_backbone(view_crop, return_all_tokens=True)
                feat_view = view_out[:, 1:].mean(dim=1)  # [B, d]
                _, bridge_logits_view = view_bridge(feat_view)
                bridge_logits_list.append(bridge_logits_view)

            # --- TCL: temporal contrastive learning ---
            tcl_features = None
            tcl_num_frames = 1

            if args.use_temporal and temporal_attn is not None:
                # video_frames provided by external data loader (currently None)
                video_frames = None  # placeholder until video data is ready
                if video_frames is not None:
                    B_v, T, C, H, W = video_frames.shape
                    flat = video_frames.reshape(B_v * T, C, H, W)
                    frame_tok = student_backbone(flat, return_all_tokens=True)[:, 1:]
                    frame_tok = temporal_attn(frame_tok, num_frames=T)
                    tcl_features = frame_tok.mean(dim=1)  # [B_v*T, d]
                    tcl_num_frames = T

            # --- Combined loss ---
            total_loss, loss_dict = loss_fn(
                student_out=student_out,
                teacher_out=teacher_out,
                epoch=epoch,
                student_mg=student_mg,
                teacher_mg=teacher_mg,
                feat_rgb=feat_rgb,
                feat_ir=feat_ir,
                modality_mask=aug_modality_mask,
                feat_original=feat_original,
                feat_view=feat_view,
                bridge_logits_list=bridge_logits_list,
                completion_losses=completion_losses,
                tcl_features=tcl_features,
                num_frames=tcl_num_frames,
            )

        # --- Backward pass ---
        optimizer.zero_grad()
        if fp16_scaler is not None:
            fp16_scaler.scale(total_loss).backward()
            if args.clip_grad > 0:
                fp16_scaler.unscale_(optimizer)
                param_list = (
                    list(student.parameters()) +
                    list(mg_student.parameters()) +
                    list(mod_completion.parameters()) +
                    list(view_bridge.parameters()))
                if temporal_attn is not None:
                    param_list += list(temporal_attn.parameters())
                nn.utils.clip_grad_norm_(param_list, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            total_loss.backward()
            if args.clip_grad > 0:
                param_list = (
                    list(student.parameters()) +
                    list(mg_student.parameters()) +
                    list(mod_completion.parameters()) +
                    list(view_bridge.parameters()))
                if temporal_attn is not None:
                    param_list += list(temporal_attn.parameters())
                nn.utils.clip_grad_norm_(param_list, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()

        # --- EMA update teacher ---
        with torch.no_grad():
            m = momentum_schedule[global_it]
            for ps, pt in zip(student.parameters(),
                              teacher.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.detach().data)
            for ps, pt in zip(mg_student.parameters(),
                              mg_teacher.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.detach().data)

        # --- Logging ---
        for k, v in loss_dict.items():
            if k in metric_logger:
                metric_logger[k].update(v)
        metric_logger['lr'].update(optimizer.param_groups[0]['lr'])

        if it % args.log_every == 0 and is_main_process():
            log_str = (
                f"Epoch [{epoch}][{it}/{len(data_loader)}]  "
                f"loss: {metric_logger['total'].avg:.4f}  "
                f"dino: {metric_logger['dino'].avg:.4f}  "
                f"mgcl: {metric_logger['mgcl'].avg:.4f}  "
                f"align: {metric_logger['align'].avg:.4f}  "
                f"view: {metric_logger['view'].avg:.4f}  "
                f"bridge: {metric_logger['bridge'].avg:.4f}  "
                f"lr: {metric_logger['lr'].avg:.6f}"
            )
            if args.use_temporal:
                log_str += f"  tcl: {metric_logger['tcl'].avg:.4f}"
            print(log_str)

    return {k: v.global_avg for k, v in metric_logger.items()}


# ============================================================================
# Main
# ============================================================================

def main(args):
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Fix random seeds
    torch.manual_seed(42 + rank)
    cudnn.benchmark = True

    # Output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    if is_main_process():
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

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
    )

    dataset = MultiModalDroneDataset(
        data_source=args.data_path,
        transform=transform,
        mode=args.data_mode,
        strict_loading=args.strict_loading,
        max_load_fail_ratio=args.max_load_fail_ratio,
        dry_run_samples=args.dry_run_samples,
    )

    if dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_multimodal,
    )

    print(f"Dataset: {len(dataset)} samples, "
          f"DataLoader: {len(data_loader)} batches/epoch")

    # Print data config summary
    if is_main_process():
        print(f"\n--- Data Config ---")
        print(f"  Manifest: {args.data_path}")
        print(f"  Mode: {args.data_mode}")
        print(f"  Samples: {len(dataset)}")
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
        for i, (crops, view_crop, modality_mask) in enumerate(data_loader):
            if i >= n_batches:
                break
            if is_main_process() and i == 0:
                print(f"  Batch shape: crops[0]={crops[0].shape}, "
                      f"modality_mask={modality_mask.shape}")
        if is_main_process():
            print(f"  Loaded {n_batches} batches OK.")
            print(f"  {dataset.load_stats.get_summary()}")
            print("=== DRY-RUN complete. Exiting. ===")
        sys.exit(0)

    # =========================================================================
    # Build model
    # =========================================================================
    (student, teacher, mg_student, mg_teacher,
     mod_completion, view_bridge, temporal_attn, embed_dim) = build_model(args)

    # Move to GPU
    student = student.cuda()
    teacher = teacher.cuda()
    mg_student = mg_student.cuda()
    mg_teacher = mg_teacher.cuda()
    mod_completion = mod_completion.cuda()
    view_bridge = view_bridge.cuda()
    if temporal_attn is not None:
        temporal_attn = temporal_attn.cuda()

    # DDP wrapping
    if dist.is_initialized():
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = DDP(student, device_ids=[local_rank])
        mg_student_ddp = DDP(mg_student, device_ids=[local_rank])
        mod_completion_ddp = DDP(mod_completion, device_ids=[local_rank])
        view_bridge_ddp = DDP(view_bridge, device_ids=[local_rank])
    else:
        mg_student_ddp = mg_student
        mod_completion_ddp = mod_completion
        view_bridge_ddp = view_bridge

    # =========================================================================
    # Build loss and optimizer
    # =========================================================================
    loss_fn = build_loss(args).cuda()

    params_groups = [
        {"params": [p for p in student.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for p in mg_student.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for p in mod_completion.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for p in view_bridge.parameters() if p.requires_grad],
         "weight_decay": args.weight_decay},
    ]
    if temporal_attn is not None:
        params_groups.append(
            {"params": [p for p in temporal_attn.parameters() if p.requires_grad],
             "weight_decay": args.weight_decay})

    optimizer = torch.optim.AdamW(params_groups)

    # FP16 scaler
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # =========================================================================
    # Schedules
    # =========================================================================
    effective_batch_size = args.batch_size_per_gpu * get_world_size()
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

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        student_state = checkpoint.get('student', {})
        if dist.is_initialized():
            student.module.load_state_dict(student_state, strict=False)
        else:
            student.load_state_dict(student_state, strict=False)
        teacher.load_state_dict(checkpoint.get('teacher', {}), strict=False)
        mg_student.load_state_dict(checkpoint.get('mg_student', {}), strict=False)
        mg_teacher.load_state_dict(checkpoint.get('mg_teacher', {}), strict=False)
        mod_completion.load_state_dict(checkpoint.get('mod_completion', {}), strict=False)
        view_bridge.load_state_dict(checkpoint.get('view_bridge', {}), strict=False)
        if temporal_attn is not None and 'temporal_attn' in checkpoint:
            temporal_attn.load_state_dict(checkpoint['temporal_attn'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # =========================================================================
    # Print config
    # =========================================================================
    if is_main_process():
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        mg_params = sum(p.numel() for p in mg_student.parameters())
        mc_params = sum(p.numel() for p in mod_completion.parameters())
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
        print(f"  MC params: {mc_params/1e6:.1f}M")
        print(f"  VB params: {vb_params/1e6:.1f}M")
        if temporal_attn is not None:
            print(f"  TA params: {ta_params/1e6:.1f}M")
        print(f"  Batch size: {effective_batch_size} "
              f"({args.batch_size_per_gpu} x {get_world_size()} GPUs)")
        print(f"  Base LR: {base_lr:.6f}")
        print(f"  Epochs: {args.epochs} (start: {start_epoch})")
        print(f"  Loss weights: mgcl={args.w_mgcl}, align={args.w_align}, "
              f"view={args.w_view}, bridge={args.w_bridge}, "
              f"latent={args.w_latent}, rec={args.w_rec}"
              + (f", tcl={args.w_tcl}" if args.use_temporal else ""))
        print(f"  Temporal: {'ON' if args.use_temporal else 'OFF'}")
        print(f"  View aug: {'ON' if args.use_view_aug else 'OFF'}")
        print(f"{'='*60}\n")

    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(start_epoch, args.epochs):
        if dist.is_initialized():
            data_loader.sampler.set_epoch(epoch)

        epoch_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            mg_student=mg_student_ddp if dist.is_initialized() else mg_student,
            mg_teacher=mg_teacher,
            mod_completion=mod_completion_ddp if dist.is_initialized() else mod_completion,
            view_bridge=view_bridge_ddp if dist.is_initialized() else view_bridge,
            temporal_attn=temporal_attn,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            epoch=epoch,
            fp16_scaler=fp16_scaler,
            args=args,
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
            'mod_completion': mod_completion.state_dict(),
            'view_bridge': view_bridge.state_dict(),
            'temporal_attn': temporal_attn.state_dict() if temporal_attn is not None else {},
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'epoch_stats': epoch_stats,
        }, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        # Save numbered checkpoint periodically
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch,
                'student': student_state,
                'teacher': teacher.state_dict(),
                'mg_student': mg_student.state_dict(),
                'mg_teacher': mg_teacher.state_dict(),
                'mod_completion': mod_completion.state_dict(),
                'view_bridge': view_bridge.state_dict(),
                'temporal_attn': temporal_attn.state_dict() if temporal_attn is not None else {},
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'epoch_stats': epoch_stats,
            }, os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth'))

        if is_main_process():
            print(f"\nEpoch {epoch} completed. Stats: {epoch_stats}")
            print(f"  {dataset.load_stats.get_summary()}\n")

    print("Training completed!")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

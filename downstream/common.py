import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal_module import TemporalAttention
from models.vision_transformer_rgbir import (
    vit_base,
    vit_giant,
    vit_huge,
    vit_large,
    vit_small,
    vit_tiny,
)


BACKBONE_FACTORY = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_huge": vit_huge,
    "vit_giant": vit_giant,
}

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
IR_MEAN = [0.5]
IR_STD = [0.5]
RGBIR_MEAN = RGB_MEAN + IR_MEAN
RGBIR_STD = RGB_STD + IR_STD


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_task_checkpoint(path, state):
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _namespace_to_dict(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    return {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}


def _select_state_dict(checkpoint, checkpoint_key="teacher"):
    if checkpoint_key in checkpoint:
        return checkpoint[checkpoint_key]
    for key in ("teacher", "student", "model", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        name = key
        if name.startswith("module.backbone."):
            name = name[len("module.backbone."):]
        elif name.startswith("backbone."):
            name = name[len("backbone."):]
        elif name.startswith("module."):
            name = name[len("module."):]

        if name.startswith("head.") or name.startswith("head0.") or name.startswith("head1."):
            continue
        cleaned[name] = value
    return cleaned


def _remap_patch_embed(model, state_dict):
    is_dual_modal = hasattr(model.patch_embed, "rgb_proj") and hasattr(model.patch_embed, "ir_proj")
    if not is_dual_modal:
        return state_dict

    for suffix in ("weight", "bias"):
        old_key = f"patch_embed.proj.{suffix}"
        new_key = f"patch_embed.rgb_proj.{suffix}"
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def _interpolate_pos_embed(model, state_dict):
    if "pos_embed" not in state_dict:
        return state_dict

    model_pos = model.state_dict().get("pos_embed")
    pretrained_pos = state_dict["pos_embed"]
    if model_pos is None or pretrained_pos.shape == model_pos.shape:
        return state_dict

    cls_pos = pretrained_pos[:, :1]
    patch_pos = pretrained_pos[:, 1:]
    dim = patch_pos.shape[-1]
    old_size = int(math.sqrt(patch_pos.shape[1]))
    new_size = int(math.sqrt(model_pos.shape[1] - 1))

    patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    patch_pos = nn.functional.interpolate(
        patch_pos,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    state_dict["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
    return state_dict


def build_backbone_from_checkpoint(
    checkpoint_path,
    checkpoint_key="teacher",
    device="cpu",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    load_temporal=True,
    init_mode="pretrained",
):
    checkpoint = {}
    args = {}
    if init_mode != "random":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required unless init_mode='random'.")
        checkpoint = load_checkpoint(checkpoint_path)
        args = _namespace_to_dict(checkpoint.get("args"))

    arch = arch or args.get("arch", "vit_small")
    patch_size = patch_size or int(args.get("patch_size", 16))
    in_chans = in_chans or int(args.get("in_chans", 4))
    fusion = fusion or args.get("fusion", "gated_cross_attn")

    if arch not in BACKBONE_FACTORY:
        raise ValueError(f"Unsupported architecture '{arch}'.")

    model = BACKBONE_FACTORY[arch](
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        num_classes=0,
    )

    if init_mode != "random":
        state_dict = _select_state_dict(checkpoint, checkpoint_key=checkpoint_key)
        state_dict = _clean_state_dict(state_dict)
        state_dict = _remap_patch_embed(model, state_dict)
        state_dict = _interpolate_pos_embed(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    temporal_attn = None
    temporal_state = checkpoint.get("temporal_attn", {})
    if init_mode != "random" and load_temporal and temporal_state:
        temporal_layers = int(args.get("temporal_layers", 2))
        temporal_heads = max(1, model.embed_dim // 64)
        temporal_attn = TemporalAttention(
            embed_dim=model.embed_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
        )
        temporal_attn.load_state_dict(temporal_state, strict=False)
        temporal_attn.to(device)

    meta = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_key": checkpoint_key,
        "init_mode": init_mode,
        "arch": arch,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "fusion": fusion,
        "use_temporal": bool(temporal_state) and load_temporal,
        "embed_dim": model.embed_dim,
    }
    return model, temporal_attn, meta


def freeze_backbone(backbone, trainable_blocks=4, train_patch_embed=False):
    total_blocks = len(backbone.blocks)
    trainable_blocks = max(0, min(total_blocks, trainable_blocks))
    first_trainable = total_blocks - trainable_blocks

    if hasattr(backbone, "patch_embed"):
        for param in backbone.patch_embed.parameters():
            param.requires_grad = bool(train_patch_embed)

    if hasattr(backbone, "cls_token"):
        backbone.cls_token.requires_grad = True
    if hasattr(backbone, "pos_embed"):
        backbone.pos_embed.requires_grad = True

    for idx, block in enumerate(backbone.blocks):
        requires_grad = idx >= first_trainable
        for param in block.parameters():
            param.requires_grad = requires_grad

    for param in backbone.norm.parameters():
        param.requires_grad = True


def tensor_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append({key: tensor_to_device(value, device) for key, value in target.items()})
    return moved

#!/usr/bin/env python3
"""Train DINO-MM + Cas-MVSNet for 3D reconstruction.

Reference: RingMo-Aerial Tab.11 (LuoJia-MVS / WHU)
  - Metrics: MAE, <0.6m accuracy, <3-interval accuracy
  - Cas-MVSNet baseline: MAE=0.122 (5-view LuoJia), MAE=0.091 (5-view WHU)

Usage:
    python train_3d_reconstruction.py \
        --data_root /path/to/LuoJia-MVS \
        --checkpoint /path/to/dino_mm.pth \
        --output_dir ./results/mvs_luojia

    # With AMP and stronger training:
    python train_3d_reconstruction.py \
        --data_root /path/to/WHU_MVS_dataset \
        --checkpoint /path/to/dino_mm.pth \
        --output_dir ./results/mvs_whu \
        --amp --grad_clip 1.0 --warmup_epochs 2 \
        --lr 5e-4 --weight_decay 0.01 --epochs 16
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import save_task_checkpoint, set_seed
from downstream.mvs_reconstruction import (
    MVSDataset,
    build_dino_mm_casmvsnet,
    compute_mvs_metrics,
    mvs_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINO-MM + Cas-MVSNet for 3D reconstruction")
    parser.add_argument(
        "--data_root", required=True,
        help="Path to MVS dataset root (e.g. WHU_MVS_dataset or LuoJia-MVS).",
    )
    parser.add_argument("--checkpoint", default=None, help="DINO-MM pretrained checkpoint")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--num_depth", type=int, default=192)
    parser.add_argument("--ndepths", type=str, default="48,32,8")
    parser.add_argument("--depth_inter_r", type=str, default="4,2,1")
    parser.add_argument("--dlossw", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--max_h", type=int, default=512)
    parser.add_argument("--max_w", type=int, default=640)
    parser.add_argument("--trainable_blocks", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_list", default=None, help="Scene list for training")
    parser.add_argument("--val_list", default=None, help="Scene list for validation")
    # Stronger training options
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0=disable)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear LR warmup epochs")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "multistep"],
                        help="LR schedule: cosine (default) or multistep (Cas-MVSNet original)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random.")
    return args


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, dlossw, scaler=None, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        imgs = batch["imgs"].to(device)
        proj_matrices = {k: v.to(device) for k, v in batch["proj_matrices"].items()}
        depth_values = batch["depth_values"].to(device)
        depth_gt_ms = {k: v.to(device) for k, v in batch["depth_gt_ms"].items()}
        mask_ms = {k: v.to(device) for k, v in batch["mask_ms"].items()}

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                outputs = model(imgs, proj_matrices, depth_values)
                loss = mvs_loss(outputs, depth_gt_ms, mask_ms, dlossw=dlossw)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs, proj_matrices, depth_values)
            loss = mvs_loss(outputs, depth_gt_ms, mask_ms, dlossw=dlossw)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optimizer.step()

        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_acc_06m = 0.0
    total_acc_3int = 0.0
    count = 0
    for batch in loader:
        imgs = batch["imgs"].to(device)
        proj_matrices = {k: v.to(device) for k, v in batch["proj_matrices"].items()}
        depth_values = batch["depth_values"].to(device)
        depth_gt_ms = {k: v.to(device) for k, v in batch["depth_gt_ms"].items()}
        mask_ms = {k: v.to(device) for k, v in batch["mask_ms"].items()}

        outputs = model(imgs, proj_matrices, depth_values)

        # Evaluate at finest stage
        depth_pred = outputs["depth"]
        depth_gt = depth_gt_ms["stage3"]
        mask = mask_ms["stage3"]
        interval = (depth_values[0, -1] - depth_values[0, 0]).item() / max(depth_values.shape[1], 1)

        metrics = compute_mvs_metrics(depth_pred, depth_gt, mask, interval=interval)
        total_mae += metrics["mae"]
        total_acc_06m += metrics["acc_06m"]
        total_acc_3int += metrics["acc_3interval"]
        count += 1

    if count == 0:
        return {"mae": 0.0, "acc_06m": 0.0, "acc_3interval": 0.0}
    return {
        "mae": total_mae / count,
        "acc_06m": total_acc_06m / count,
        "acc_3interval": total_acc_3int / count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ndepths = [int(x) for x in args.ndepths.split(",")]
    depth_inter_r = [float(x) for x in args.depth_inter_r.split(",")]
    dlossw = [float(x) for x in args.dlossw.split(",")]

    # Build model
    model, meta = build_dino_mm_casmvsnet(
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
        ndepths=ndepths,
        depth_interals_ratio=depth_inter_r,
        device=device,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params / 1e6:.1f}M trainable / {total_params / 1e6:.1f}M total")

    # Build datasets
    train_dataset = MVSDataset(
        args.data_root, split="train", num_views=args.num_views,
        num_depth=args.num_depth, interval_scale=args.interval_scale,
        max_h=args.max_h, max_w=args.max_w, scene_list=args.train_list,
    )
    val_dataset = MVSDataset(
        args.data_root, split="val", num_views=args.num_views,
        num_depth=args.num_depth, interval_scale=args.interval_scale,
        max_h=args.max_h, max_w=args.max_w, scene_list=args.val_list,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optimizer — separate LR for backbone vs MVS head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "feature.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # AMP scaler
    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    # LR scheduler
    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 12, 14], gamma=0.5
        )
    else:
        scheduler = None  # handled manually with warmup+cosine

    # Resume
    start_epoch = 1
    best_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae = ckpt.get("metrics", {}).get("mae", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}, best MAE={best_mae:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Model: arch={meta.get('arch')}, ndepths={ndepths}, scheduler={args.scheduler}")
    print(f"AMP={'on' if scaler else 'off'}, grad_clip={args.grad_clip}, warmup={args.warmup_epochs}ep")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Set LR (warmup + cosine)
        if scheduler is None:
            lr = _get_warmup_cosine_lr(epoch - 1, args.warmup_epochs, args.epochs, args.lr)
            _set_lr(optimizer, lr)
            # backbone group gets 0.1x
            optimizer.param_groups[0]["lr"] = lr * 0.1

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, dlossw,
            scaler=scaler, grad_clip=args.grad_clip,
        )
        metrics = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"MAE={metrics['mae']:.4f} "
            f"<0.6m={metrics['acc_06m']:.1f}% "
            f"<3-int={metrics['acc_3interval']:.1f}% "
            f"lr={current_lr:.2e} "
            f"time={elapsed:.0f}s"
        )

        state = {
            "task": "3d_reconstruction",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "task_config": {
                "data_root": args.data_root,
                "num_views": args.num_views,
                "ndepths": ndepths,
                "init_mode": args.init_mode,
                "arch": meta.get("arch"),
                "patch_size": meta.get("patch_size"),
                "in_chans": meta.get("in_chans"),
                "fusion": meta.get("fusion"),
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
            },
            "backbone_meta": meta,
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)
            print(f"  -> new best MAE={best_mae:.4f}")

    print(f"\nTraining complete. Best MAE={best_mae:.4f}")


if __name__ == "__main__":
    main()

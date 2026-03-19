#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import save_task_checkpoint, set_seed
from downstream.datasets import build_segmentation_dataset, collate_segmentation_batch
from downstream.metrics import SegmentationMetric
from downstream.models import build_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on segmentation tasks")
    parser.add_argument("--dataset", required=True, choices=["uavid", "udd5", "udd6", "aeroscapes"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--framework", default="upernet", choices=["upernet", "lite_uper"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--trainable_blocks", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--aux_loss_weight", type=float, default=0.4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0=disable)")
    parser.add_argument("--warmup_iters", type=int, default=1500, help="Linear LR warmup iterations")
    parser.add_argument("--scheduler", default="poly", choices=["poly", "cosine"],
                        help="LR schedule: poly (mmseg standard) or cosine")
    parser.add_argument("--poly_power", type=float, default=0.9, help="Poly LR power")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, ignore_index, aux_loss_weight, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            if hasattr(model, "forward_with_aux"):
                logits, aux_logits = model.forward_with_aux(images)
                loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
                loss = loss + aux_loss_weight * F.cross_entropy(aux_logits, labels, ignore_index=ignore_index)
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index):
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = labels.numpy()
        for pred, target in zip(predictions, targets):
            metric.update(pred, target)
    return metric.compute()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = build_segmentation_dataset(
        args.dataset,
        args.data_root,
        split="train",
        image_size=args.image_size,
        random_flip=True,
    )
    val_dataset = build_segmentation_dataset(
        args.dataset,
        args.data_root,
        split="val",
        image_size=args.image_size,
        random_flip=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_segmentation_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_segmentation_batch,
    )

    model, backbone_meta = build_segmentation_model(
        args.checkpoint,
        num_classes=train_dataset.num_classes,
        checkpoint_key=args.checkpoint_key,
        device=device,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
        feature_dim=args.feature_dim,
        framework=args.framework,
    )

    # Separate backbone / head param groups (backbone gets 0.1x LR)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # LR schedule
    total_iters = args.epochs * len(train_loader)
    warmup_iters = min(args.warmup_iters, total_iters // 4)
    global_iter = [0]

    def _get_lr(base_lr):
        it = global_iter[0]
        if it < warmup_iters:
            return base_lr * (it + 1) / max(warmup_iters, 1)
        if args.scheduler == "poly":
            progress = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
            return base_lr * (1.0 - progress) ** args.poly_power
        # cosine
        import math
        progress = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    best_miou = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"scheduler={args.scheduler}, warmup={warmup_iters}it, total={total_iters}it")

    for epoch in range(1, args.epochs + 1):
        # Update LR per-epoch (approximate; per-iter inside train_one_epoch would be ideal
        # but this keeps the loop simple while still providing warmup + poly decay)
        lr = _get_lr(args.lr)
        optimizer.param_groups[0]["lr"] = lr * 0.1  # backbone
        optimizer.param_groups[1]["lr"] = lr         # head
        global_iter[0] = epoch * len(train_loader)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp,
            args.ignore_index,
            args.aux_loss_weight,
            grad_clip=args.grad_clip,
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=train_dataset.num_classes,
            ignore_index=args.ignore_index,
        )

        current_lr = optimizer.param_groups[1]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"mIoU={metrics['mIoU']:.2f} "
            f"pixel_acc={metrics['pixel_acc']:.2f} "
            f"lr={current_lr:.2e}"
        )

        state = {
            "task": "segmentation",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "task_config": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "num_classes": train_dataset.num_classes,
                "class_names": train_dataset.class_names,
                "feature_dim": args.feature_dim,
                "trainable_blocks": args.trainable_blocks,
                "init_mode": args.init_mode,
                "arch": backbone_meta["arch"],
                "patch_size": backbone_meta["patch_size"],
                "in_chans": backbone_meta["in_chans"],
                "fusion": backbone_meta["fusion"],
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
                "image_size": args.image_size,
                "ignore_index": args.ignore_index,
                "framework": args.framework,
                "scheduler": args.scheduler,
            },
            "backbone_meta": backbone_meta,
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["mIoU"] >= best_miou:
            best_miou = metrics["mIoU"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

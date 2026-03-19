#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import move_targets_to_device, save_task_checkpoint, set_seed
from downstream.datasets import build_detection_dataset, collate_detection_batch
from downstream.metrics import compute_detection_metrics
from downstream.models import build_detection_model
from downstream.openmmlab import infer_mmdet_detector, train_mmdet_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on detection tasks")
    parser.add_argument("--dataset", required=True,
                        choices=["dronevehicle", "llvip", "rgbt_tiny", "vt_tiny_mot", "hit_uav", "m3ot",
                                 "visdrone_det", "uavdt"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", default=None, help="Pretrain checkpoint path")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--framework", default="fasterrcnn",
                        choices=["fasterrcnn", "fcos", "cascade_rcnn"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", default="both",
                        choices=["both", "rgb_only", "ir_only"])
    parser.add_argument("--trainable_blocks", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--min_size", type=int, default=640)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping max norm (0=disable)")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Linear LR warmup epochs")
    parser.add_argument("--annotation_source", default="rgb", choices=["rgb", "ir"])
    parser.add_argument("--rgbt_subset", default="00", choices=["00", "01"])
    parser.add_argument("--val_split", default=None)
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def _build_val_split(args):
    if args.val_split is not None:
        return args.val_split
    if args.dataset in {"llvip", "rgbt_tiny", "vt_tiny_mot"}:
        return "test"
    return "val"


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, framework, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        if framework == "cascade_rcnn":
            with autocast(enabled=False):
                loss, _ = train_mmdet_detector(model, images, targets)
        else:
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)
            with autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

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
def evaluate(model, loader, device, num_classes, framework):
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in loader:
        if framework == "cascade_rcnn":
            image_ids = [target["image_id"] for target in batch_targets]
            outputs = infer_mmdet_detector(model, images, image_ids=image_ids)
        else:
            images = [image.to(device) for image in images]
            outputs = model(images)
        for output, target in zip(outputs, batch_targets):
            predictions.append(
                {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                    "image_id": target["image_id"].cpu(),
                }
            )
            targets.append(
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                    "image_id": target["image_id"].cpu(),
                }
            )
    return compute_detection_metrics(predictions, targets, num_classes=num_classes)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_kwargs = {}
    if args.dataset == "dronevehicle":
        dataset_kwargs["annotation_source"] = args.annotation_source
    if args.dataset == "rgbt_tiny":
        dataset_kwargs["subset"] = args.rgbt_subset

    train_dataset = build_detection_dataset(
        args.dataset,
        args.data_root,
        split="train",
        modality=args.modality,
        **dataset_kwargs,
    )
    val_dataset = build_detection_dataset(
        args.dataset,
        args.data_root,
        split=_build_val_split(args),
        modality=args.modality,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )

    model, backbone_meta = build_detection_model(
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
        min_size=args.min_size,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
        framework=args.framework,
    )
    # Separate backbone / head param groups (backbone gets 0.1x LR)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name and "fpn" not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Warmup + cosine schedule
    def _lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        import math
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = GradScaler(enabled=args.amp)

    best_map = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp,
            args.framework,
            grad_clip=args.grad_clip,
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=train_dataset.num_classes,
            framework=args.framework,
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"mAP50={metrics['mAP50']:.2f} "
            f"mAP={metrics['mAP']:.2f}"
        )

        state = {
            "task": "detection",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "task_config": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "modality": args.modality,
                "num_classes": train_dataset.num_classes,
                "class_names": train_dataset.class_names,
                "feature_dim": args.feature_dim,
                "trainable_blocks": args.trainable_blocks,
                "min_size": args.min_size,
                "max_size": args.max_size,
                "score_thresh": args.score_thresh,
                "annotation_source": args.annotation_source,
                "rgbt_subset": args.rgbt_subset,
                "init_mode": args.init_mode,
                "framework": args.framework,
                "arch": backbone_meta["arch"],
                "patch_size": backbone_meta["patch_size"],
                "in_chans": backbone_meta["in_chans"],
                "fusion": backbone_meta["fusion"],
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
            },
            "backbone_meta": backbone_meta,
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["mAP"] >= best_map:
            best_map = metrics["mAP"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

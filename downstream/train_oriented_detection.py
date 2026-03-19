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

from downstream.common import save_task_checkpoint, set_seed
from downstream.datasets import build_oriented_detection_dataset, collate_detection_batch
from downstream.metrics import compute_detection_metrics
from downstream.openmmlab import (
    build_mmrotate_oriented_rcnn,
    infer_mmrotate_detector,
    train_mmrotate_detector,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on oriented detection tasks")
    parser.add_argument("--dataset", required=True, choices=["dior_r", "fair1m", "custom"])
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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", default="rgb_only", choices=["rgb_only", "both", "ir_only"])
    parser.add_argument("--trainable_blocks", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--angle_version", default="le90", choices=["le90", "le135", "oc"])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--ann_dir", default=None)
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--val_split_file", default=None)
    parser.add_argument("--class_names", default=None, help="Comma-separated class names")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def _parse_class_names(raw_text):
    if not raw_text:
        return None
    names = [item.strip() for item in raw_text.split(",")]
    names = [item for item in names if item]
    return names if names else None


def train_one_epoch(model, loader, optimizer, scaler, use_amp):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        # mmrotate 0.x + mmcv1 is less stable with AMP; keep the default path in FP32.
        with autocast(enabled=False):
            loss, _ = train_mmrotate_detector(model, images, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in loader:
        image_ids = [target["image_id"] for target in batch_targets]
        outputs = infer_mmrotate_detector(model, images, image_ids=image_ids)
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
    class_names = _parse_class_names(args.class_names)

    train_dataset = build_oriented_detection_dataset(
        name=args.dataset,
        root=args.data_root,
        split=args.train_split,
        modality=args.modality,
        image_dir=args.image_dir,
        ann_dir=args.ann_dir,
        split_file=args.train_split_file,
        class_names=class_names,
        angle_version=args.angle_version,
    )
    val_dataset = build_oriented_detection_dataset(
        name=args.dataset,
        root=args.data_root,
        split=args.val_split,
        modality=args.modality,
        image_dir=args.image_dir,
        ann_dir=args.ann_dir,
        split_file=args.val_split_file,
        class_names=train_dataset.class_names,
        angle_version=args.angle_version,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training split has 0 samples, please check path/split settings.")

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

    model = build_mmrotate_oriented_rcnn(
        checkpoint_path=args.checkpoint,
        num_classes=train_dataset.num_classes,
        checkpoint_key=args.checkpoint_key,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
        feature_dim=args.feature_dim,
        score_thresh=args.score_thresh,
        angle_version=args.angle_version,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
        )
        metrics = evaluate(model, val_loader, num_classes=train_dataset.num_classes)
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"mAP50={metrics['mAP50']:.2f} "
            f"mAP={metrics['mAP']:.2f}"
        )

        state = {
            "task": "oriented_detection",
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
                "angle_version": args.angle_version,
                "feature_dim": args.feature_dim,
                "trainable_blocks": args.trainable_blocks,
                "score_thresh": args.score_thresh,
                "init_mode": args.init_mode,
                "arch": args.arch,
                "patch_size": args.patch_size,
                "in_chans": args.in_chans,
                "fusion": args.fusion,
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "image_dir": args.image_dir,
                "ann_dir": args.ann_dir,
                "train_split_file": args.train_split_file,
                "val_split_file": args.val_split_file,
            },
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["mAP"] >= best_map:
            best_map = metrics["mAP"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

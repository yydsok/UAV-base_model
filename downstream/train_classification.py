#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import RGBIR_MEAN, RGBIR_STD, save_task_checkpoint, set_seed
from downstream.datasets import build_scene_classification_dataset
from downstream.models import build_classification_model
from evaluation.eval_utils import DroneVehicleClassification


class ImageFolder4ChDataset(Dataset):
    def __init__(self, root, split="train", image_size=224):
        self.root = Path(root) / split
        self.image_size = image_size
        self.class_names = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((image_path, self.class_to_idx[class_name]))
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        ir = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
        merged = np.concatenate([image, ir], axis=2)
        tensor = torch.from_numpy(merged).permute(2, 0, 1).float()
        mean = torch.tensor(RGBIR_MEAN).view(-1, 1, 1)
        std = torch.tensor(RGBIR_STD).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, label


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on classification tasks")
    parser.add_argument("--dataset", required=True, choices=["dronevehicle", "imagefolder", "aid", "resisc45"])
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--trainable_blocks", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="Per-class train split ratio when dataset has no train/val folders.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Per-class val split ratio when dataset has no train/val folders.")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def build_datasets(args):
    if args.dataset == "dronevehicle":
        train_set = DroneVehicleClassification(args.data_root, split="train")
        val_set = DroneVehicleClassification(args.data_root, split="val")
        class_names = [train_set.class_names[idx] for idx in sorted(train_set.class_names)]
        return train_set, val_set, class_names
    if args.dataset in {"aid", "resisc45", "imagefolder"}:
        train_set = build_scene_classification_dataset(
            args.dataset,
            args.data_root,
            split="train",
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        val_set = build_scene_classification_dataset(
            args.dataset,
            args.data_root,
            split="val",
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if len(train_set) == 0 or len(val_set) == 0:
            raise RuntimeError(
                f"{args.dataset} split is empty. Please check --data_root and split ratios."
            )
        return train_set, val_set, train_set.class_names
    train_set = ImageFolder4ChDataset(args.data_root, split="train", image_size=args.image_size)
    val_set = ImageFolder4ChDataset(args.data_root, split="val", image_size=args.image_size)
    return train_set, val_set, train_set.class_names


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
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
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set, val_set, class_names = build_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, backbone_meta = build_classification_model(
        args.checkpoint,
        num_classes=len(class_names),
        checkpoint_key=args.checkpoint_key,
        device=device,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    best_acc = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch={epoch:03d} loss={train_loss:.4f} acc={val_acc:.2f}")

        state = {
            "task": "classification",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "task_config": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "class_names": class_names,
                "num_classes": len(class_names),
                "init_mode": args.init_mode,
                "arch": backbone_meta["arch"],
                "patch_size": backbone_meta["patch_size"],
                "in_chans": backbone_meta["in_chans"],
                "fusion": backbone_meta["fusion"],
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
                "trainable_blocks": args.trainable_blocks,
            },
            "backbone_meta": backbone_meta,
            "metrics": {"acc": val_acc},
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if val_acc >= best_acc:
            best_acc = val_acc
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

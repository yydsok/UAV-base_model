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

from downstream.change_detection import build_change_detection_model
from downstream.common import save_task_checkpoint, set_seed
from downstream.datasets import build_change_detection_dataset, collate_change_detection_batch
from downstream.metrics import compute_change_detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on change detection tasks")
    parser.add_argument("--dataset", required=True, choices=["levir_cd", "cdd", "oscd", "dsifn_cd", "custom"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--trainable_blocks", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--token_len", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0=disable)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs")
    parser.add_argument("--aux_loss_weight", type=float, default=0.4, help="Deep supervision aux loss weight")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, aux_loss_weight=0.4, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for image_a, image_b, labels in loader:
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            output = model(image_a, image_b)
            if isinstance(output, tuple):
                logits, aux_logits = output
                loss = F.cross_entropy(logits, labels)
                loss = loss + aux_loss_weight * F.cross_entropy(aux_logits, labels)
            else:
                loss = F.cross_entropy(output, labels)
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
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for image_a, image_b, labels in loader:
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        logits = model(image_a, image_b)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(list(pred))
        gts.extend(list(labels.numpy()))
    return compute_change_detection_metrics(preds, gts)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = build_change_detection_dataset(
        args.dataset,
        args.data_root,
        split=args.train_split,
        image_size=args.image_size,
    )
    val_set = build_change_detection_dataset(
        args.dataset,
        args.data_root,
        split=args.val_split,
        image_size=args.image_size,
    )
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("Change detection dataset is empty. Please verify data_root and split files.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_change_detection_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_change_detection_batch,
    )

    model, backbone_meta = build_change_detection_model(
        args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        token_len=args.token_len,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
    )
    # Separate backbone / head param groups
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

    # Warmup + cosine schedule
    def _lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        import math
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = GradScaler(enabled=args.amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            use_amp=args.amp,
            aux_loss_weight=args.aux_loss_weight,
            grad_clip=args.grad_clip,
        )
        metrics = evaluate(model, val_loader, device=device)
        scheduler.step()
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"F1={metrics['F1']:.2f} "
            f"IoU={metrics['IoU']:.2f}"
        )

        state = {
            "task": "change_detection",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "task_config": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "image_size": args.image_size,
                "init_mode": args.init_mode,
                "arch": backbone_meta["arch"],
                "patch_size": backbone_meta["patch_size"],
                "in_chans": backbone_meta["in_chans"],
                "fusion": backbone_meta["fusion"],
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
                "trainable_blocks": args.trainable_blocks,
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "token_len": args.token_len,
                "encoder_layers": args.encoder_layers,
                "decoder_layers": args.decoder_layers,
                "num_heads": args.num_heads,
            },
            "backbone_meta": backbone_meta,
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["F1"] >= best_f1:
            best_f1 = metrics["F1"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

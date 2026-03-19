"""
Linear Probing Evaluation for DINO-MM.

Freezes the pretrained backbone and trains a linear classifier on concatenated
CLS tokens from the last N transformer blocks.

Supports DDP multi-GPU training for faster feature extraction.

Reference: DINO / DINOv2 linear probing protocol.

Usage:
    # Single GPU
    python eval_linear_probe.py --checkpoint /path/to/checkpoint.pth

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 eval_linear_probe.py --checkpoint /path/to/checkpoint.pth
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, DroneVehicleClassification, get_num_classes,
    prepare_input, infer_modality_masks, DRONEVEHICLE_CLASS_MAP,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features."""
    def __init__(self, dim, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class SmoothedValue:
    """Track a smoothed value over a sliding window."""
    def __init__(self, window_size=20):
        self.window = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)

    @property
    def median(self):
        if not self.window:
            return 0
        return sorted(self.window)[len(self.window) // 2]


def init_distributed():
    """Initialize DDP if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def train_one_epoch(model, classifier, optimizer, loader, epoch,
                    n_last_blocks, modality_mode, device):
    """Train linear classifier for one epoch."""
    classifier.train()
    loss_meter = SmoothedValue()
    acc_meter = SmoothedValue()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inp, target) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Apply modality masking
        inp = prepare_input(inp, modality_mode)
        modality_masks = infer_modality_masks(inp)

        # Forward through frozen backbone
        with torch.no_grad():
            intermediate = model.get_intermediate_layers(
                inp, n=n_last_blocks, modality_masks=modality_masks)
            feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)

        # Forward through classifier
        logits = classifier(feat)
        loss = criterion(logits, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pred = logits.argmax(dim=1)
        acc = (pred == target).float().mean().item() * 100
        loss_meter.update(loss.item(), n=inp.shape[0])
        acc_meter.update(acc, n=inp.shape[0])

        if is_main_process() and (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.2f}%")

    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}


@torch.no_grad()
def validate(model, classifier, loader, n_last_blocks, modality_mode, device):
    """Validate linear classifier."""
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for inp, target in loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        inp = prepare_input(inp, modality_mode)
        modality_masks = infer_modality_masks(inp)

        intermediate = model.get_intermediate_layers(
            inp, n=n_last_blocks, modality_masks=modality_masks)
        feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)

        logits = classifier(feat)
        loss = criterion(logits, target)

        pred = logits.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        total_loss += loss.item() * inp.shape[0]
        total_samples += inp.shape[0]

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    acc = (preds == targets).float().mean().item() * 100
    avg_loss = total_loss / max(total_samples, 1)

    # Per-class accuracy
    num_classes = get_num_classes()
    per_class = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == targets[mask]).float().mean().item() * 100

    # Confusion matrix
    cm = confusion_matrix(targets.numpy(), preds.numpy(),
                          labels=list(range(num_classes)))

    return {
        'loss': avg_loss,
        'acc': acc,
        'per_class': per_class,
        'confusion_matrix': cm,
    }


def eval_linear_probe(ckpt_path, dataset_root=DRONEVEHICLE_ROOT,
                      n_last_blocks=4, epochs=100, lr=0.001,
                      batch_size=128, modality_mode='both',
                      device='cuda', output_dir=None, num_workers=4):
    """Run linear probing evaluation.

    Args:
        ckpt_path: path to pretrained checkpoint
        dataset_root: DroneVehicle root
        n_last_blocks: number of last blocks for feature concatenation
        epochs: training epochs
        lr: base learning rate
        batch_size: per-GPU batch size
        modality_mode: 'both', 'rgb_only', 'ir_only'
        device: torch device
        output_dir: output directory

    Returns:
        dict with best accuracy and final results
    """
    rank, world_size, local_rank = init_distributed()
    is_ddp = world_size > 1

    if is_ddp:
        device = f'cuda:{local_rank}'

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'linear_probe')
    os.makedirs(output_dir, exist_ok=True)

    cudnn.benchmark = True

    # Build model
    model, train_args = build_model(ckpt_path, device=device)
    embed_dim = model.embed_dim * n_last_blocks
    num_classes = get_num_classes()

    if is_main_process():
        print(f"\nLinear Probing Config:")
        print(f"  Feature dim: {embed_dim} ({n_last_blocks} blocks x {model.embed_dim})")
        print(f"  Classes: {num_classes}")
        print(f"  Modality: {modality_mode}")
        print(f"  Epochs: {epochs}")
        print(f"  LR: {lr}")

    # Build classifier
    classifier = LinearClassifier(embed_dim, num_classes).to(device)
    if is_ddp:
        classifier = nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank])

    # Datasets
    train_dataset = DroneVehicleClassification(dataset_root, split='train')
    val_dataset = DroneVehicleClassification(dataset_root, split='val')

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    # Optimizer + scheduler
    effective_lr = lr * (batch_size * world_size) / 256.0
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=effective_lr,
        momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    # Training loop
    best_acc = 0.0
    log_entries = []

    for epoch in range(epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, classifier, optimizer, train_loader, epoch,
            n_last_blocks, modality_mode, device)
        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_stats = validate(
                model, classifier, val_loader,
                n_last_blocks, modality_mode, device)

            if is_main_process():
                print(f"\n  Epoch {epoch}: Val Acc = {val_stats['acc']:.2f}% "
                      f"(Loss: {val_stats['loss']:.4f})")
                for c, acc in sorted(val_stats['per_class'].items()):
                    class_name = {v: k for k, v in DRONEVEHICLE_CLASS_MAP.items()}.get(c, str(c))
                    print(f"    {class_name}: {acc:.2f}%")

                if val_stats['acc'] > best_acc:
                    best_acc = val_stats['acc']
                    # Save best classifier
                    save_path = os.path.join(
                        output_dir, f'best_linear_{modality_mode}.pth')
                    state = classifier.module.state_dict() if is_ddp else classifier.state_dict()
                    torch.save({
                        'state_dict': state,
                        'epoch': epoch,
                        'acc': best_acc,
                        'modality_mode': modality_mode,
                    }, save_path)

            log_entry = {
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'train_acc': train_stats['acc'],
                'val_loss': val_stats['loss'],
                'val_acc': val_stats['acc'],
            }
            log_entries.append(log_entry)

    # Save logs
    if is_main_process():
        log_path = os.path.join(output_dir, f'log_{modality_mode}.json')
        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Linear Probe Results ({modality_mode}):")
        print(f"  Best Accuracy: {best_acc:.2f}%")
        print(f"  Logs saved to: {log_path}")
        print(f"{'='*50}")

    if is_ddp:
        dist.destroy_process_group()

    return {'best_acc': best_acc, 'log': log_entries}


def main():
    parser = argparse.ArgumentParser(description='Linear Probing Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--n_last_blocks', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--modality_modes', nargs='+', type=str,
                        default=['both', 'rgb_only', 'ir_only'],
                        help='Modality modes to evaluate')
    args = parser.parse_args()

    results = {}
    for mode in args.modality_modes:
        print(f"\n{'='*60}")
        print(f"Linear Probe — {mode}")
        print(f"{'='*60}")

        result = eval_linear_probe(
            ckpt_path=args.checkpoint,
            dataset_root=args.dataset_root,
            n_last_blocks=args.n_last_blocks,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            modality_mode=mode,
            device=args.device,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
        results[mode] = result

    # Summary
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Linear Probe Summary")
        print(f"{'='*60}")
        print(f"{'Mode':<15} | {'Best Acc':>10}")
        print(f"{'-'*15} | {'-'*10}")
        for mode in args.modality_modes:
            print(f"{mode:<15} | {results[mode]['best_acc']:>9.2f}%")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import save_task_checkpoint, set_seed
from downstream.datasets import build_retrieval_dataset
from downstream.eval_retrieval import compute_retrieval_metrics
from downstream.models import build_retrieval_model


DEFAULT_DATA_ROOTS = {
    "dronevehicle": "/root/autodl-tmp/data/DroneVehicle",
    "llvip": "/root/autodl-tmp/data/LLVIP/registered",
    "sues_200": "/root/autodl-tmp/data/SUES-200",
    "cvogl": "/root/autodl-tmp/data/CVOGL",
}

DEFAULT_VIEWS = {
    "dronevehicle": ("rgb", "ir"),
    "llvip": ("rgb", "ir"),
    "sues_200": ("drone", "satellite"),
    "cvogl": ("drone", "svi"),
}


class PairedRetrievalDataset(Dataset):
    def __init__(self, query_dataset, gallery_dataset, max_pairs=None):
        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
        query_by_label = defaultdict(list)
        gallery_by_label = defaultdict(list)

        for idx, sample in enumerate(query_dataset.samples):
            query_by_label[int(sample["label"])].append(idx)
        for idx, sample in enumerate(gallery_dataset.samples):
            gallery_by_label[int(sample["label"])].append(idx)

        shared_labels = sorted(set(query_by_label) & set(gallery_by_label))
        self.pairs = []
        for label in shared_labels:
            q_idx = query_by_label[label]
            g_idx = gallery_by_label[label]
            pair_count = max(len(q_idx), len(g_idx))
            for i in range(pair_count):
                self.pairs.append((q_idx[i % len(q_idx)], g_idx[i % len(g_idx)], label))
                if max_pairs is not None and len(self.pairs) >= max_pairs:
                    return

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        q_index, g_index, label = self.pairs[index]
        q_image, _, _ = self.query_dataset[q_index]
        g_image, _, _ = self.gallery_dataset[g_index]
        return q_image, g_image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINO-MM on retrieval tasks")
    parser.add_argument("--dataset", required=True, choices=["dronevehicle", "llvip", "sues_200", "cvogl"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--query_view", default=None)
    parser.add_argument("--gallery_view", default=None)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--eval_split", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--trainable_blocks", type=int, default=12)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_eval_query_samples", type=int, default=None)
    parser.add_argument("--max_eval_gallery_samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.init_mode != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --init_mode random is used.")
    return args


def resolve_data_root(dataset, data_root):
    if data_root:
        return data_root
    if dataset in DEFAULT_DATA_ROOTS:
        return DEFAULT_DATA_ROOTS[dataset]
    raise ValueError(f"Please provide --data_root for dataset '{dataset}'.")


def resolve_views(dataset, query_view=None, gallery_view=None):
    default_query, default_gallery = DEFAULT_VIEWS[dataset]
    return query_view or default_query, gallery_view or default_gallery


def resolve_train_split(dataset, split):
    if split is not None:
        return split
    if dataset == "dronevehicle":
        return "train"
    if dataset == "llvip":
        return "train"
    return None


def resolve_eval_split(dataset, split):
    if split is not None:
        return split
    if dataset == "dronevehicle":
        return "val"
    if dataset == "llvip":
        return "test"
    return None


def extract_features(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device, non_blocking=True)
            feats = model(images)
            features.append(feats.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def evaluate_retrieval(model, query_dataset, gallery_dataset, args, device):
    q_feat, q_labels = extract_features(
        model,
        query_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    g_feat, g_labels = extract_features(
        model,
        gallery_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    return compute_retrieval_metrics(
        q_feat,
        g_feat,
        q_labels,
        g_labels,
        chunk_size=args.chunk_size,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for query_images, gallery_images, _labels in loader:
        query_images = query_images.to(device, non_blocking=True)
        gallery_images = gallery_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            query_feats = model(query_images)
            gallery_feats = model(gallery_images)
            logit_scale = model.logit_scale.exp().clamp(max=100.0)
            logits = logit_scale * query_feats @ gallery_feats.t()
            targets = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, targets) +
                F.cross_entropy(logits.t(), targets)
            )
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = resolve_data_root(args.dataset, args.data_root)
    query_view, gallery_view = resolve_views(args.dataset, args.query_view, args.gallery_view)
    train_split = resolve_train_split(args.dataset, args.train_split)
    eval_split = resolve_eval_split(args.dataset, args.eval_split)

    train_query_dataset = build_retrieval_dataset(
        args.dataset,
        data_root,
        view=query_view,
        split=train_split,
        image_size=args.image_size,
    )
    train_gallery_dataset = build_retrieval_dataset(
        args.dataset,
        data_root,
        view=gallery_view,
        split=train_split,
        image_size=args.image_size,
    )
    paired_dataset = PairedRetrievalDataset(
        train_query_dataset,
        train_gallery_dataset,
        max_pairs=args.max_pairs,
    )
    if len(paired_dataset) == 0:
        raise RuntimeError("No valid query-gallery pairs were found for retrieval training.")

    eval_query_dataset = build_retrieval_dataset(
        args.dataset,
        data_root,
        view=query_view,
        split=eval_split,
        image_size=args.image_size,
        max_samples=args.max_eval_query_samples,
    )
    eval_gallery_dataset = build_retrieval_dataset(
        args.dataset,
        data_root,
        view=gallery_view,
        split=eval_split,
        image_size=args.image_size,
        max_samples=args.max_eval_gallery_samples,
    )

    train_loader = DataLoader(
        paired_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model, backbone_meta = build_retrieval_model(
        args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
        init_mode=args.init_mode,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        trainable_blocks=args.trainable_blocks,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        temperature=args.temperature,
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
    best_rank1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            use_amp=args.amp,
        )
        metrics = evaluate_retrieval(
            model,
            eval_query_dataset,
            eval_gallery_dataset,
            args=args,
            device=device,
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"rank1={metrics['rank_1']:.2f} "
            f"mAP={metrics['mAP']:.2f} "
            f"MRR={metrics['MRR']:.2f}"
        )

        state = {
            "task": "retrieval",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "task_config": {
                "dataset": args.dataset,
                "data_root": data_root,
                "query_view": query_view,
                "gallery_view": gallery_view,
                "train_split": train_split,
                "eval_split": eval_split,
                "init_mode": args.init_mode,
                "arch": backbone_meta["arch"],
                "patch_size": backbone_meta["patch_size"],
                "in_chans": backbone_meta["in_chans"],
                "fusion": backbone_meta["fusion"],
                "checkpoint": args.checkpoint,
                "checkpoint_key": args.checkpoint_key,
                "trainable_blocks": args.trainable_blocks,
                "projection_dim": args.projection_dim,
                "temperature": args.temperature,
            },
            "backbone_meta": backbone_meta,
            "metrics": metrics,
            "args": vars(args),
        }
        save_task_checkpoint(output_dir / "checkpoint_latest.pth", state)
        if metrics["rank_1"] >= best_rank1:
            best_rank1 = metrics["rank_1"]
            save_task_checkpoint(output_dir / "checkpoint_best.pth", state)


if __name__ == "__main__":
    main()

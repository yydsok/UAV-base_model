#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import build_backbone_from_checkpoint, set_seed
from downstream.datasets import build_retrieval_dataset


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINO-MM on retrieval tasks")
    parser.add_argument("--dataset", default="dronevehicle", choices=["dronevehicle", "llvip", "sues_200", "cvogl"])
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
    parser.add_argument("--split", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_query_samples", type=int, default=None)
    parser.add_argument("--max_gallery_samples", type=int, default=None)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--output_json", default=None)
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


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    for images, batch_labels, _ in loader:
        images = images.to(device)
        feats = model(images, return_all_tokens=False)
        features.append(F.normalize(feats, dim=1).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_retrieval_metrics(query_features, gallery_features, query_labels, gallery_labels, ks=(1, 5, 10), chunk_size=1024):
    if query_features.numel() == 0 or gallery_features.numel() == 0:
        raise ValueError("Query and gallery features must be non-empty.")

    gallery_features = F.normalize(gallery_features, dim=1)
    query_features = F.normalize(query_features, dim=1)
    query_labels = query_labels.view(-1)
    gallery_labels = gallery_labels.view(-1)

    hits = {k: 0 for k in ks}
    ap_sum = 0.0
    mrr_sum = 0.0
    first_ranks = []
    valid_queries = 0

    for start in range(0, query_features.shape[0], chunk_size):
        end = min(start + chunk_size, query_features.shape[0])
        sim = query_features[start:end] @ gallery_features.T
        ranking = sim.argsort(dim=1, descending=True)
        ranked_labels = gallery_labels[ranking]
        positive_mask = ranked_labels.eq(query_labels[start:end].unsqueeze(1))

        for row in positive_mask:
            positive_ranks = torch.nonzero(row, as_tuple=False).flatten()
            if positive_ranks.numel() == 0:
                continue
            valid_queries += 1
            first_rank = int(positive_ranks[0].item())
            first_ranks.append(first_rank + 1)
            mrr_sum += 1.0 / float(first_rank + 1)
            for k in ks:
                hits[k] += int(torch.any(positive_ranks < k).item())

            precision_at_hits = torch.arange(
                1,
                positive_ranks.numel() + 1,
                dtype=torch.float32,
            ) / (positive_ranks.float() + 1.0)
            ap_sum += float(precision_at_hits.mean().item())

    denom = max(valid_queries, 1)
    metrics = {
        f"rank_{k}": 100.0 * hits[k] / denom for k in ks
    }
    metrics.update(
        {
            "mAP": 100.0 * ap_sum / denom,
            "MRR": 100.0 * mrr_sum / denom,
            "median_rank": float(torch.tensor(first_ranks, dtype=torch.float32).median().item()) if first_ranks else 0.0,
            "num_queries": int(query_features.shape[0]),
            "num_gallery": int(gallery_features.shape[0]),
            "valid_queries": int(valid_queries),
        }
    )
    return metrics


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def run_direction(model, dataset_name, data_root, split, query_view, gallery_view, image_size, batch_size, num_workers,
                  max_query_samples, max_gallery_samples, query_stride, device, chunk_size):
    extra_kwargs = {}
    if dataset_name == "sues_200":
        extra_kwargs["query_stride"] = max(1, query_stride)

    query_dataset = build_retrieval_dataset(
        dataset_name,
        data_root,
        view=query_view,
        split=split,
        image_size=image_size,
        max_samples=max_query_samples,
        **extra_kwargs,
    )
    gallery_dataset = build_retrieval_dataset(
        dataset_name,
        data_root,
        view=gallery_view,
        split=split,
        image_size=image_size,
        max_samples=max_gallery_samples,
        **extra_kwargs,
    )

    query_loader = build_loader(query_dataset, batch_size=batch_size, num_workers=num_workers)
    gallery_loader = build_loader(gallery_dataset, batch_size=batch_size, num_workers=num_workers)

    query_features, query_labels = extract_features(model, query_loader, device=device)
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device=device)
    return compute_retrieval_metrics(
        query_features,
        gallery_features,
        query_labels,
        gallery_labels,
        chunk_size=chunk_size,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    data_root = resolve_data_root(args.dataset, args.data_root)
    query_view, gallery_view = resolve_views(args.dataset, args.query_view, args.gallery_view)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone, _, backbone_meta = build_backbone_from_checkpoint(
        args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        fusion=args.fusion,
        load_temporal=False,
        init_mode=args.init_mode,
    )
    backbone.eval()

    results = {
        "dataset": args.dataset,
        "data_root": data_root,
        "init_mode": args.init_mode,
        "backbone": backbone_meta,
        "query_view": query_view,
        "gallery_view": gallery_view,
        "query_to_gallery": run_direction(
            backbone,
            args.dataset,
            data_root,
            args.split,
            query_view,
            gallery_view,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.max_query_samples,
            args.max_gallery_samples,
            args.query_stride,
            device,
            args.chunk_size,
        ),
    }

    if args.bidirectional:
        results["gallery_to_query"] = run_direction(
            backbone,
            args.dataset,
            data_root,
            args.split,
            gallery_view,
            query_view,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.max_gallery_samples,
            args.max_query_samples,
            args.query_stride,
            device,
            args.chunk_size,
        )

    print(json.dumps(results, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

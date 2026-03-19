#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.change_detection import build_change_detection_model
from downstream.datasets import build_change_detection_dataset, collate_change_detection_batch
from downstream.metrics import compute_change_detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINO-MM change detection model")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_best.pth from train_change_detection.py")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def load_model(task_checkpoint, device):
    state = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
    config = state["task_config"]
    model, _ = build_change_detection_model(
        config["checkpoint"],
        checkpoint_key=config.get("checkpoint_key", "teacher"),
        device=device,
        init_mode=config.get("init_mode", "pretrained"),
        arch=config.get("arch"),
        patch_size=config.get("patch_size"),
        in_chans=config.get("in_chans"),
        fusion=config.get("fusion"),
        trainable_blocks=config.get("trainable_blocks", 4),
        feature_dim=config.get("feature_dim", 256),
        hidden_dim=config.get("hidden_dim", 128),
        token_len=config.get("token_len", 4),
        encoder_layers=config.get("encoder_layers", 2),
        decoder_layers=config.get("decoder_layers", 1),
        num_heads=config.get("num_heads", 4),
    )
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    return model, config


@torch.no_grad()
def evaluate(model, loader, device):
    predictions = []
    targets = []
    model.eval()
    for image_a, image_b, labels in loader:
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        logits = model(image_a, image_b)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(list(pred))
        targets.extend(list(labels.numpy()))
    return compute_change_detection_metrics(predictions, targets)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)

    data_root = args.data_root or config["data_root"]
    split = args.split or config.get("val_split", "val")
    dataset = build_change_detection_dataset(
        config["dataset"],
        data_root,
        split=split,
        image_size=config.get("image_size", 256),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_change_detection_batch,
    )
    metrics = evaluate(model, loader, device=device)
    payload = {"split": split, "metrics": metrics}
    print(json.dumps(payload, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

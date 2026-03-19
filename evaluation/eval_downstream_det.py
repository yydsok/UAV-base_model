"""
Downstream Detection Evaluation (Simplified) for DINO-MM.

Instead of a full detection framework, this script:
1. Crops object regions from DroneVehicle using VOC annotations
2. Extracts features from cropped regions using the pretrained backbone
3. Trains a linear classifier on the crop features
4. Reports classification accuracy and mAP

This avoids heavy dependencies like mmdet or detectron2.

Usage:
    python eval_downstream_det.py --checkpoint /path/to/checkpoint.pth
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (
    build_model, build_random_model, extract_features,
    DroneVehicleCropDataset, get_dataloader, get_num_classes,
    prepare_input, infer_modality_masks, DRONEVEHICLE_CLASS_MAP,
)

DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


@torch.no_grad()
def extract_crop_features(model, dataloader, n_last_blocks=4,
                          modality_mode='both', device='cuda', verbose=True):
    """Extract features from cropped object regions.

    Returns:
        features: [N, D]
        labels: [N]
    """
    model.eval()
    all_features = []
    all_labels = []
    n_collected = 0

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = prepare_input(imgs, modality_mode)
        modality_masks = infer_modality_masks(imgs)

        if n_last_blocks > 1:
            intermediate = model.get_intermediate_layers(
                imgs, n=n_last_blocks, modality_masks=modality_masks)
            feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)
        else:
            feat = model(imgs, return_all_tokens=False, modality_masks=modality_masks)

        all_features.append(feat.cpu())
        all_labels.append(labels)
        n_collected += imgs.shape[0]

        if verbose and (batch_idx + 1) % 100 == 0:
            print(f"  Extracted {n_collected} crops...")

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if verbose:
        print(f"  Total: {features.shape[0]} crops, dim={features.shape[1]}")
    return features, labels


def train_linear_on_crops(train_features, train_labels, val_features, val_labels,
                          num_classes, epochs=50, lr=0.01):
    """Train a linear classifier on extracted crop features.

    Returns:
        dict with accuracy, mAP, classification report
    """
    feat_dim = train_features.shape[1]
    classifier = LinearClassifier(feat_dim, num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    # Mini-batch training on features (features already in memory)
    batch_size = 256
    n_train = train_features.shape[0]
    train_features_gpu = train_features.cuda()
    train_labels_gpu = train_labels.cuda()

    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            feat = train_features_gpu[idx]
            target = train_labels_gpu[idx]

            logits = classifier(feat)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (n_train // batch_size + 1)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_features.cuda())
        val_preds = val_logits.argmax(dim=1).cpu()
        val_probs = torch.softmax(val_logits, dim=1).cpu()

    # Accuracy
    acc = (val_preds == val_labels).float().mean().item() * 100

    # Per-class accuracy
    class_names = {v: k for k, v in DRONEVEHICLE_CLASS_MAP.items()}
    per_class = {}
    for c in range(num_classes):
        mask = val_labels == c
        if mask.sum() > 0:
            per_class[class_names.get(c, str(c))] = (
                (val_preds[mask] == val_labels[mask]).float().mean().item() * 100
            )

    # mAP
    try:
        # One-hot encode labels for mAP
        labels_onehot = np.zeros((len(val_labels), num_classes))
        for i, l in enumerate(val_labels):
            labels_onehot[i, l] = 1
        mAP = average_precision_score(labels_onehot, val_probs.numpy(), average='macro') * 100
    except Exception:
        mAP = 0.0

    # Confusion matrix
    cm = confusion_matrix(val_labels.numpy(), val_preds.numpy(),
                          labels=list(range(num_classes)))

    return {
        'accuracy': acc,
        'mAP': mAP,
        'per_class': per_class,
        'confusion_matrix': cm,
    }


def eval_downstream_det(ckpt_path, dataset_root=DRONEVEHICLE_ROOT,
                        n_last_blocks=4, epochs=50, batch_size=64,
                        modality_mode='both', device='cuda', output_dir=None):
    """Run downstream detection evaluation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'downstream_det')
    os.makedirs(output_dir, exist_ok=True)

    model, args = build_model(ckpt_path, device=device)
    num_classes = get_num_classes()

    # Build crop datasets
    print("\nLoading crop datasets...")
    train_dataset = DroneVehicleCropDataset(dataset_root, split='train')
    val_dataset = DroneVehicleCropDataset(dataset_root, split='val')

    train_loader = get_dataloader(train_dataset, batch_size=batch_size,
                                  num_workers=4, shuffle=False)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size,
                                num_workers=4, shuffle=False)

    # Extract features
    print(f"\nExtracting train crop features ({modality_mode})...")
    train_features, train_labels = extract_crop_features(
        model, train_loader, n_last_blocks=n_last_blocks,
        modality_mode=modality_mode, device=device)

    print(f"Extracting val crop features ({modality_mode})...")
    val_features, val_labels = extract_crop_features(
        model, val_loader, n_last_blocks=n_last_blocks,
        modality_mode=modality_mode, device=device)

    # Train linear classifier
    print(f"\nTraining linear classifier on crops...")
    results = train_linear_on_crops(
        train_features, train_labels,
        val_features, val_labels,
        num_classes=num_classes,
        epochs=epochs,
    )

    print(f"\n{'='*50}")
    print(f"Downstream Detection Results ({modality_mode}):")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  mAP:      {results['mAP']:.2f}%")
    print(f"  Per-class:")
    for cls_name, acc in sorted(results['per_class'].items()):
        print(f"    {cls_name}: {acc:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"  {results['confusion_matrix']}")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Downstream Detection (Crop-based)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--n_last_blocks', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--random_baseline', action='store_true')
    parser.add_argument('--modality_modes', nargs='+', type=str,
                        default=['both', 'rgb_only', 'ir_only'])
    args = parser.parse_args()

    all_results = {}

    for mode in args.modality_modes:
        print(f"\n{'='*60}")
        print(f"Downstream Detection — {mode}")
        print(f"{'='*60}")

        result = eval_downstream_det(
            ckpt_path=args.checkpoint,
            dataset_root=args.dataset_root,
            n_last_blocks=args.n_last_blocks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            modality_mode=mode,
            device=args.device,
            output_dir=args.output_dir,
        )
        all_results[mode] = result

    if args.random_baseline:
        print(f"\n{'='*60}")
        print(f"Random Baseline")
        print(f"{'='*60}")

        # Use a temporary random model
        random_model = build_random_model(device=args.device)

        train_dataset = DroneVehicleCropDataset(args.dataset_root, split='train')
        val_dataset = DroneVehicleCropDataset(args.dataset_root, split='val')

        train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, num_workers=4)
        val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, num_workers=4)

        train_feat, train_lab = extract_crop_features(
            random_model, train_loader, n_last_blocks=args.n_last_blocks,
            device=args.device)
        val_feat, val_lab = extract_crop_features(
            random_model, val_loader, n_last_blocks=args.n_last_blocks,
            device=args.device)

        rand_result = train_linear_on_crops(
            train_feat, train_lab, val_feat, val_lab,
            num_classes=get_num_classes(), epochs=args.epochs)

        all_results['random'] = rand_result
        print(f"\n  Random Baseline Acc: {rand_result['accuracy']:.2f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"Downstream Detection Summary")
    print(f"{'='*60}")
    print(f"{'Mode':<15} | {'Accuracy':>10} | {'mAP':>10}")
    print(f"{'-'*15} | {'-'*10} | {'-'*10}")
    for mode, result in all_results.items():
        print(f"{mode:<15} | {result['accuracy']:>9.2f}% | {result['mAP']:>9.2f}%")


if __name__ == '__main__':
    main()

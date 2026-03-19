"""
Comprehensive Evaluation for DINO-MM Pretrained Models.

Unified script that runs all evaluation benchmarks and compares three models:
  1. Random Init (ViT-S/16, 4ch) — performance lower bound
  2. DINO ViT-S/16 (ImageNet, 3ch) — RGB-only baseline
  3. DINO-MM (ours, 4ch) — multi-modal pretrained

Evaluations (ordered by speed):
  1. RankMe (~2min): effective rank of feature representations
  2. KNN Classification (~10min): k=10,20 on DroneVehicle 5-class
  3. Cross-Modal Retrieval (~5min): DroneVehicle + LLVIP, 4ch models only
  4. Linear Probe (~60-90min): frozen backbone + linear classifier, 100 epochs
  5. Detection (~20-30min): crop-based classification, 50 epochs

Usage:
    # Quick (skip linear probe + detection, ~15 min)
    python evaluation/eval_comprehensive.py \
        --checkpoint checkpoint_latest.pth \
        --dino_baseline dino_deitsmall16_pretrain.pth \
        --skip_linear --skip_det

    # Full (~90-120 min)
    python evaluation/eval_comprehensive.py \
        --checkpoint checkpoint_latest.pth \
        --dino_baseline dino_deitsmall16_pretrain.pth
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch

# Add project root and evaluation dir to path
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(EVAL_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from eval_utils import (
    build_model, build_random_model, build_dino_baseline,
    extract_features, extract_features_3ch,
    DroneVehicleClassification, DroneVehicleClassification3Ch,
    DroneVehicleCropDataset,
    LLVIPPaired, DroneVehiclePaired,
    get_dataloader, get_num_classes, prepare_input, infer_modality_masks,
)
from eval_rankme import compute_rankme
from eval_knn import knn_classify
from eval_crossmodal_retrieval import extract_paired_features, compute_retrieval_metrics
from eval_downstream_det import extract_crop_features, train_linear_on_crops

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DRONEVEHICLE_ROOT = '/root/autodl-tmp/data/DroneVehicle'
LLVIP_ROOT = '/root/autodl-tmp/data/LLVIP/registered/'


# ---------------------------------------------------------------------------
# Per-evaluation runners
# ---------------------------------------------------------------------------

def run_rankme(model, dataset_root, num_samples=2000, batch_size=64,
               device='cuda', is_3ch=False):
    """Compute RankMe for a single model."""
    if is_3ch:
        ds = DroneVehicleClassification3Ch(dataset_root, split='train')
        loader = get_dataloader(ds, batch_size=batch_size, num_workers=12)
        features, _ = extract_features_3ch(model, loader, device=device,
                                           max_samples=num_samples)
    else:
        ds = DroneVehicleClassification(dataset_root, split='train')
        loader = get_dataloader(ds, batch_size=batch_size, num_workers=12)
        features, _ = extract_features(model, loader, modality_mode='both',
                                       device=device, max_samples=num_samples)
    rank, _ = compute_rankme(features)
    return rank


def run_knn(model, dataset_root, k_values=[10, 20], batch_size=64,
            device='cuda', modality_mode='both', is_3ch=False):
    """Run KNN classification for one model + one modality."""
    if is_3ch:
        train_ds = DroneVehicleClassification3Ch(dataset_root, split='train')
        val_ds = DroneVehicleClassification3Ch(dataset_root, split='val')
        train_loader = get_dataloader(train_ds, batch_size=batch_size, num_workers=12)
        val_loader = get_dataloader(val_ds, batch_size=batch_size, num_workers=12)
        train_feat, train_lab = extract_features_3ch(model, train_loader, device=device)
        val_feat, val_lab = extract_features_3ch(model, val_loader, device=device)
    else:
        train_ds = DroneVehicleClassification(dataset_root, split='train')
        val_ds = DroneVehicleClassification(dataset_root, split='val')
        train_loader = get_dataloader(train_ds, batch_size=batch_size, num_workers=12)
        val_loader = get_dataloader(val_ds, batch_size=batch_size, num_workers=12)
        train_feat, train_lab = extract_features(model, train_loader,
                                                 modality_mode=modality_mode,
                                                 device=device)
        val_feat, val_lab = extract_features(model, val_loader,
                                             modality_mode=modality_mode,
                                             device=device)

    num_classes = get_num_classes()
    results = {}
    for k in k_values:
        res = knn_classify(train_feat, train_lab, val_feat, val_lab,
                           k=k, num_classes=num_classes)
        results[k] = res['top1']
        print(f"    k={k}: {res['top1']:.2f}%")
    return results


def run_retrieval(model, dataset_name, dataset_root, batch_size=64,
                  device='cuda', max_samples=None):
    """Run cross-modal retrieval for one 4ch model on one dataset."""
    if dataset_name == 'llvip':
        ds = LLVIPPaired(LLVIP_ROOT, split='test', max_samples=max_samples)
    elif dataset_name == 'dronevehicle':
        ds = DroneVehiclePaired(dataset_root, split='val', max_samples=max_samples)
    else:
        raise ValueError(f"Unknown retrieval dataset: {dataset_name}")

    loader = get_dataloader(ds, batch_size=batch_size, num_workers=12)
    rgb_feat, ir_feat = extract_paired_features(model, loader, device=device,
                                                max_samples=max_samples)

    rgb2ir = compute_retrieval_metrics(rgb_feat, ir_feat)
    ir2rgb = compute_retrieval_metrics(ir_feat, rgb_feat)

    print(f"    RGB->IR  R@1={rgb2ir['rank_1']:.2f}%  R@5={rgb2ir['rank_5']:.2f}%  mAP={rgb2ir['mAP']:.2f}%")
    print(f"    IR->RGB  R@1={ir2rgb['rank_1']:.2f}%  R@5={ir2rgb['rank_5']:.2f}%  mAP={ir2rgb['mAP']:.2f}%")

    return {'rgb2ir': rgb2ir, 'ir2rgb': ir2rgb}


def run_linear_probe(model, dataset_root, n_last_blocks=4, epochs=100,
                     lr=0.001, batch_size=128, modality_mode='both',
                     device='cuda', is_3ch=False):
    """Run linear probing for one model + one modality."""
    import torch.nn as nn

    embed_dim = model.embed_dim * n_last_blocks
    num_classes = get_num_classes()

    classifier = nn.Linear(embed_dim, num_classes).to(device)
    nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
    nn.init.zeros_(classifier.bias)

    if is_3ch:
        train_ds = DroneVehicleClassification3Ch(dataset_root, split='train')
        val_ds = DroneVehicleClassification3Ch(dataset_root, split='val')
    else:
        train_ds = DroneVehicleClassification(dataset_root, split='train')
        val_ds = DroneVehicleClassification(dataset_root, split='val')

    train_loader = get_dataloader(train_ds, batch_size=batch_size,
                                  num_workers=12, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, num_workers=12)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        classifier.train()
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                if is_3ch:
                    intermediate = model.get_intermediate_layers(imgs, n=n_last_blocks)
                else:
                    imgs = prepare_input(imgs, modality_mode)
                    modality_masks = infer_modality_masks(imgs)
                    intermediate = model.get_intermediate_layers(
                        imgs, n=n_last_blocks, modality_masks=modality_masks)
                feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)

            logits = classifier(feat)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    if is_3ch:
                        intermediate = model.get_intermediate_layers(imgs, n=n_last_blocks)
                    else:
                        imgs = prepare_input(imgs, modality_mode)
                        modality_masks = infer_modality_masks(imgs)
                        intermediate = model.get_intermediate_layers(
                            imgs, n=n_last_blocks, modality_masks=modality_masks)
                    feat = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)

                    preds = classifier(feat).argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            val_acc = correct / total * 100
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"    Epoch {epoch+1}/{epochs}  Val Acc={val_acc:.2f}%  (Best={best_acc:.2f}%)")

    return best_acc


def run_detection(model, dataset_root, n_last_blocks=4, epochs=50,
                  batch_size=64, modality_mode='both', device='cuda'):
    """Run crop-based detection evaluation for one 4ch model."""
    train_ds = DroneVehicleCropDataset(dataset_root, split='train')
    val_ds = DroneVehicleCropDataset(dataset_root, split='val')

    train_loader = get_dataloader(train_ds, batch_size=batch_size, num_workers=12)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, num_workers=12)

    print("    Extracting train crop features...")
    train_feat, train_lab = extract_crop_features(
        model, train_loader, n_last_blocks=n_last_blocks,
        modality_mode=modality_mode, device=device)

    print("    Extracting val crop features...")
    val_feat, val_lab = extract_crop_features(
        model, val_loader, n_last_blocks=n_last_blocks,
        modality_mode=modality_mode, device=device)

    num_classes = get_num_classes()
    result = train_linear_on_crops(train_feat, train_lab, val_feat, val_lab,
                                   num_classes=num_classes, epochs=epochs)
    print(f"    Accuracy={result['accuracy']:.2f}%  mAP={result['mAP']:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DINO-MM Comprehensive Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DINO-MM checkpoint (checkpoint_latest.pth)')
    parser.add_argument('--dino_baseline', type=str, default=None,
                        help='Path to DINO ViT-S/16 ImageNet weights')
    parser.add_argument('--dataset_root', type=str, default=DRONEVEHICLE_ROOT)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(EVAL_DIR, 'outputs', 'comprehensive'))
    # Skip flags
    parser.add_argument('--skip_rankme', action='store_true')
    parser.add_argument('--skip_knn', action='store_true')
    parser.add_argument('--skip_retrieval', action='store_true')
    parser.add_argument('--skip_linear', action='store_true')
    parser.add_argument('--skip_det', action='store_true')
    # Linear probe config
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--linear_lr', type=float, default=0.001)
    parser.add_argument('--linear_batch_size', type=int, default=512)
    parser.add_argument('--linear_n_blocks', type=int, default=4)
    # Detection config
    parser.add_argument('--det_epochs', type=int, default=50)
    # RankMe config
    parser.add_argument('--rankme_samples', type=int, default=2000)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    results = {}  # {model_name: {modality: {metric: value}}}
    t_start = time.time()

    print("=" * 75)
    print("DINO-MM Comprehensive Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  DINO baseline: {args.dino_baseline or 'N/A'}")
    print(f"  Device: {device}")
    print(f"  Skip: rankme={args.skip_rankme} knn={args.skip_knn} "
          f"retrieval={args.skip_retrieval} linear={args.skip_linear} det={args.skip_det}")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("\n[1/6] Loading models...")

    print("\n  --- Random Init (ViT-S/16, 4ch) ---")
    random_model = build_random_model(device=device)

    dino_baseline = None
    if args.dino_baseline and os.path.isfile(args.dino_baseline):
        print("\n  --- DINO ViT-S/16 (ImageNet, 3ch) ---")
        dino_baseline = build_dino_baseline(args.dino_baseline, device=device)

    print("\n  --- DINO-MM (Ours, 4ch) ---")
    dinomm_model, dinomm_args = build_model(args.checkpoint, device=device)

    # ------------------------------------------------------------------
    # 2. RankMe
    # ------------------------------------------------------------------
    if not args.skip_rankme:
        print("\n" + "=" * 75)
        print("[2/6] RankMe Evaluation")
        print("=" * 75)

        print("\n  Random Init (RGB+IR):")
        random_rankme = run_rankme(random_model, args.dataset_root,
                                   num_samples=args.rankme_samples,
                                   batch_size=args.batch_size, device=device)
        results.setdefault('Random Init', {}).setdefault('RGB+IR', {})['rankme'] = round(random_rankme, 1)
        print(f"    RankMe = {random_rankme:.1f}")

        if dino_baseline is not None:
            print("\n  DINO ViT-S/16 (RGB):")
            dino_rankme = run_rankme(dino_baseline, args.dataset_root,
                                     num_samples=args.rankme_samples,
                                     batch_size=args.batch_size, device=device,
                                     is_3ch=True)
            results.setdefault('DINO ViT-S/16', {}).setdefault('RGB', {})['rankme'] = round(dino_rankme, 1)
            print(f"    RankMe = {dino_rankme:.1f}")

        print("\n  DINO-MM (RGB+IR):")
        dinomm_rankme = run_rankme(dinomm_model, args.dataset_root,
                                   num_samples=args.rankme_samples,
                                   batch_size=args.batch_size, device=device)
        results.setdefault('DINO-MM (Ours)', {}).setdefault('RGB+IR', {})['rankme'] = round(dinomm_rankme, 1)
        print(f"    RankMe = {dinomm_rankme:.1f}")
    else:
        print("\n[2/6] RankMe — SKIPPED")

    # ------------------------------------------------------------------
    # 3. KNN Classification
    # ------------------------------------------------------------------
    if not args.skip_knn:
        print("\n" + "=" * 75)
        print("[3/6] KNN Classification")
        print("=" * 75)

        print("\n  Random Init (RGB+IR):")
        random_knn = run_knn(random_model, args.dataset_root,
                             batch_size=args.batch_size, device=device,
                             modality_mode='both')
        for k, acc in random_knn.items():
            results.setdefault('Random Init', {}).setdefault('RGB+IR', {})[f'knn@{k}'] = round(acc, 2)

        if dino_baseline is not None:
            print("\n  DINO ViT-S/16 (RGB):")
            dino_knn = run_knn(dino_baseline, args.dataset_root,
                               batch_size=args.batch_size, device=device,
                               is_3ch=True)
            for k, acc in dino_knn.items():
                results.setdefault('DINO ViT-S/16', {}).setdefault('RGB', {})[f'knn@{k}'] = round(acc, 2)

        # DINO-MM: all three modalities
        for mode, label in [('both', 'RGB+IR'), ('rgb_only', 'RGB'), ('ir_only', 'IR')]:
            print(f"\n  DINO-MM ({label}):")
            mm_knn = run_knn(dinomm_model, args.dataset_root,
                             batch_size=args.batch_size, device=device,
                             modality_mode=mode)
            for k, acc in mm_knn.items():
                results.setdefault('DINO-MM (Ours)', {}).setdefault(label, {})[f'knn@{k}'] = round(acc, 2)
    else:
        print("\n[3/6] KNN Classification — SKIPPED")

    # ------------------------------------------------------------------
    # 4. Cross-Modal Retrieval (4ch models only)
    # ------------------------------------------------------------------
    if not args.skip_retrieval:
        print("\n" + "=" * 75)
        print("[4/6] Cross-Modal Retrieval")
        print("=" * 75)

        for ds_name, ds_label in [('dronevehicle', 'DroneVehicle'), ('llvip', 'LLVIP')]:
            print(f"\n  --- {ds_label} ---")

            print(f"\n  Random Init:")
            random_ret = run_retrieval(random_model, ds_name, args.dataset_root,
                                       batch_size=args.batch_size, device=device)
            r1_key = f'R@1_{ds_label}'
            avg_r1 = (random_ret['rgb2ir']['rank_1'] + random_ret['ir2rgb']['rank_1']) / 2
            results.setdefault('Random Init', {}).setdefault('RGB+IR', {})[r1_key] = round(avg_r1, 2)

            print(f"\n  DINO-MM:")
            mm_ret = run_retrieval(dinomm_model, ds_name, args.dataset_root,
                                   batch_size=args.batch_size, device=device)
            avg_r1 = (mm_ret['rgb2ir']['rank_1'] + mm_ret['ir2rgb']['rank_1']) / 2
            results.setdefault('DINO-MM (Ours)', {}).setdefault('RGB+IR', {})[r1_key] = round(avg_r1, 2)

            # Store full retrieval results
            results.setdefault('DINO-MM (Ours)', {}).setdefault('RGB+IR', {})[f'retrieval_{ds_label}'] = {
                'rgb2ir': {k: round(v, 2) for k, v in mm_ret['rgb2ir'].items()},
                'ir2rgb': {k: round(v, 2) for k, v in mm_ret['ir2rgb'].items()},
            }
    else:
        print("\n[4/6] Cross-Modal Retrieval — SKIPPED")

    # ------------------------------------------------------------------
    # 5. Linear Probe
    # ------------------------------------------------------------------
    if not args.skip_linear:
        print("\n" + "=" * 75)
        print("[5/6] Linear Probe")
        print("=" * 75)

        print("\n  Random Init (RGB+IR):")
        random_lp = run_linear_probe(
            random_model, args.dataset_root,
            n_last_blocks=args.linear_n_blocks, epochs=args.linear_epochs,
            lr=args.linear_lr, batch_size=args.linear_batch_size,
            modality_mode='both', device=device)
        results.setdefault('Random Init', {}).setdefault('RGB+IR', {})['linear_probe'] = round(random_lp, 2)

        if dino_baseline is not None:
            print("\n  DINO ViT-S/16 (RGB):")
            dino_lp = run_linear_probe(
                dino_baseline, args.dataset_root,
                n_last_blocks=args.linear_n_blocks, epochs=args.linear_epochs,
                lr=args.linear_lr, batch_size=args.linear_batch_size,
                device=device, is_3ch=True)
            results.setdefault('DINO ViT-S/16', {}).setdefault('RGB', {})['linear_probe'] = round(dino_lp, 2)

        for mode, label in [('both', 'RGB+IR'), ('rgb_only', 'RGB'), ('ir_only', 'IR')]:
            print(f"\n  DINO-MM ({label}):")
            mm_lp = run_linear_probe(
                dinomm_model, args.dataset_root,
                n_last_blocks=args.linear_n_blocks, epochs=args.linear_epochs,
                lr=args.linear_lr, batch_size=args.linear_batch_size,
                modality_mode=mode, device=device)
            results.setdefault('DINO-MM (Ours)', {}).setdefault(label, {})['linear_probe'] = round(mm_lp, 2)
    else:
        print("\n[5/6] Linear Probe — SKIPPED")

    # ------------------------------------------------------------------
    # 6. Detection (4ch models only)
    # ------------------------------------------------------------------
    if not args.skip_det:
        print("\n" + "=" * 75)
        print("[6/6] Detection (Crop-based)")
        print("=" * 75)

        print("\n  Random Init (RGB+IR):")
        random_det = run_detection(random_model, args.dataset_root,
                                   epochs=args.det_epochs,
                                   batch_size=args.batch_size, device=device)
        results.setdefault('Random Init', {}).setdefault('RGB+IR', {})['det_acc'] = round(random_det['accuracy'], 2)

        print("\n  DINO-MM (RGB+IR):")
        mm_det = run_detection(dinomm_model, args.dataset_root,
                               epochs=args.det_epochs,
                               batch_size=args.batch_size, device=device)
        results.setdefault('DINO-MM (Ours)', {}).setdefault('RGB+IR', {})['det_acc'] = round(mm_det['accuracy'], 2)
    else:
        print("\n[6/6] Detection — SKIPPED")

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 100)

    # Helper to get metric or '-'
    def g(model_name, modality, metric):
        return results.get(model_name, {}).get(modality, {}).get(metric, None)

    def fmt(val, suffix='%'):
        if val is None:
            return '  -   '
        if suffix == '%':
            return f'{val:6.2f}'
        return f'{val:6.1f}'

    header = (f"{'Model':<20} | {'Modality':<8} | {'RankMe':>7} | {'KNN@10':>7} | "
              f"{'KNN@20':>7} | {'LinProbe':>8} | {'Det':>7} | {'R@1_DV':>7} | {'R@1_LL':>7}")
    print(header)
    print("-" * len(header))

    # Define rows: (model_name, modality, is_retrieval_capable)
    rows = [
        ('Random Init',     'RGB+IR', True),
        ('DINO ViT-S/16',   'RGB',    False),
        ('DINO-MM (Ours)',   'RGB+IR', True),
        ('DINO-MM (Ours)',   'RGB',    False),
        ('DINO-MM (Ours)',   'IR',     False),
    ]

    for model_name, modality, has_retrieval in rows:
        mod_results = results.get(model_name, {}).get(modality, {})
        if not mod_results:
            continue

        rankme = fmt(mod_results.get('rankme'), suffix='')
        knn10 = fmt(mod_results.get('knn@10'))
        knn20 = fmt(mod_results.get('knn@20'))
        lp = fmt(mod_results.get('linear_probe'))
        det = fmt(mod_results.get('det_acc')) if has_retrieval else '  N/A '
        r1_dv = fmt(mod_results.get('R@1_DroneVehicle')) if has_retrieval else '  N/A '
        r1_ll = fmt(mod_results.get('R@1_LLVIP')) if has_retrieval else '  N/A '

        print(f"{model_name:<20} | {modality:<8} | {rankme:>7} | {knn10:>7} | "
              f"{knn20:>7} | {lp:>8} | {det:>7} | {r1_dv:>7} | {r1_ll:>7}")

    print("=" * len(header))
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    # Clean results for JSON serialization (remove numpy arrays etc)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    output = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'dino_baseline': args.dino_baseline,
        'elapsed_minutes': round(elapsed / 60, 1),
        'results': clean_for_json(results),
    }

    json_path = os.path.join(args.output_dir, 'comprehensive_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()

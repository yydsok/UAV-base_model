#!/usr/bin/env python3
"""
全量预训练 Manifest 生成器
==========================
扫描 /root/autodl-tmp/data/ 中所有 RGB 和 IR 数据集，
进行质量清洗，生成统一的全量 manifest 用于无标签自监督预训练。

处理内容:
  Phase 1: 重用已清洗的配对数据集 manifest
  Phase 2: 生成缺失的 manifest (WebUAV-3M, VisDrone-SOT, VisDrone-CC, TarDAL-roadscene, TarDAL-Fusion, UTUAV, M3OT-1)
  Phase 3: 对 IR-only 数据集做质量清洗 (LSOTB-TIR, BIRDSAI, MONET, HIT-UAV)
  Phase 4: 对 RGB-only 数据集做质量清洗
  Phase 5: 合并为一个全量 manifest
"""

import os
import sys
import json
import glob
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict
import time
import traceback

# ============================================================
# 配置
# ============================================================
DATA_ROOT = '/root/autodl-tmp/data'
TRAIN1_ROOT = '/root/autodl-tmp/train1'
MANIFEST_DIR = os.path.join(TRAIN1_ROOT, 'manifests')
CONVERTED_IR = os.path.join(TRAIN1_ROOT, 'converted_ir')

# 质量检测阈值
MEAN_LOW = 5.0
MEAN_HIGH = 250.0
STD_LOW = 10.0

# 图像文件扩展名
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}

NUM_WORKERS = min(cpu_count(), 16)


# ============================================================
# 工具函数
# ============================================================
def is_image(path):
    return os.path.splitext(path)[1] in IMG_EXTS


def find_images(directory, recursive=True):
    """查找目录下所有图像文件"""
    images = []
    if not os.path.isdir(directory):
        return images
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in sorted(files):
                if os.path.splitext(f)[1] in IMG_EXTS:
                    images.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(directory)):
            fp = os.path.join(directory, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1] in IMG_EXTS:
                images.append(fp)
    return images


def check_image_quality(img_path):
    """检查单张图像质量，返回 (path, is_clean, anomalies)"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return (img_path, False, ['corrupt'])

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray.dtype == np.uint16:
            gmin, gmax = float(gray.min()), float(gray.max())
            if gmax == gmin:
                return (img_path, False, ['constant_16bit'])
            gray = ((gray.astype(np.float32) - gmin) / (gmax - gmin) * 255).astype(np.uint8)

        mean_val = float(gray.mean())
        std_val = float(gray.std())

        anomalies = []
        if mean_val < MEAN_LOW:
            anomalies.append(f'all_black(mean={mean_val:.1f})')
        if mean_val > MEAN_HIGH:
            anomalies.append(f'all_white(mean={mean_val:.1f})')
        if std_val < STD_LOW:
            anomalies.append(f'low_contrast(std={std_val:.1f})')

        return (img_path, len(anomalies) == 0, anomalies)
    except Exception as e:
        return (img_path, False, [f'error({str(e)[:80]})'])


def batch_quality_check(image_paths, desc="Checking", num_workers=None):
    """批量质量检查，使用多进程"""
    if not image_paths:
        return [], [], {}
    if num_workers is None:
        num_workers = NUM_WORKERS

    clean_paths = []
    excluded = []
    anomaly_counts = defaultdict(int)
    total = len(image_paths)

    print(f"  [{desc}] 开始质量检查: {total} 张图像, {num_workers} 进程...")

    with Pool(num_workers) as pool:
        for i, (path, is_clean, anomalies) in enumerate(
            pool.imap_unordered(check_image_quality, image_paths, chunksize=200)
        ):
            if is_clean:
                clean_paths.append(path)
            else:
                excluded.append((path, anomalies))
                for a in anomalies:
                    key = a.split('(')[0]
                    anomaly_counts[key] += 1
            if (i + 1) % 50000 == 0:
                print(f"    进度: {i+1}/{total}, 已排除: {len(excluded)}")

    print(f"  [{desc}] 完成: {len(clean_paths)} 通过, {len(excluded)} 排除")
    if anomaly_counts:
        for k, v in sorted(anomaly_counts.items()):
            print(f"    - {k}: {v}")

    return clean_paths, excluded, dict(anomaly_counts)


def write_manifest(filepath, dataset, samples, total_scanned, total_excluded, anomaly_counts):
    """写入 manifest JSON"""
    manifest = {
        'dataset': dataset,
        'generated_by': 'generate_full_manifest.py',
        'total_scanned': total_scanned,
        'total_clean': len(samples),
        'total_excluded': total_excluded,
        'anomaly_counts': anomaly_counts,
        'samples': samples,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(manifest, f, ensure_ascii=False)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  已写入: {filepath} ({size_mb:.1f} MB, {len(samples)} 样本)")
    return manifest


def load_existing_manifest(filepath):
    """加载已有 manifest 的 samples"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    samples = data.get('samples', data.get('data', []))
    return samples


# ============================================================
# Phase 1: 重用已清洗的配对数据集
# ============================================================
def phase1_reuse_clean_paired():
    """加载已经正确清洗的配对数据集 manifest"""
    print("\n" + "=" * 70)
    print("Phase 1: 加载已清洗的配对数据集 manifest")
    print("=" * 70)

    # 这些 manifest 已经经过正确的质量检查和 IR 转换
    paired_manifests = [
        'RGBT-Tiny_train.json',
        'RGBT-Tiny_test.json',
        'VT-Tiny-MOT_train.json',
        'VT-Tiny-MOT_test.json',
        'LLVIP_train.json',
        'LLVIP_test.json',
        'LLVIP_unreg_all.json',
        'DroneVehicle_train.json',
        'DroneVehicle_val.json',
        'DroneVehicle_test.json',
        'M3OT_train.json',
        'M3OT_val.json',
        'M3OT_test.json',
        'TarDAL_M3FD.json',
        'TarDAL_tno.json',
        'DroneRGBT_train.json',
        'DroneRGBT_test.json',
    ]

    all_samples = []
    for mf in paired_manifests:
        path = os.path.join(MANIFEST_DIR, mf)
        if os.path.exists(path):
            samples = load_existing_manifest(path)
            print(f"  {mf}: {len(samples)} 样本")
            all_samples.extend(samples)
        else:
            print(f"  [警告] {mf} 不存在!")

    print(f"  Phase 1 总计: {len(all_samples)} 样本 (已清洗配对数据)")
    return all_samples


# ============================================================
# Phase 2: 生成缺失的 manifest
# ============================================================
def gen_utuav():
    """UTUAV: 重新生成损坏的 manifest (配对, 仅路径扫描)"""
    print("\n--- UTUAV (配对, ~1.66M, 仅路径扫描) ---")
    base = os.path.join(DATA_ROOT, 'UTUAV')
    if not os.path.isdir(base):
        print("  [错误] UTUAV 目录不存在!")
        return []

    sequences = sorted([d for d in os.listdir(base)
                        if os.path.isdir(os.path.join(base, d))])
    print(f"  发现 {len(sequences)} 个序列")

    samples = []
    no_match = 0
    for seq in sequences:
        rgb_dir = os.path.join(base, seq, 'rgb')
        ir_dir = os.path.join(base, seq, 'ir')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
            continue

        rgb_files = {f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS}
        ir_files = {f for f in os.listdir(ir_dir) if os.path.splitext(f)[1] in IMG_EXTS}
        common = sorted(rgb_files & ir_files)

        for f in common:
            samples.append({
                'rgb_path': os.path.join(rgb_dir, f),
                'ir_path': os.path.join(ir_dir, f),
                'type': 'paired',
                'dataset': 'UTUAV',
                'sequence': seq,
                'ir_is_pseudo_rgb': True,
            })

        no_match += len(rgb_files - ir_files) + len(ir_files - rgb_files)

    print(f"  UTUAV: {len(samples)} 配对样本, {no_match} 无法配对")

    # 写入独立 manifest
    write_manifest(
        os.path.join(MANIFEST_DIR, 'UTUAV_all_v2.json'),
        'UTUAV', samples, len(samples) + no_match, no_match, {}
    )
    return samples


def gen_webuav3m():
    """WebUAV-3M: 生成新 manifest (RGB-only, 仅路径扫描, 数据量太大)"""
    print("\n--- WebUAV-3M (RGB-only, ~1.59M, 仅路径扫描) ---")
    base = os.path.join(DATA_ROOT, 'WebUAV-3M')
    if not os.path.isdir(base):
        print("  [错误] WebUAV-3M 目录不存在!")
        return []

    samples = []
    for split in ['Train', 'Val', 'Test']:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        sequences = sorted(os.listdir(split_dir))
        split_count = 0
        for seq in sequences:
            img_dir = os.path.join(split_dir, seq, 'img')
            if not os.path.isdir(img_dir):
                continue
            for f in sorted(os.listdir(img_dir)):
                if os.path.splitext(f)[1] in IMG_EXTS:
                    samples.append({
                        'rgb_path': os.path.join(img_dir, f),
                        'ir_path': None,
                        'type': 'rgb_only',
                        'dataset': 'WebUAV-3M',
                        'sequence': seq,
                    })
                    split_count += 1
        print(f"  {split}: {split_count} 张, {len(sequences)} 序列")

    write_manifest(
        os.path.join(MANIFEST_DIR, 'WebUAV-3M_all.json'),
        'WebUAV-3M', samples, len(samples), 0, {}
    )
    return samples


def gen_visdrone_sot():
    """VisDrone-SOT: 生成新 manifest (RGB-only, 带质量检查)"""
    print("\n--- VisDrone-SOT (RGB-only, ~222K, 带质量检查) ---")
    base = os.path.join(DATA_ROOT, 'VisDrone', 'Single-Object Tracking')
    if not os.path.isdir(base):
        print("  [错误] VisDrone SOT 目录不存在!")
        return []

    all_images = []
    for part_dir in sorted(os.listdir(base)):
        part_path = os.path.join(base, part_dir)
        if not os.path.isdir(part_path):
            continue
        if 'initialization' in part_dir:
            continue
        images = find_images(part_path, recursive=True)
        # 排除 annotations 目录下的图
        images = [p for p in images if '/annotations/' not in p and '/attributes/' not in p]
        print(f"  {part_dir}: {len(images)} 张")
        all_images.extend(images)

    clean_paths, excluded, anomaly_counts = batch_quality_check(all_images, "VisDrone-SOT")

    samples = [{
        'rgb_path': p,
        'ir_path': None,
        'type': 'rgb_only',
        'dataset': 'VisDrone-SOT',
    } for p in clean_paths]

    write_manifest(
        os.path.join(MANIFEST_DIR, 'VisDrone-SOT_all.json'),
        'VisDrone-SOT', samples, len(all_images), len(excluded), anomaly_counts
    )
    return samples


def gen_visdrone_cc():
    """VisDrone-CC: 生成新 manifest (RGB-only, 带质量检查)"""
    print("\n--- VisDrone-CC (RGB-only, ~3.4K, 带质量检查) ---")
    base = os.path.join(DATA_ROOT, 'VisDrone', 'Crowd Counting')
    if not os.path.isdir(base):
        print("  [错误] VisDrone CC 目录不存在!")
        return []

    seq_dir = os.path.join(base, 'sequences')
    all_images = find_images(seq_dir, recursive=True)
    print(f"  发现: {len(all_images)} 张")

    clean_paths, excluded, anomaly_counts = batch_quality_check(all_images, "VisDrone-CC")

    samples = [{
        'rgb_path': p,
        'ir_path': None,
        'type': 'rgb_only',
        'dataset': 'VisDrone-CC',
    } for p in clean_paths]

    write_manifest(
        os.path.join(MANIFEST_DIR, 'VisDrone-CC_all.json'),
        'VisDrone-CC', samples, len(all_images), len(excluded), anomaly_counts
    )
    return samples


def gen_tardal_roadscene():
    """TarDAL-roadscene: 生成新 manifest (配对, 带质量检查)"""
    print("\n--- TarDAL-roadscene (配对, 42对, 带质量检查) ---")
    vi_dir = os.path.join(DATA_ROOT, 'TarDAL', 'roadscene', 'vi')
    ir_dir = os.path.join(DATA_ROOT, 'TarDAL', 'roadscene', 'ir')

    if not os.path.isdir(vi_dir) or not os.path.isdir(ir_dir):
        print("  [错误] TarDAL roadscene 目录不存在!")
        return []

    vi_files = {os.path.splitext(f)[0]: f for f in os.listdir(vi_dir) if os.path.splitext(f)[1] in IMG_EXTS}
    ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(ir_dir) if os.path.splitext(f)[1] in IMG_EXTS}
    common = sorted(vi_files.keys() & ir_files.keys())

    # 质量检查
    all_imgs = [os.path.join(vi_dir, vi_files[k]) for k in common] + \
               [os.path.join(ir_dir, ir_files[k]) for k in common]
    _, excluded_list, _ = batch_quality_check(all_imgs, "TarDAL-roadscene")
    bad_paths = {e[0] for e in excluded_list}

    samples = []
    excluded_count = 0
    for k in common:
        vp = os.path.join(vi_dir, vi_files[k])
        ip = os.path.join(ir_dir, ir_files[k])
        if vp in bad_paths or ip in bad_paths:
            excluded_count += 1
            continue
        samples.append({
            'rgb_path': vp,
            'ir_path': ip,
            'type': 'paired',
            'dataset': 'TarDAL-roadscene',
        })

    write_manifest(
        os.path.join(MANIFEST_DIR, 'TarDAL_roadscene.json'),
        'TarDAL-roadscene', samples, len(common), excluded_count, {}
    )
    return samples


def gen_tardal_fusion():
    """TarDAL M3FD_Fusion: 生成新 manifest (配对, 300对, 带质量检查)"""
    print("\n--- TarDAL-M3FD-Fusion (配对, 300对, 带质量检查) ---")
    vis_dir = os.path.join(DATA_ROOT, 'TarDAL', 'M3FD_Fusion', 'Vis')
    ir_dir = os.path.join(DATA_ROOT, 'TarDAL', 'M3FD_Fusion', 'Ir')

    if not os.path.isdir(vis_dir) or not os.path.isdir(ir_dir):
        print("  [错误] TarDAL M3FD_Fusion 目录不存在!")
        return []

    vis_files = {os.path.splitext(f)[0]: f for f in os.listdir(vis_dir) if os.path.splitext(f)[1] in IMG_EXTS}
    ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(ir_dir) if os.path.splitext(f)[1] in IMG_EXTS}
    common = sorted(vis_files.keys() & ir_files.keys())

    all_imgs = [os.path.join(vis_dir, vis_files[k]) for k in common] + \
               [os.path.join(ir_dir, ir_files[k]) for k in common]
    _, excluded_list, _ = batch_quality_check(all_imgs, "TarDAL-Fusion")
    bad_paths = {e[0] for e in excluded_list}

    samples = []
    excluded_count = 0
    for k in common:
        vp = os.path.join(vis_dir, vis_files[k])
        ip = os.path.join(ir_dir, ir_files[k])
        if vp in bad_paths or ip in bad_paths:
            excluded_count += 1
            continue
        samples.append({
            'rgb_path': vp,
            'ir_path': ip,
            'type': 'paired',
            'dataset': 'TarDAL-M3FD-Fusion',
        })

    write_manifest(
        os.path.join(MANIFEST_DIR, 'TarDAL_M3FD_Fusion.json'),
        'TarDAL-M3FD-Fusion', samples, len(common), excluded_count, {}
    )
    return samples


def gen_m3ot_1_ir():
    """M3OT subdir 1: IR-only, 418 张"""
    print("\n--- M3OT-1 (IR-only, 418张, 带质量检查) ---")
    ir_dir = os.path.join(DATA_ROOT, 'M3OT', '1', 'ir')
    if not os.path.isdir(ir_dir):
        print("  [错误] M3OT/1/ir 不存在!")
        return []

    all_images = find_images(ir_dir, recursive=True)
    print(f"  发现: {len(all_images)} 张")

    clean_paths, excluded, anomaly_counts = batch_quality_check(all_images, "M3OT-1")

    samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': 'M3OT-1-IR',
    } for p in clean_paths]

    write_manifest(
        os.path.join(MANIFEST_DIR, 'M3OT_1_ir.json'),
        'M3OT-1-IR', samples, len(all_images), len(excluded), anomaly_counts
    )
    return samples


def phase2_generate_missing():
    """生成所有缺失的 manifest"""
    print("\n" + "=" * 70)
    print("Phase 2: 生成缺失的 manifest")
    print("=" * 70)

    all_samples = []

    # UTUAV - 重新生成 (损坏的 manifest)
    all_samples.extend(gen_utuav())

    # WebUAV-3M - 全新生成
    all_samples.extend(gen_webuav3m())

    # VisDrone-SOT - 全新生成
    all_samples.extend(gen_visdrone_sot())

    # VisDrone-CC - 全新生成
    all_samples.extend(gen_visdrone_cc())

    # TarDAL-roadscene - 全新生成
    all_samples.extend(gen_tardal_roadscene())

    # TarDAL-M3FD-Fusion - 全新生成
    all_samples.extend(gen_tardal_fusion())

    # M3OT subdir 1 IR-only
    all_samples.extend(gen_m3ot_1_ir())

    print(f"\n  Phase 2 总计: {len(all_samples)} 样本 (新生成)")
    return all_samples


# ============================================================
# Phase 3: 质量清洗 IR-only 数据集
# ============================================================
def clean_ir_dataset(name, base_dir, recursive=True):
    """对 IR-only 数据集进行质量检查"""
    print(f"\n--- {name} (IR-only, 质量清洗) ---")
    if not os.path.isdir(base_dir):
        print(f"  [错误] {base_dir} 不存在!")
        return []

    all_images = find_images(base_dir, recursive=recursive)
    # 排除 annotations 等非图像目录
    all_images = [p for p in all_images
                  if '/anno/' not in p
                  and '/annotations/' not in p
                  and '/Annotations/' not in p
                  and '/gt/' not in p]
    print(f"  发现: {len(all_images)} 张 IR 图像")

    clean_paths, excluded, anomaly_counts = batch_quality_check(all_images, name)

    samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': name,
    } for p in clean_paths]

    manifest_name = name.replace('-', '_').replace(' ', '_') + '_clean.json'
    write_manifest(
        os.path.join(MANIFEST_DIR, manifest_name),
        name, samples, len(all_images), len(excluded), anomaly_counts
    )
    return samples


def phase3_clean_ir_only():
    """质量清洗所有 IR-only 数据集"""
    print("\n" + "=" * 70)
    print("Phase 3: 质量清洗 IR-only 数据集")
    print("=" * 70)

    all_samples = []

    # LSOTB-TIR
    lsotb_dirs = [
        os.path.join(DATA_ROOT, 'LSOTB-TIR', 'Training Dataset', 'TrainingData'),
        os.path.join(DATA_ROOT, 'LSOTB-TIR', 'Evaluation Dataset'),
    ]
    lsotb_images = []
    for d in lsotb_dirs:
        if os.path.isdir(d):
            imgs = find_images(d, recursive=True)
            imgs = [p for p in imgs if '/anno/' not in p and '/Annotations/' not in p
                    and '/Evaluation Results/' not in p]
            lsotb_images.extend(imgs)
    print(f"\n--- LSOTB-TIR (IR-only, 质量清洗) ---")
    print(f"  发现: {len(lsotb_images)} 张 IR 图像")
    clean_paths, excluded, anomaly_counts = batch_quality_check(lsotb_images, "LSOTB-TIR")
    lsotb_samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': 'LSOTB-TIR',
    } for p in clean_paths]
    write_manifest(
        os.path.join(MANIFEST_DIR, 'LSOTB_TIR_clean.json'),
        'LSOTB-TIR', lsotb_samples, len(lsotb_images), len(excluded), anomaly_counts
    )
    all_samples.extend(lsotb_samples)

    # BIRDSAI
    birdsai_dirs = [
        os.path.join(DATA_ROOT, 'BIRDSAI', 'TrainReal', 'images'),
        os.path.join(DATA_ROOT, 'BIRDSAI', 'TestReal', 'images'),
    ]
    birdsai_images = []
    for d in birdsai_dirs:
        if os.path.isdir(d):
            birdsai_images.extend(find_images(d, recursive=True))
    print(f"\n--- BIRDSAI (IR-only, 质量清洗) ---")
    print(f"  发现: {len(birdsai_images)} 张 IR 图像")
    clean_paths, excluded, anomaly_counts = batch_quality_check(birdsai_images, "BIRDSAI")
    birdsai_samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': 'BIRDSAI',
    } for p in clean_paths]
    write_manifest(
        os.path.join(MANIFEST_DIR, 'BIRDSAI_clean.json'),
        'BIRDSAI', birdsai_samples, len(birdsai_images), len(excluded), anomaly_counts
    )
    all_samples.extend(birdsai_samples)

    # MONET
    monet_base = os.path.join(DATA_ROOT, 'MONET')
    monet_images = []
    for split in ['monet_train', 'monet_val', 'monet_test']:
        d = os.path.join(monet_base, split)
        if os.path.isdir(d):
            imgs = find_images(d, recursive=True)
            # 只要 lwir 目录下的图
            imgs = [p for p in imgs if '/lwir/' in p or '/LWIR/' in p
                    or split in ['monet_val', 'monet_test']]
            # 排除 annotations
            imgs = [p for p in imgs if '/annotations/' not in p]
            monet_images.extend(imgs)
    print(f"\n--- MONET (IR-only, 质量清洗) ---")
    print(f"  发现: {len(monet_images)} 张 IR 图像")
    clean_paths, excluded, anomaly_counts = batch_quality_check(monet_images, "MONET")
    monet_samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': 'MONET',
    } for p in clean_paths]
    write_manifest(
        os.path.join(MANIFEST_DIR, 'MONET_clean.json'),
        'MONET', monet_samples, len(monet_images), len(excluded), anomaly_counts
    )
    all_samples.extend(monet_samples)

    # HIT-UAV
    hituav_base = os.path.join(DATA_ROOT, 'HIT-UAV')
    hituav_images = []
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(hituav_base, split, 'img')
        if os.path.isdir(img_dir):
            hituav_images.extend(find_images(img_dir, recursive=False))
    print(f"\n--- HIT-UAV (IR-only, 质量清洗) ---")
    print(f"  发现: {len(hituav_images)} 张 IR 图像")
    clean_paths, excluded, anomaly_counts = batch_quality_check(hituav_images, "HIT-UAV")
    hituav_samples = [{
        'rgb_path': None,
        'ir_path': p,
        'type': 'ir_only',
        'dataset': 'HIT-UAV',
    } for p in clean_paths]
    write_manifest(
        os.path.join(MANIFEST_DIR, 'HIT_UAV_clean.json'),
        'HIT-UAV', hituav_samples, len(hituav_images), len(excluded), anomaly_counts
    )
    all_samples.extend(hituav_samples)

    print(f"\n  Phase 3 总计: {len(all_samples)} 样本 (IR-only 清洗后)")
    return all_samples


# ============================================================
# Phase 4: 质量清洗 RGB-only 数据集
# ============================================================
def clean_rgb_dataset(name, directories, recursive=True, exclude_patterns=None):
    """对 RGB-only 数据集进行质量检查"""
    print(f"\n--- {name} (RGB-only, 质量清洗) ---")
    if exclude_patterns is None:
        exclude_patterns = ['/annotations/', '/labels/', '/gt/', '/anno/',
                            '/mask/', '/Annotations/', '/Labels/',
                            '/SegmentationClass/', '/ground_truth/']

    all_images = []
    for d in directories:
        if os.path.isdir(d):
            imgs = find_images(d, recursive=recursive)
            for pat in exclude_patterns:
                imgs = [p for p in imgs if pat not in p]
            all_images.extend(imgs)

    if not all_images:
        print(f"  [警告] {name}: 没有找到图像!")
        return []

    print(f"  发现: {len(all_images)} 张 RGB 图像")
    clean_paths, excluded, anomaly_counts = batch_quality_check(all_images, name)

    samples = [{
        'rgb_path': p,
        'ir_path': None,
        'type': 'rgb_only',
        'dataset': name,
    } for p in clean_paths]

    manifest_name = name.replace('-', '_').replace(' ', '_') + '_clean.json'
    write_manifest(
        os.path.join(MANIFEST_DIR, manifest_name),
        name, samples, len(all_images), len(excluded), anomaly_counts
    )
    return samples


def phase4_clean_rgb_only():
    """质量清洗所有 RGB-only 数据集"""
    print("\n" + "=" * 70)
    print("Phase 4: 质量清洗 RGB-only 数据集")
    print("=" * 70)

    all_samples = []

    # 定义所有 RGB-only 数据集及其路径
    rgb_datasets = [
        ('AU-AIR', [os.path.join(DATA_ROOT, 'AU-AIR')],
         True, ['/annotations/', '/auair2019annotations/']),

        ('AeroScapes', [os.path.join(DATA_ROOT, 'AeroScapes', 'JPEGImages')],
         False, ['/SegmentationClass/']),

        ('AnimalDrone', [
            os.path.join(DATA_ROOT, 'AnimalDrone', 'AnimalDrone_PartA', 'jpg-train', 'jpg'),
            os.path.join(DATA_ROOT, 'AnimalDrone', 'AnimalDrone_PartA', 'jpg-test', 'jpg'),
            os.path.join(DATA_ROOT, 'AnimalDrone', 'AnimalDrone_PartB'),
        ], True, ['/mat-train/', '/mat-test/', '/annotation/', '/mat/']),

        ('DroneCrowd', [
            os.path.join(DATA_ROOT, 'DroneCrowd', 'train_data', 'images'),
            os.path.join(DATA_ROOT, 'DroneCrowd', 'val_data', 'images'),
            os.path.join(DATA_ROOT, 'DroneCrowd', 'test_data', 'images'),
        ], False, []),

        ('EVD4UAV', [os.path.join(DATA_ROOT, 'EVD4UAV', 'images')],
         False, ['/bb/', '/rotated_bb/']),

        ('UAV123', [os.path.join(DATA_ROOT, 'UAV123', 'data_seq')],
         True, ['/anno/']),

        ('UAVDT', [
            os.path.join(DATA_ROOT, 'UAVDT', 'UAV-benchmark-M'),
            os.path.join(DATA_ROOT, 'UAVDT', 'UAV-benchmark-S'),
        ], True, ['/M_attr/', '/UAV-benchmark-MOTD/', '/GT/', '/gt/']),

        ('UAVScenes', [os.path.join(DATA_ROOT, 'UAVScenes')],
         True, ['/LIDAR/', '/label/', '/interval=1/']),

        ('UAVid', [os.path.join(DATA_ROOT, 'UAVid')],
         True, ['/Labels/']),

        ('UDD', [os.path.join(DATA_ROOT, 'UDD')],
         True, ['/gt/', '/metadata/']),

        ('UAV-Human', [os.path.join(DATA_ROOT, 'UAV-Human', 'PoseEstimation', 'frames')],
         True, ['/Skeleton/', '/annotations/']),

        ('University-1652', [
            os.path.join(DATA_ROOT, 'University-1652', 'train', 'drone'),
            os.path.join(DATA_ROOT, 'University-1652', 'test', 'gallery_drone'),
            os.path.join(DATA_ROOT, 'University-1652', 'test', 'query_drone'),
            os.path.join(DATA_ROOT, 'University-1652', 'test', '4K_drone'),
        ], True, []),

        ('SUES-200', [os.path.join(DATA_ROOT, 'SUES-200', 'drone_view_512')],
         True, []),

        ('CVOGL', [os.path.join(DATA_ROOT, 'CVOGL', 'CVOGL_DroneAerial', 'query')],
         False, []),

        ('DTB70', [os.path.join(DATA_ROOT, 'DTB70')],
         True, ['/groundtruth_rect.txt']),

        ('Manipal-UAV', [os.path.join(DATA_ROOT, 'Manipal-UAV')],
         True, ['/labels/']),

        ('MDMT', [os.path.join(DATA_ROOT, 'MDMT')],
         True, ['/gt/', '/det/']),

        ('SeaDronesSee', [os.path.join(DATA_ROOT, 'SeaDronesSee')],
         True, ['/annotations/', '/mods/']),

        ('StanfordDrone', [os.path.join(DATA_ROOT, 'Stanford Drone Dataset')],
         True, []),

        # VisDrone DET + MOT (已有manifest但未做质量检查)
        ('VisDrone-DET', [os.path.join(DATA_ROOT, 'VisDrone', 'Object Detection in Images')],
         True, ['/annotations/']),

        ('VisDrone-MOT', [os.path.join(DATA_ROOT, 'VisDrone', 'Multi-Object Tracking')],
         True, ['/annotations/', '/attributes/']),
    ]

    for name, dirs, recursive, excl_patterns in rgb_datasets:
        full_excl = ['/annotations/', '/labels/', '/gt/', '/anno/',
                     '/mask/', '/Annotations/', '/Labels/',
                     '/SegmentationClass/', '/ground_truth/'] + excl_patterns
        # 去重
        full_excl = list(set(full_excl))
        samples = clean_rgb_dataset(name, dirs, recursive, full_excl)
        all_samples.extend(samples)

    print(f"\n  Phase 4 总计: {len(all_samples)} 样本 (RGB-only 清洗后)")
    return all_samples


# ============================================================
# Phase 5: 合并全量 manifest
# ============================================================
def phase5_merge(all_samples):
    """合并所有样本为一个全量 manifest"""
    print("\n" + "=" * 70)
    print("Phase 5: 合并全量 manifest")
    print("=" * 70)

    # 统计
    dataset_counts = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        t = s.get('type', 'unknown')
        # 归一化 type
        type_map = {
            'paired': 'paired', 'pair': 'paired', 'rgb_ir': 'paired',
            'rgb_only': 'rgb_only', 'rgb': 'rgb_only', 'visible_only': 'rgb_only',
            'ir_only': 'ir_only', 'ir': 'ir_only', 'thermal_only': 'ir_only',
        }
        t = type_map.get(t, t)
        s['type'] = t  # 同时修正样本中的 type
        ds = s.get('dataset', 'unknown')
        dataset_counts[ds][t] += 1
        dataset_counts[ds]['total'] += 1

    # 打印统计表
    print(f"\n  {'数据集':<25} {'paired':>8} {'rgb_only':>10} {'ir_only':>10} {'total':>10}")
    print("  " + "-" * 65)
    total_paired = total_rgb = total_ir = 0
    for ds in sorted(dataset_counts.keys()):
        c = dataset_counts[ds]
        print(f"  {ds:<25} {c.get('paired',0):>8} {c.get('rgb_only',0):>10} {c.get('ir_only',0):>10} {c.get('total',0):>10}")
        total_paired += c.get('paired', 0)
        total_rgb += c.get('rgb_only', 0)
        total_ir += c.get('ir_only', 0)
    print("  " + "-" * 65)
    total = total_paired + total_rgb + total_ir
    print(f"  {'总计':<25} {total_paired:>8} {total_rgb:>10} {total_ir:>10} {total:>10}")

    # 写入全量 manifest
    merged = {
        'description': '全量预训练 manifest - 包含所有 RGB 和 IR 数据集',
        'generated_by': 'generate_full_manifest.py',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(all_samples),
        'total_datasets': len(dataset_counts),
        'statistics': {
            'paired': total_paired,
            'rgb_only': total_rgb,
            'ir_only': total_ir,
        },
        'dataset_counts': {ds: dict(c) for ds, c in sorted(dataset_counts.items())},
        'samples': all_samples,
    }

    output_path = os.path.join(MANIFEST_DIR, 'merged_full_pretrain.json')
    print(f"\n  写入全量 manifest: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(merged, f, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")
    print(f"  总样本数: {len(all_samples)}")

    return output_path


# ============================================================
# 主函数
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("全量预训练 Manifest 生成器")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"输出目录: {MANIFEST_DIR}")
    print(f"CPU 核心数: {NUM_WORKERS}")
    print("=" * 70)

    all_samples = []

    # Phase 1: 重用已清洗配对数据
    phase1_samples = phase1_reuse_clean_paired()
    all_samples.extend(phase1_samples)

    # Phase 2: 生成缺失 manifest
    phase2_samples = phase2_generate_missing()
    all_samples.extend(phase2_samples)

    # Phase 3: IR-only 质量清洗
    phase3_samples = phase3_clean_ir_only()
    all_samples.extend(phase3_samples)

    # Phase 4: RGB-only 质量清洗
    phase4_samples = phase4_clean_rgb_only()
    all_samples.extend(phase4_samples)

    # Phase 5: 合并
    output_path = phase5_merge(all_samples)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"全部完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"全量 manifest: {output_path}")
    print(f"总样本数: {len(all_samples)}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

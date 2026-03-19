#!/usr/bin/env python3
"""
修复旧配对数据集 manifest 的路径问题，然后重新合并全量 manifest。

路径问题:
- VT-Tiny-MOT: /data/VT-Tiny-MOT/VT-Tiny-MOT/ → /data/VT-Tiny-MOT/
- LLVIP reg: /data/LLVIP/registered/LLVIP/ → /data/LLVIP/registered/
- LLVIP unreg: /data/LLVIP/unregistered/LLVIP_raw_data/LLVIP_raw_images/ → /data/LLVIP/unregistered/LLVIP_raw_images/
- DroneVehicle: /data/DroneVehicle/train/train/ → /data/DroneVehicle/train/
- M3OT: /data/M3OT/M3OT/2/ → /data/M3OT/2/
- DroneRGBT: /data/DroneRGBT/DroneRGBT/DroneRGBT/ → /data/DroneRGBT/
"""

import os, json, time
from collections import defaultdict

DATA_ROOT = '/root/autodl-tmp/data'
MANIFEST_DIR = '/root/autodl-tmp/train1/manifests'
CONVERTED_IR = '/root/autodl-tmp/train1/converted_ir'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}


def find_images(directory, recursive=True):
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


def write_manifest(filepath, dataset, samples, total_scanned=None):
    if total_scanned is None:
        total_scanned = len(samples)
    manifest = {
        'dataset': dataset,
        'generated_by': 'fix_paired_manifests.py',
        'total_scanned': total_scanned,
        'total_clean': len(samples),
        'total_excluded': total_scanned - len(samples),
        'samples': samples,
    }
    with open(filepath, 'w') as f:
        json.dump(manifest, f, ensure_ascii=False)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  写入: {filepath} ({size_mb:.1f} MB, {len(samples)} 样本)")


def gen_vt_tiny_mot():
    """VT-Tiny-MOT: 正确路径 /data/VT-Tiny-MOT/{train,test}2017/<seq>/{00,01}/"""
    print("\n--- 修复 VT-Tiny-MOT ---")
    base = os.path.join(DATA_ROOT, 'VT-Tiny-MOT')
    all_samples = []

    for split in ['train2017', 'test2017']:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        count = 0
        for seq in sorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq, '00')
            ir_dir = os.path.join(split_dir, seq, '01')
            if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
                continue
            rgb_files = {f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS}
            ir_files = {f for f in os.listdir(ir_dir) if os.path.splitext(f)[1] in IMG_EXTS}
            for f in sorted(rgb_files & ir_files):
                all_samples.append({
                    'rgb_path': os.path.join(rgb_dir, f),
                    'ir_path': os.path.join(ir_dir, f),
                    'type': 'paired',
                    'dataset': 'VT-Tiny-MOT',
                    'sequence': seq,
                })
                count += 1
        print(f"  {split}: {count} 对")

    write_manifest(os.path.join(MANIFEST_DIR, 'VT-Tiny-MOT_all_fixed.json'),
                    'VT-Tiny-MOT', all_samples)
    return all_samples


def gen_llvip_registered():
    """LLVIP registered: RGB /data/LLVIP/registered/visible/{train,test}/, IR converted"""
    print("\n--- 修复 LLVIP (registered) ---")
    all_samples = []

    for split in ['train', 'test']:
        rgb_dir = os.path.join(DATA_ROOT, 'LLVIP', 'registered', 'visible', split)
        ir_dir = os.path.join(CONVERTED_IR, 'LLVIP', split)

        if not os.path.isdir(rgb_dir):
            print(f"  [警告] {rgb_dir} 不存在!")
            continue

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS])
        count = 0
        for f in rgb_files:
            rgb_path = os.path.join(rgb_dir, f)
            ir_path = os.path.join(ir_dir, f)
            if os.path.isdir(ir_dir) and os.path.exists(ir_path):
                all_samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'type': 'paired',
                    'dataset': 'LLVIP',
                })
                count += 1
            else:
                # IR not converted, use original IR from registered/infrared/
                orig_ir = os.path.join(DATA_ROOT, 'LLVIP', 'registered', 'infrared', split, f)
                if os.path.exists(orig_ir):
                    all_samples.append({
                        'rgb_path': rgb_path,
                        'ir_path': orig_ir,
                        'type': 'paired',
                        'dataset': 'LLVIP',
                        'ir_is_pseudo_rgb': True,
                    })
                    count += 1
        print(f"  registered {split}: {count} 对")

    return all_samples


def gen_llvip_unreg():
    """LLVIP unregistered: /data/LLVIP/unregistered/LLVIP_raw_images/{train,test}/{visible,infrared}/"""
    print("\n--- 修复 LLVIP (unregistered) ---")
    all_samples = []

    for split in ['train', 'test']:
        rgb_dir = os.path.join(DATA_ROOT, 'LLVIP', 'unregistered', 'LLVIP_raw_images', split, 'visible')
        ir_conv_dir = os.path.join(CONVERTED_IR, 'LLVIP_unreg', split)

        if not os.path.isdir(rgb_dir):
            print(f"  [警告] {rgb_dir} 不存在!")
            continue

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS])
        count = 0
        for f in rgb_files:
            rgb_path = os.path.join(rgb_dir, f)
            ir_path = os.path.join(ir_conv_dir, f)
            if os.path.isdir(ir_conv_dir) and os.path.exists(ir_path):
                all_samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'type': 'paired',
                    'dataset': 'LLVIP-unreg',
                })
                count += 1
            else:
                orig_ir = os.path.join(DATA_ROOT, 'LLVIP', 'unregistered', 'LLVIP_raw_images', split, 'infrared', f)
                if os.path.exists(orig_ir):
                    all_samples.append({
                        'rgb_path': rgb_path,
                        'ir_path': orig_ir,
                        'type': 'paired',
                        'dataset': 'LLVIP-unreg',
                        'ir_is_pseudo_rgb': True,
                    })
                    count += 1
        print(f"  unregistered {split}: {count} 对")

    return all_samples


def gen_drone_vehicle():
    """DroneVehicle: /data/DroneVehicle/{train,val,test}/trainimg/ etc."""
    print("\n--- 修复 DroneVehicle ---")
    all_samples = []

    splits = {
        'train': ('trainimg', 'trainimgr'),
        'val': ('valimg', 'valimgr'),
        'test': ('testimg', 'testimgr'),
    }

    for split, (rgb_sub, ir_sub) in splits.items():
        rgb_dir = os.path.join(DATA_ROOT, 'DroneVehicle', split, rgb_sub)
        ir_conv_dir = os.path.join(CONVERTED_IR, 'DroneVehicle', split)

        if not os.path.isdir(rgb_dir):
            print(f"  [警告] {rgb_dir} 不存在!")
            continue

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS])
        count = 0
        for f in rgb_files:
            rgb_path = os.path.join(rgb_dir, f)
            # converted IR 使用 .jpg 扩展名
            ir_path = os.path.join(ir_conv_dir, f)
            if os.path.isdir(ir_conv_dir) and os.path.exists(ir_path):
                all_samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'type': 'paired',
                    'dataset': 'DroneVehicle',
                })
                count += 1
            else:
                # fallback to original IR
                orig_ir = os.path.join(DATA_ROOT, 'DroneVehicle', split, ir_sub, f)
                if os.path.exists(orig_ir):
                    all_samples.append({
                        'rgb_path': rgb_path,
                        'ir_path': orig_ir,
                        'type': 'paired',
                        'dataset': 'DroneVehicle',
                        'ir_is_pseudo_rgb': True,
                    })
                    count += 1
        print(f"  {split}: {count} 对")

    return all_samples


def gen_m3ot():
    """M3OT subdir 2: /data/M3OT/2/rgb/{train,test,val}/<seq>/img1/*.PNG"""
    print("\n--- 修复 M3OT ---")
    all_samples = []

    for split in ['train', 'test', 'val']:
        rgb_base = os.path.join(DATA_ROOT, 'M3OT', '2', 'rgb', split)
        ir_base = os.path.join(DATA_ROOT, 'M3OT', '2', 'ir', split)
        ir_conv_base = os.path.join(CONVERTED_IR, 'M3OT', split)

        if not os.path.isdir(rgb_base):
            print(f"  [警告] {rgb_base} 不存在!")
            continue

        count = 0
        for seq in sorted(os.listdir(rgb_base)):
            rgb_dir = os.path.join(rgb_base, seq, 'img1')
            ir_seq = seq + 'T'  # IR sequences have T suffix
            ir_conv_dir = os.path.join(ir_conv_base, ir_seq)

            if not os.path.isdir(rgb_dir):
                continue

            rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS])
            for f in rgb_files:
                rgb_path = os.path.join(rgb_dir, f)
                ir_path = os.path.join(ir_conv_dir, f)

                if os.path.isdir(ir_conv_dir) and os.path.exists(ir_path):
                    all_samples.append({
                        'rgb_path': rgb_path,
                        'ir_path': ir_path,
                        'type': 'paired',
                        'dataset': 'M3OT',
                        'sequence': seq,
                    })
                    count += 1
                else:
                    # fallback: try original IR
                    orig_ir_dir = os.path.join(ir_base, ir_seq, 'img1')
                    orig_ir = os.path.join(orig_ir_dir, f)
                    if os.path.exists(orig_ir):
                        all_samples.append({
                            'rgb_path': rgb_path,
                            'ir_path': orig_ir,
                            'type': 'paired',
                            'dataset': 'M3OT',
                            'sequence': seq,
                            'ir_is_pseudo_rgb': True,
                        })
                        count += 1
        print(f"  {split}: {count} 对")

    return all_samples


def gen_dronergbt():
    """DroneRGBT: /data/DroneRGBT/{Train,Test}/RGB/ and converted_ir"""
    print("\n--- 修复 DroneRGBT ---")
    all_samples = []

    for split in ['Train', 'Test']:
        rgb_dir = os.path.join(DATA_ROOT, 'DroneRGBT', split, 'RGB')
        ir_conv_dir = os.path.join(CONVERTED_IR, 'DroneRGBT', split)

        if not os.path.isdir(rgb_dir):
            print(f"  [警告] {rgb_dir} 不存在!")
            continue

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS])
        count = 0
        for f in rgb_files:
            rgb_path = os.path.join(rgb_dir, f)
            ir_path = os.path.join(ir_conv_dir, f)

            if os.path.isdir(ir_conv_dir) and os.path.exists(ir_path):
                all_samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'type': 'paired',
                    'dataset': 'DroneRGBT',
                })
                count += 1
            else:
                orig_ir = os.path.join(DATA_ROOT, 'DroneRGBT', split, 'Infrared', f)
                if os.path.exists(orig_ir):
                    all_samples.append({
                        'rgb_path': rgb_path,
                        'ir_path': orig_ir,
                        'type': 'paired',
                        'dataset': 'DroneRGBT',
                        'ir_is_pseudo_rgb': True,
                    })
                    count += 1
        print(f"  {split}: {count} 对")

    return all_samples


def gen_rgbt_tiny():
    """RGBT-Tiny: verified OK, but regenerate with type field for completeness"""
    print("\n--- 修复 RGBT-Tiny (添加 type 字段) ---")
    base = os.path.join(DATA_ROOT, 'RGBT-Tiny', 'images')
    all_samples = []

    if not os.path.isdir(base):
        print("  [错误] RGBT-Tiny/images 不存在!")
        return []

    for seq in sorted(os.listdir(base)):
        rgb_dir = os.path.join(base, seq, '00')
        ir_dir = os.path.join(base, seq, '01')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
            continue
        rgb_files = {f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1] in IMG_EXTS}
        ir_files = {f for f in os.listdir(ir_dir) if os.path.splitext(f)[1] in IMG_EXTS}
        for f in sorted(rgb_files & ir_files):
            all_samples.append({
                'rgb_path': os.path.join(rgb_dir, f),
                'ir_path': os.path.join(ir_dir, f),
                'type': 'paired',
                'dataset': 'RGBT-Tiny',
                'sequence': seq,
            })

    print(f"  总计: {len(all_samples)} 对")
    write_manifest(os.path.join(MANIFEST_DIR, 'RGBT-Tiny_all_fixed.json'),
                    'RGBT-Tiny', all_samples)
    return all_samples


def main():
    print("=" * 60)
    print("修复配对数据集 manifest 路径")
    print("=" * 60)

    all_fixed = []

    # RGBT-Tiny (path OK, just add type)
    all_fixed.extend(gen_rgbt_tiny())

    # VT-Tiny-MOT
    all_fixed.extend(gen_vt_tiny_mot())

    # LLVIP
    llvip_reg = gen_llvip_registered()
    llvip_unreg = gen_llvip_unreg()
    llvip_all = llvip_reg + llvip_unreg
    write_manifest(os.path.join(MANIFEST_DIR, 'LLVIP_all_fixed.json'), 'LLVIP', llvip_all)
    all_fixed.extend(llvip_all)

    # DroneVehicle
    dv = gen_drone_vehicle()
    write_manifest(os.path.join(MANIFEST_DIR, 'DroneVehicle_all_fixed.json'), 'DroneVehicle', dv)
    all_fixed.extend(dv)

    # M3OT
    m3ot = gen_m3ot()
    write_manifest(os.path.join(MANIFEST_DIR, 'M3OT_all_fixed.json'), 'M3OT', m3ot)
    all_fixed.extend(m3ot)

    # DroneRGBT
    drgbt = gen_dronergbt()
    write_manifest(os.path.join(MANIFEST_DIR, 'DroneRGBT_all_fixed.json'), 'DroneRGBT', drgbt)
    all_fixed.extend(drgbt)

    print(f"\n总计修复: {len(all_fixed)} 个配对样本")

    # 验证所有路径
    print("\n验证路径...")
    missing = 0
    for s in all_fixed:
        for key in ['rgb_path', 'ir_path']:
            p = s.get(key)
            if p and not os.path.exists(p):
                missing += 1
                if missing <= 5:
                    print(f"  缺失: {p}")
    print(f"路径验证: {len(all_fixed)*2 - missing} 存在, {missing} 缺失")

    # ================================================
    # 重新合并全量 manifest
    # ================================================
    print("\n" + "=" * 60)
    print("重新合并全量 manifest (使用修复后的配对数据)")
    print("=" * 60)

    # 修复后的配对数据 (替代旧的)
    fixed_paired = all_fixed

    # TarDAL (路径已OK)
    tardal_manifests = ['TarDAL_M3FD.json', 'TarDAL_tno.json']
    for mf in tardal_manifests:
        path = os.path.join(MANIFEST_DIR, mf)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            samples = data.get('samples', [])
            # 添加 type 字段
            for s in samples:
                if 'type' not in s:
                    s['type'] = 'paired'
            fixed_paired.extend(samples)
            print(f"  {mf}: {len(samples)} 样本")

    # Phase 2 新生成的
    new_manifests = [
        'UTUAV_all_v2.json', 'WebUAV-3M_all.json', 'VisDrone-SOT_all.json',
        'VisDrone-CC_all.json', 'TarDAL_roadscene.json', 'TarDAL_M3FD_Fusion.json',
        'M3OT_1_ir.json',
    ]

    # Phase 3 IR-only 清洗后
    ir_manifests = [
        'LSOTB_TIR_clean.json', 'BIRDSAI_clean.json',
        'MONET_clean.json', 'HIT_UAV_clean.json',
    ]

    # Phase 4 RGB-only 清洗后
    rgb_manifests = [
        'AU_AIR_clean.json', 'AeroScapes_clean.json', 'AnimalDrone_clean.json',
        'DroneCrowd_clean.json', 'EVD4UAV_clean.json', 'UAV123_clean.json',
        'UAVDT_clean.json', 'UAVScenes_clean.json', 'UAVid_clean.json',
        'UDD_clean.json', 'UAV_Human_clean.json', 'University_1652_clean.json',
        'SUES_200_clean.json', 'CVOGL_clean.json', 'DTB70_clean.json',
        'Manipal_UAV_clean.json', 'MDMT_clean.json', 'SeaDronesSee_clean.json',
        'VisDrone_DET_clean.json', 'VisDrone_MOT_clean.json',
    ]

    all_samples = list(fixed_paired)

    for mf_list in [new_manifests, ir_manifests, rgb_manifests]:
        for mf in mf_list:
            path = os.path.join(MANIFEST_DIR, mf)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                samples = data.get('samples', data.get('data', []))
                # type 归一化
                TYPE_MAP = {
                    'paired': 'paired', 'pair': 'paired',
                    'rgb_only': 'rgb_only', 'rgb': 'rgb_only',
                    'ir_only': 'ir_only', 'ir': 'ir_only',
                }
                for s in samples:
                    t = s.get('type')
                    if t:
                        s['type'] = TYPE_MAP.get(t, t)
                    else:
                        has_rgb = s.get('rgb_path') is not None
                        has_ir = s.get('ir_path') is not None
                        s['type'] = 'paired' if (has_rgb and has_ir) else ('rgb_only' if has_rgb else 'ir_only')
                all_samples.extend(samples)
                print(f"  {mf}: {len(samples)} 样本")
            else:
                print(f"  [缺失] {mf}")

    # 统计
    ds_counts = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        ds = s.get('dataset', 'unknown')
        t = s.get('type', 'unknown')
        ds_counts[ds][t] += 1
        ds_counts[ds]['total'] += 1

    print(f"\n{'数据集':<30} {'paired':>8} {'rgb_only':>10} {'ir_only':>10} {'total':>10}")
    print("-" * 70)
    tp = tr = ti = 0
    for ds in sorted(ds_counts.keys()):
        c = ds_counts[ds]
        p, r, i, t = c.get('paired',0), c.get('rgb_only',0), c.get('ir_only',0), c.get('total',0)
        print(f"  {ds:<28} {p:>8} {r:>10} {i:>10} {t:>10}")
        tp += p; tr += r; ti += i
    print("-" * 70)
    total = tp + tr + ti
    print(f"  {'总计':<28} {tp:>8} {tr:>10} {ti:>10} {total:>10}")

    # 写入
    merged = {
        'description': '全量预训练 manifest (路径已修复) - 包含所有 RGB+IR 数据集',
        'generated_by': 'fix_paired_manifests.py',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(all_samples),
        'total_datasets': len(ds_counts),
        'statistics': {'paired': tp, 'rgb_only': tr, 'ir_only': ti},
        'dataset_counts': {ds: dict(c) for ds, c in sorted(ds_counts.items())},
        'samples': all_samples,
    }

    output_path = os.path.join(MANIFEST_DIR, 'merged_full_pretrain.json')
    print(f"\n写入: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(merged, f, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"文件大小: {size_mb:.1f} MB, 总样本: {len(all_samples)}")

    # 最终路径验证
    print("\n最终路径抽样验证 (200样本)...")
    import random
    random.seed(42)
    check_samples = random.sample(all_samples, min(200, len(all_samples)))
    ok = bad = 0
    for s in check_samples:
        for key in ['rgb_path', 'ir_path']:
            p = s.get(key)
            if p and os.path.exists(p):
                ok += 1
            elif p:
                bad += 1
    print(f"  路径存在: {ok}, 路径缺失: {bad}")
    if bad == 0:
        print("  所有抽样路径验证通过!")


if __name__ == '__main__':
    main()

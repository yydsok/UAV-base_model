#!/usr/bin/env python3
"""
快速合并所有已生成的 manifest 为全量 manifest
（Phase 1-4 已完成，只需执行 Phase 5 合并）
"""
import os, json, time
from collections import defaultdict

MANIFEST_DIR = '/root/autodl-tmp/train1/manifests'

# 定义要合并的 manifest 文件
# Phase 1: 已清洗配对数据集
PAIRED_MANIFESTS = [
    'RGBT-Tiny_train.json', 'RGBT-Tiny_test.json',
    'VT-Tiny-MOT_train.json', 'VT-Tiny-MOT_test.json',
    'LLVIP_train.json', 'LLVIP_test.json', 'LLVIP_unreg_all.json',
    'DroneVehicle_train.json', 'DroneVehicle_val.json', 'DroneVehicle_test.json',
    'M3OT_train.json', 'M3OT_val.json', 'M3OT_test.json',
    'TarDAL_M3FD.json', 'TarDAL_tno.json',
    'DroneRGBT_train.json', 'DroneRGBT_test.json',
]

# Phase 2: 新生成的 manifest
NEW_MANIFESTS = [
    'UTUAV_all_v2.json',
    'WebUAV-3M_all.json',
    'VisDrone-SOT_all.json',
    'VisDrone-CC_all.json',
    'TarDAL_roadscene.json',
    'TarDAL_M3FD_Fusion.json',
    'M3OT_1_ir.json',
]

# Phase 3: IR-only 清洗后
IR_CLEAN_MANIFESTS = [
    'LSOTB_TIR_clean.json',
    'BIRDSAI_clean.json',
    'MONET_clean.json',
    'HIT_UAV_clean.json',
]

# Phase 4: RGB-only 清洗后
RGB_CLEAN_MANIFESTS = [
    'AU_AIR_clean.json',
    'AeroScapes_clean.json',
    'AnimalDrone_clean.json',
    'DroneCrowd_clean.json',
    'EVD4UAV_clean.json',
    'UAV123_clean.json',
    'UAVDT_clean.json',
    'UAVScenes_clean.json',
    'UAVid_clean.json',
    'UDD_clean.json',
    'UAV_Human_clean.json',
    'University_1652_clean.json',
    'SUES_200_clean.json',
    'CVOGL_clean.json',
    'DTB70_clean.json',
    'Manipal_UAV_clean.json',
    'MDMT_clean.json',
    'SeaDronesSee_clean.json',
    'VisDrone_DET_clean.json',
    'VisDrone_MOT_clean.json',
]

# Type 归一化映射
TYPE_MAP = {
    'paired': 'paired', 'pair': 'paired', 'rgb_ir': 'paired',
    'rgb_only': 'rgb_only', 'rgb': 'rgb_only', 'visible_only': 'rgb_only',
    'ir_only': 'ir_only', 'ir': 'ir_only', 'thermal_only': 'ir_only',
}

def load_manifest(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('samples', data.get('data', []))

def main():
    all_manifests = PAIRED_MANIFESTS + NEW_MANIFESTS + IR_CLEAN_MANIFESTS + RGB_CLEAN_MANIFESTS
    all_samples = []

    print(f"合并 {len(all_manifests)} 个 manifest...")
    for mf in all_manifests:
        path = os.path.join(MANIFEST_DIR, mf)
        if os.path.exists(path):
            samples = load_manifest(path)
            print(f"  {mf}: {len(samples)} 样本")
            all_samples.extend(samples)
        else:
            print(f"  [缺失] {mf}")

    # 归一化 type：如果 type 缺失，根据 rgb_path/ir_path 推断
    for s in all_samples:
        t = s.get('type', None)
        if t:
            s['type'] = TYPE_MAP.get(t, t)
        else:
            # 推断 type
            has_rgb = s.get('rgb_path') is not None
            has_ir = s.get('ir_path') is not None
            if has_rgb and has_ir:
                s['type'] = 'paired'
            elif has_rgb:
                s['type'] = 'rgb_only'
            elif has_ir:
                s['type'] = 'ir_only'
            else:
                s['type'] = 'unknown'

    # 统计
    dataset_counts = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        ds = s.get('dataset', 'unknown')
        t = s.get('type', 'unknown')
        dataset_counts[ds][t] += 1
        dataset_counts[ds]['total'] += 1

    print(f"\n{'数据集':<30} {'paired':>8} {'rgb_only':>10} {'ir_only':>10} {'total':>10}")
    print("-" * 70)
    total_p = total_r = total_i = 0
    for ds in sorted(dataset_counts.keys()):
        c = dataset_counts[ds]
        p, r, i, t = c.get('paired',0), c.get('rgb_only',0), c.get('ir_only',0), c.get('total',0)
        print(f"  {ds:<28} {p:>8} {r:>10} {i:>10} {t:>10}")
        total_p += p; total_r += r; total_i += i
    print("-" * 70)
    total = total_p + total_r + total_i
    print(f"  {'总计':<28} {total_p:>8} {total_r:>10} {total_i:>10} {total:>10}")

    # 写入
    merged = {
        'description': '全量预训练 manifest - 包含 /root/autodl-tmp/data/ 中所有 RGB 和 IR 数据集',
        'generated_by': 'merge_all_manifests.py',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(all_samples),
        'total_datasets': len(dataset_counts),
        'statistics': {
            'paired': total_p,
            'rgb_only': total_r,
            'ir_only': total_i,
        },
        'dataset_counts': {ds: dict(c) for ds, c in sorted(dataset_counts.items())},
        'samples': all_samples,
    }

    output_path = os.path.join(MANIFEST_DIR, 'merged_full_pretrain.json')
    print(f"\n写入: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(merged, f, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"文件大小: {size_mb:.1f} MB")
    print(f"总样本数: {len(all_samples)}")
    print(f"总数据集: {len(dataset_counts)}")

if __name__ == '__main__':
    main()

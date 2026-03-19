#!/usr/bin/env python3
"""
Generate JSON manifests for newly added datasets (excluding WebUAV-3M).

Covers 15 datasets:
  1. LLVIP unregistered (paired) — replaces old registered LLVIP
  2. BIRDSAI (ir_only)
  3. LSOTB-TIR (ir_only)
  4. SeaDronesSee MOT (rgb_only)
  5. CVOGL drone (rgb_only)
  6. University-1652 drone (rgb_only)
  7. SUES-200 drone (rgb_only)
  8. UAV-Human (rgb_only)
  9. UAVScenes (rgb_only)
 10. MDMT (rgb_only)
 11. Manipal-UAV (rgb_only)
 12. DTB70 (rgb_only)
 13. UAVid (rgb_only)
 14. UDD (rgb_only)
 15. Stanford Drone (rgb_only)

Also converts LLVIP unregistered IR images (pseudo-RGB) to single-channel
grayscale and saves to converted_ir/LLVIP_unreg/.

Usage:
    python scripts/generate_new_manifests.py [--data_root /root/autodl-tmp/data]
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}


def is_image(p: str) -> bool:
    return os.path.splitext(p)[1] in IMAGE_EXTS


def collect_images(root: str, recursive: bool = True) -> list[str]:
    """Collect image file paths under *root*."""
    paths = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1] in IMAGE_EXTS:
                    paths.append(os.path.join(dirpath, fn))
    else:
        for fn in sorted(os.listdir(root)):
            fp = os.path.join(root, fn)
            if os.path.isfile(fp) and os.path.splitext(fn)[1] in IMAGE_EXTS:
                paths.append(fp)
    return paths


def write_manifest(out_path: str, dataset: str, modality_type: str,
                   samples: list[dict]):
    """Write a manifest JSON file."""
    manifest = {
        "dataset": dataset,
        "split": "all",
        "modality_type": modality_type,
        "total_scanned": len(samples),
        "total_clean": len(samples),
        "total_excluded": 0,
        "anomaly_counts": {},
        "samples": samples,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  -> {out_path}  ({len(samples)} samples)")


# ---------------------------------------------------------------------------
# Per-dataset generators
# ---------------------------------------------------------------------------

def gen_llvip_unreg(data_root: str, out_dir: str, ir_out_root: str):
    """LLVIP unregistered — paired (visible + IR grayscale-converted)."""
    base = os.path.join(data_root, "LLVIP", "unregistered",
                        "LLVIP_raw_data", "LLVIP_raw_images")
    converted_root = os.path.join(ir_out_root, "LLVIP_unreg")
    samples = []
    converted_count = 0

    for split in ("train", "test"):
        vis_dir = os.path.join(base, split, "visible")
        ir_dir = os.path.join(base, split, "infrared")
        conv_dir = os.path.join(converted_root, split)
        os.makedirs(conv_dir, exist_ok=True)

        vis_files = sorted(os.listdir(vis_dir))
        for fn in vis_files:
            if not is_image(fn):
                continue
            vis_path = os.path.join(vis_dir, fn)
            ir_src = os.path.join(ir_dir, fn)
            if not os.path.isfile(ir_src):
                continue

            # Convert IR (pseudo-RGB 1280x720) to single-channel grayscale
            ir_dst = os.path.join(conv_dir, fn)
            if not os.path.exists(ir_dst):
                img = Image.open(ir_src)
                if img.mode != 'L':
                    img = img.convert('L')
                img.save(ir_dst)
                converted_count += 1

            samples.append({
                "rgb_path": vis_path,
                "ir_path": ir_dst,
                "dataset": "LLVIP-unreg",
                "type": "paired",
                "ir_converted": False,  # real IR, just grayscale-converted
            })

    print(f"  LLVIP-unreg: converted {converted_count} new IR images")
    write_manifest(os.path.join(out_dir, "LLVIP_unreg_all.json"),
                   "LLVIP-unreg", "paired", samples)


def gen_birdsai(data_root: str, out_dir: str):
    """BIRDSAI — ir_only (640x512 thermal)."""
    base = os.path.join(data_root, "BIRDSAI")
    samples = []
    for split_dir in ("TrainReal", "TestReal"):
        img_root = os.path.join(base, split_dir, "images")
        if not os.path.isdir(img_root):
            continue
        for seq in sorted(os.listdir(img_root)):
            seq_dir = os.path.join(img_root, seq)
            if not os.path.isdir(seq_dir):
                continue
            for fn in sorted(os.listdir(seq_dir)):
                fp = os.path.join(seq_dir, fn)
                if os.path.isfile(fp) and is_image(fp):
                    samples.append({
                        "rgb_path": None,
                        "ir_path": fp,
                        "dataset": "BIRDSAI",
                        "type": "ir_only",
                        "ir_converted": False,
                    })

    write_manifest(os.path.join(out_dir, "BIRDSAI_all.json"),
                   "BIRDSAI", "ir_only", samples)


def gen_lsotb_tir(data_root: str, out_dir: str):
    """LSOTB-TIR — ir_only (thermal tracking dataset)."""
    base = os.path.join(data_root, "LSOTB-TIR")
    samples = []

    # Training Dataset / TrainingData / TIR_training_00X / <seq> / *.jpg
    training_root = os.path.join(base, "Training Dataset", "TrainingData")
    if os.path.isdir(training_root):
        for part in sorted(os.listdir(training_root)):
            part_dir = os.path.join(training_root, part)
            if not os.path.isdir(part_dir):
                continue
            for seq in sorted(os.listdir(part_dir)):
                seq_dir = os.path.join(part_dir, seq)
                if not os.path.isdir(seq_dir):
                    continue
                for fn in sorted(os.listdir(seq_dir)):
                    fp = os.path.join(seq_dir, fn)
                    if os.path.isfile(fp) and is_image(fp):
                        samples.append({
                            "rgb_path": None,
                            "ir_path": fp,
                            "dataset": "LSOTB-TIR",
                            "type": "ir_only",
                            "ir_converted": False,
                        })

    # Evaluation Dataset / <target> / img / *.jpg
    eval_root = os.path.join(base, "Evaluation Dataset")
    if os.path.isdir(eval_root):
        for target in sorted(os.listdir(eval_root)):
            img_dir = os.path.join(eval_root, target, "img")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                fp = os.path.join(img_dir, fn)
                if os.path.isfile(fp) and is_image(fp):
                    samples.append({
                        "rgb_path": None,
                        "ir_path": fp,
                        "dataset": "LSOTB-TIR",
                        "type": "ir_only",
                        "ir_converted": False,
                    })

    write_manifest(os.path.join(out_dir, "LSOTB-TIR_all.json"),
                   "LSOTB-TIR", "ir_only", samples)


def gen_seadronessee_mot(data_root: str, out_dir: str):
    """SeaDronesSee MOT — rgb_only (Compressed version)."""
    base = os.path.join(data_root, "SeaDronesSee",
                        "SeaDronesSee_MOT_Multi-Object Tracking",
                        "SeaDronesSee_MOT", "Compressed")
    samples = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in sorted(os.listdir(split_dir)):
            fp = os.path.join(split_dir, fn)
            if os.path.isfile(fp) and is_image(fp):
                samples.append({
                    "rgb_path": fp,
                    "ir_path": None,
                    "dataset": "SeaDronesSee-MOT",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "SeaDronesSee_MOT_all.json"),
                   "SeaDronesSee-MOT", "rgb_only", samples)


def gen_cvogl_drone(data_root: str, out_dir: str):
    """CVOGL drone aerial — rgb_only (exclude satellite/ and SVI)."""
    query_dir = os.path.join(data_root, "CVOGL", "CVOGL_DroneAerial", "query")
    samples = []
    for fn in sorted(os.listdir(query_dir)):
        fp = os.path.join(query_dir, fn)
        if os.path.isfile(fp) and is_image(fp):
            samples.append({
                "rgb_path": fp,
                "ir_path": None,
                "dataset": "CVOGL-drone",
                "type": "rgb_only",
                "ir_converted": False,
            })

    write_manifest(os.path.join(out_dir, "CVOGL_drone_all.json"),
                   "CVOGL-drone", "rgb_only", samples)


def gen_university1652_drone(data_root: str, out_dir: str):
    """University-1652 drone — rgb_only (exclude satellite/google/street)."""
    base = os.path.join(data_root, "University-1652")
    samples = []

    # train/drone/<id>/*.jpeg
    train_drone = os.path.join(base, "train", "drone")
    if os.path.isdir(train_drone):
        for subdir in sorted(os.listdir(train_drone)):
            d = os.path.join(train_drone, subdir)
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    fp = os.path.join(d, fn)
                    if os.path.isfile(fp) and is_image(fp):
                        samples.append({
                            "rgb_path": fp,
                            "ir_path": None,
                            "dataset": "University-1652-drone",
                            "type": "rgb_only",
                            "ir_converted": False,
                        })

    # test/gallery_drone/<id>/*.jpeg  and  test/query_drone/<id>/*.jpeg
    for subname in ("gallery_drone", "query_drone", "4K_drone"):
        test_dir = os.path.join(base, "test", subname)
        if not os.path.isdir(test_dir):
            continue
        for subdir in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, subdir)
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    fp = os.path.join(d, fn)
                    if os.path.isfile(fp) and is_image(fp):
                        samples.append({
                            "rgb_path": fp,
                            "ir_path": None,
                            "dataset": "University-1652-drone",
                            "type": "rgb_only",
                            "ir_converted": False,
                        })

    write_manifest(os.path.join(out_dir, "University-1652_drone_all.json"),
                   "University-1652-drone", "rgb_only", samples)


def gen_sues200_drone(data_root: str, out_dir: str):
    """SUES-200 drone_view_512 — rgb_only (exclude satellite-view)."""
    base = os.path.join(data_root, "SUES-200", "drone_view_512")
    samples = []

    # Structure: drone_view_512/<id>/<scale>/*.jpg
    for subdir in sorted(os.listdir(base)):
        id_dir = os.path.join(base, subdir)
        if not os.path.isdir(id_dir):
            continue
        for scale in sorted(os.listdir(id_dir)):
            scale_dir = os.path.join(id_dir, scale)
            if not os.path.isdir(scale_dir):
                continue
            for fn in sorted(os.listdir(scale_dir)):
                fp = os.path.join(scale_dir, fn)
                if os.path.isfile(fp) and is_image(fp):
                    samples.append({
                        "rgb_path": fp,
                        "ir_path": None,
                        "dataset": "SUES-200-drone",
                        "type": "rgb_only",
                        "ir_converted": False,
                    })

    write_manifest(os.path.join(out_dir, "SUES-200_drone_all.json"),
                   "SUES-200-drone", "rgb_only", samples)


def gen_uav_human(data_root: str, out_dir: str):
    """UAV-Human PoseEstimation — rgb_only."""
    frames_dir = os.path.join(data_root, "UAV-Human", "PoseEstimation", "frames")
    samples = []
    for fn in sorted(os.listdir(frames_dir)):
        fp = os.path.join(frames_dir, fn)
        if os.path.isfile(fp) and is_image(fp):
            samples.append({
                "rgb_path": fp,
                "ir_path": None,
                "dataset": "UAV-Human",
                "type": "rgb_only",
                "ir_converted": False,
            })

    write_manifest(os.path.join(out_dir, "UAV-Human_all.json"),
                   "UAV-Human", "rgb_only", samples)


def gen_uavscenes(data_root: str, out_dir: str):
    """UAVScenes interval=5 — rgb_only (only camera images)."""
    base = os.path.join(data_root, "UAVScenes", "interval=5",
                        "interval5_CAM_LIDAR")
    samples = []
    if not os.path.isdir(base):
        print(f"  WARNING: {base} not found")
        return

    for scene in sorted(os.listdir(base)):
        scene_dir = os.path.join(base, scene)
        if not os.path.isdir(scene_dir):
            continue
        cam_dir = os.path.join(scene_dir, "interval5_CAM")
        if not os.path.isdir(cam_dir):
            continue
        for fn in sorted(os.listdir(cam_dir)):
            fp = os.path.join(cam_dir, fn)
            if os.path.isfile(fp) and is_image(fp):
                samples.append({
                    "rgb_path": fp,
                    "ir_path": None,
                    "dataset": "UAVScenes",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "UAVScenes_all.json"),
                   "UAVScenes", "rgb_only", samples)


def gen_mdmt(data_root: str, out_dir: str):
    """MDMT — rgb_only (multi-domain multi-target tracking)."""
    base = os.path.join(data_root, "MDMT")
    samples = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        for fp in collect_images(split_dir, recursive=True):
            samples.append({
                "rgb_path": fp,
                "ir_path": None,
                "dataset": "MDMT",
                "type": "rgb_only",
                "ir_converted": False,
            })

    write_manifest(os.path.join(out_dir, "MDMT_all.json"),
                   "MDMT", "rgb_only", samples)


def gen_manipal_uav(data_root: str, out_dir: str):
    """Manipal-UAV — rgb_only (only images/, exclude labels/)."""
    base = os.path.join(data_root, "Manipal-UAV", "Manipal-UAV")
    samples = []
    for split in ("test", "validation"):
        img_dir = os.path.join(base, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            fp = os.path.join(img_dir, fn)
            if os.path.isfile(fp) and is_image(fp):
                samples.append({
                    "rgb_path": fp,
                    "ir_path": None,
                    "dataset": "Manipal-UAV",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "Manipal-UAV_all.json"),
                   "Manipal-UAV", "rgb_only", samples)


def gen_dtb70(data_root: str, out_dir: str):
    """DTB70 — rgb_only (drone tracking benchmark)."""
    base = os.path.join(data_root, "DTB70")
    samples = []
    for seq in sorted(os.listdir(base)):
        img_dir = os.path.join(base, seq, "img")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            fp = os.path.join(img_dir, fn)
            if os.path.isfile(fp) and is_image(fp):
                samples.append({
                    "rgb_path": fp,
                    "ir_path": None,
                    "dataset": "DTB70",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "DTB70_all.json"),
                   "DTB70", "rgb_only", samples)


def gen_uavid(data_root: str, out_dir: str):
    """UAVid — rgb_only (only Images/, exclude Labels/)."""
    base = os.path.join(data_root, "UAVid")
    samples = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        for seq in sorted(os.listdir(split_dir)):
            img_dir = os.path.join(split_dir, seq, "Images")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                fp = os.path.join(img_dir, fn)
                if os.path.isfile(fp) and is_image(fp):
                    samples.append({
                        "rgb_path": fp,
                        "ir_path": None,
                        "dataset": "UAVid",
                        "type": "rgb_only",
                        "ir_converted": False,
                    })

    write_manifest(os.path.join(out_dir, "UAVid_all.json"),
                   "UAVid", "rgb_only", samples)


def gen_udd(data_root: str, out_dir: str):
    """UDD — rgb_only (only src/, exclude gt/)."""
    base = os.path.join(data_root, "UDD", "UDD")
    samples = []

    # UDD5 and UDD6: {train,val}/src/
    for variant in ("UDD5", "UDD6"):
        for split in ("train", "val"):
            src_dir = os.path.join(base, variant, split, "src")
            if not os.path.isdir(src_dir):
                continue
            for fn in sorted(os.listdir(src_dir)):
                fp = os.path.join(src_dir, fn)
                if os.path.isfile(fp) and is_image(fp):
                    samples.append({
                        "rgb_path": fp,
                        "ir_path": None,
                        "dataset": "UDD",
                        "type": "rgb_only",
                        "ir_converted": False,
                    })

    # m1-only-src/ (additional images)
    m1_dir = os.path.join(data_root, "UDD", "m1-only-src")
    if os.path.isdir(m1_dir):
        for fn in sorted(os.listdir(m1_dir)):
            fp = os.path.join(m1_dir, fn)
            if os.path.isfile(fp) and is_image(fp):
                samples.append({
                    "rgb_path": fp,
                    "ir_path": None,
                    "dataset": "UDD",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "UDD_all.json"),
                   "UDD", "rgb_only", samples)


def gen_stanford_drone(data_root: str, out_dir: str):
    """Stanford Drone Dataset — rgb_only (only reference.jpg images)."""
    base = os.path.join(data_root, "Stanford Drone Dataset", "archive",
                        "annotations")
    samples = []
    if not os.path.isdir(base):
        print(f"  WARNING: {base} not found")
        return

    for scene in sorted(os.listdir(base)):
        scene_dir = os.path.join(base, scene)
        if not os.path.isdir(scene_dir):
            continue
        for video in sorted(os.listdir(scene_dir)):
            ref = os.path.join(scene_dir, video, "reference.jpg")
            if os.path.isfile(ref):
                samples.append({
                    "rgb_path": ref,
                    "ir_path": None,
                    "dataset": "StanfordDrone",
                    "type": "rgb_only",
                    "ir_converted": False,
                })

    write_manifest(os.path.join(out_dir, "StanfordDrone_all.json"),
                   "StanfordDrone", "rgb_only", samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate manifests for new datasets")
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/data",
                        help="Root directory containing all datasets")
    parser.add_argument("--out_dir", type=str,
                        default="/root/autodl-tmp/train1/manifests",
                        help="Output directory for manifest JSON files")
    parser.add_argument("--ir_out_root", type=str,
                        default="/root/autodl-tmp/train1/converted_ir",
                        help="Root directory for converted IR images")
    args = parser.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir
    ir_out_root = args.ir_out_root

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Generating manifests for new datasets")
    print(f"  data_root:   {data_root}")
    print(f"  out_dir:     {out_dir}")
    print(f"  ir_out_root: {ir_out_root}")
    print("=" * 60)

    generators = [
        ("1/15 LLVIP unregistered (paired)",
         lambda: gen_llvip_unreg(data_root, out_dir, ir_out_root)),
        ("2/15 BIRDSAI (ir_only)",
         lambda: gen_birdsai(data_root, out_dir)),
        ("3/15 LSOTB-TIR (ir_only)",
         lambda: gen_lsotb_tir(data_root, out_dir)),
        ("4/15 SeaDronesSee MOT (rgb_only)",
         lambda: gen_seadronessee_mot(data_root, out_dir)),
        ("5/15 CVOGL drone (rgb_only)",
         lambda: gen_cvogl_drone(data_root, out_dir)),
        ("6/15 University-1652 drone (rgb_only)",
         lambda: gen_university1652_drone(data_root, out_dir)),
        ("7/15 SUES-200 drone (rgb_only)",
         lambda: gen_sues200_drone(data_root, out_dir)),
        ("8/15 UAV-Human (rgb_only)",
         lambda: gen_uav_human(data_root, out_dir)),
        ("9/15 UAVScenes (rgb_only)",
         lambda: gen_uavscenes(data_root, out_dir)),
        ("10/15 MDMT (rgb_only)",
         lambda: gen_mdmt(data_root, out_dir)),
        ("11/15 Manipal-UAV (rgb_only)",
         lambda: gen_manipal_uav(data_root, out_dir)),
        ("12/15 DTB70 (rgb_only)",
         lambda: gen_dtb70(data_root, out_dir)),
        ("13/15 UAVid (rgb_only)",
         lambda: gen_uavid(data_root, out_dir)),
        ("14/15 UDD (rgb_only)",
         lambda: gen_udd(data_root, out_dir)),
        ("15/15 Stanford Drone (rgb_only)",
         lambda: gen_stanford_drone(data_root, out_dir)),
    ]

    for label, gen_fn in generators:
        print(f"\n[{label}]")
        gen_fn()

    print("\n" + "=" * 60)
    print("All manifests generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

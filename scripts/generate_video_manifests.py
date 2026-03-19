"""
Generate video manifest JSON files for all frame-extracted datasets.

Supports:
- UAV-Human ActionRecognition (.avi videos)
- aerial-rgbt (paired color/thermal16 image sequences)
- 18 frame-extracted video datasets (RGB-IR / RGB-only / IR-only)

Usage:
    python scripts/generate_video_manifests.py \
        --data_root /root/autodl-tmp/data \
        --output_dir /root/autodl-tmp/data/manifests \
        --min_frames 4
"""

import argparse
import json
import os
import re
from statistics import median
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _natural_sort_key(name):
    parts = re.split(r'(\d+)', name)
    key = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def list_images(directory):
    """List image file names in natural order."""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    files.sort(key=_natural_sort_key)
    return files


def infer_frame_values(file_names):
    """Infer frame ids from file names."""
    if not file_names:
        return None
    stems = [os.path.splitext(f)[0] for f in file_names]

    # If pure numeric, parse as float.
    pure = []
    pure_ok = True
    for s in stems:
        if re.fullmatch(r'\d+(?:\.\d+)?', s):
            pure.append(float(s))
        else:
            pure_ok = False
            break
    if pure_ok:
        return pure

    # Otherwise parse the last integer token.
    out = []
    for s in stems:
        nums = re.findall(r'(\d+)', s)
        if not nums:
            return None
        out.append(float(nums[-1]))
    return out


def estimate_stride_from_names(file_names):
    """Estimate nominal frame stride from file names."""
    vals = infer_frame_values(file_names)
    if vals is None or len(vals) < 2:
        return 1.0
    diffs = []
    for a, b in zip(vals, vals[1:]):
        d = b - a
        if d > 0:
            diffs.append(d)
    if not diffs:
        return 1.0
    return float(median(diffs))


def make_entry(dataset, modality, num_frames, fps, sequence, **kwargs):
    entry = {
        "dataset": dataset,
        "modality": modality,
        "num_frames": int(num_frames),
        "fps": float(fps),
        "sequence": str(sequence),
    }
    entry.update(kwargs)
    return entry


def count_images(directory):
    """Count image files in a directory without loading them."""
    return len(list_images(directory))


def get_video_info(video_path):
    """Get fps and frame count from a video file."""
    if not HAS_CV2:
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


# ============================================================================
# RGB-IR Paired datasets
# ============================================================================

def scan_utuav(data_root, min_frames=4):
    """UTUAV: <seq>/rgb/*.jpg + <seq>/ir/*.jpg"""
    root = Path(data_root) / "UTUAV"
    if not root.exists():
        return []

    manifest = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        rgb_dir = seq_dir / "rgb"
        ir_dir = seq_dir / "ir"
        if not rgb_dir.exists() or not ir_dir.exists():
            continue
        n_rgb = count_images(str(rgb_dir))
        n_ir = count_images(str(ir_dir))
        n = min(n_rgb, n_ir)
        if n < min_frames:
            continue
        manifest.append({
            "dataset": "utuav",
            "modality": "rgbir_paired",
            "rgb_dir": str(rgb_dir),
            "ir_dir": str(ir_dir),
            "num_frames": n,
            "fps": 30.0,
            "sequence": seq_dir.name,
        })

    print(f"UTUAV: {len(manifest)} sequences")
    return manifest


def scan_rgbt_tiny(data_root, min_frames=4):
    """RGBT-Tiny: images/<seq>/00/*.jpg (RGB) + images/<seq>/01/*.jpg (IR)"""
    root = Path(data_root) / "RGBT-Tiny" / "images"
    if not root.exists():
        return []

    manifest = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        rgb_dir = seq_dir / "00"
        ir_dir = seq_dir / "01"
        if not rgb_dir.exists() or not ir_dir.exists():
            continue
        n_rgb = count_images(str(rgb_dir))
        n_ir = count_images(str(ir_dir))
        n = min(n_rgb, n_ir)
        if n < min_frames:
            continue
        manifest.append({
            "dataset": "rgbt_tiny",
            "modality": "rgbir_paired",
            "rgb_dir": str(rgb_dir),
            "ir_dir": str(ir_dir),
            "num_frames": n,
            "fps": 25.0,
            "sequence": seq_dir.name,
        })

    print(f"RGBT-Tiny: {len(manifest)} sequences")
    return manifest


def scan_vt_tiny_mot(data_root, min_frames=4):
    """VT-Tiny-MOT: train2017/<seq>/00/*.jpg + train2017/<seq>/01/*.jpg"""
    manifest = []
    for split in ('train2017', 'test2017'):
        root = Path(data_root) / "VT-Tiny-MOT" / split
        if not root.exists():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            rgb_dir = seq_dir / "00"
            ir_dir = seq_dir / "01"
            if not rgb_dir.exists() or not ir_dir.exists():
                continue
            n_rgb = count_images(str(rgb_dir))
            n_ir = count_images(str(ir_dir))
            n = min(n_rgb, n_ir)
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "vt_tiny_mot",
                "modality": "rgbir_paired",
                "rgb_dir": str(rgb_dir),
                "ir_dir": str(ir_dir),
                "num_frames": n,
                "fps": 25.0,
                "sequence": seq_dir.name,
            })

    print(f"VT-Tiny-MOT: {len(manifest)} sequences")
    return manifest


def scan_m3ot(data_root, min_frames=4):
    """M3OT: 2/rgb/{train,test}/<seq>/img1/*.PNG + 2/ir/{train,test}/<seq>/img1/*.PNG"""
    root = Path(data_root) / "M3OT"
    if not root.exists():
        return []

    manifest = []
    for modality_set in ('1', '2'):
        base = root / modality_set
        if not base.exists():
            continue

        # Check structure: could be rgb/train/<seq>/img1/ or ir/<seq>/
        rgb_base = base / "rgb"
        ir_base = base / "ir"
        if not rgb_base.exists() or not ir_base.exists():
            # M3OT/1 only has ir
            if ir_base.exists() and not rgb_base.exists():
                # ir-only sequences in set 1
                for split in ('train', 'val', 'test'):
                    split_dir = ir_base / split
                    if not split_dir.exists():
                        continue
                    for seq_dir in sorted(split_dir.iterdir()):
                        if not seq_dir.is_dir():
                            continue
                        img_dir = seq_dir / "img1"
                        if not img_dir.exists():
                            img_dir = seq_dir
                        n = count_images(str(img_dir))
                        if n < min_frames:
                            continue
                        files = list_images(str(img_dir))
                        stride = estimate_stride_from_names(files)
                        manifest.append(make_entry(
                            dataset="m3ot",
                            modality="ir_only",
                            frame_dir=str(img_dir),
                            num_frames=n,
                            fps=30.0,
                            sequence=f"{split}_{seq_dir.name}",
                            sequence_stride=max(1.0, stride),
                            already_subsampled=bool(stride > 1.5),
                        ))
            continue

        for split in ('train', 'val', 'test'):
            rgb_split = rgb_base / split
            ir_split = ir_base / split
            if not rgb_split.exists() or not ir_split.exists():
                continue
            for seq_dir in sorted(rgb_split.iterdir()):
                if not seq_dir.is_dir():
                    continue
                rgb_img = seq_dir / "img1"
                if not rgb_img.exists():
                    rgb_img = seq_dir
                # Find matching IR sequence (may have T suffix for test)
                ir_seq_name = seq_dir.name
                ir_seq = ir_split / ir_seq_name / "img1"
                if not ir_seq.exists():
                    ir_seq = ir_split / (ir_seq_name + "T") / "img1"
                if not ir_seq.exists():
                    ir_seq = ir_split / ir_seq_name
                if not ir_seq.exists():
                    continue

                n_rgb = count_images(str(rgb_img))
                n_ir = count_images(str(ir_seq))
                n = min(n_rgb, n_ir)
                if n < min_frames:
                    continue
                rgb_files = list_images(str(rgb_img))
                stride = estimate_stride_from_names(rgb_files)
                manifest.append(make_entry(
                    dataset="m3ot",
                    modality="rgbir_paired",
                    rgb_dir=str(rgb_img),
                    ir_dir=str(ir_seq),
                    num_frames=n,
                    fps=30.0,
                    sequence=f"{split}_{seq_dir.name}",
                    sequence_stride=max(1.0, stride),
                    already_subsampled=bool(stride > 1.5),
                ))

    print(f"M3OT: {len(manifest)} sequences")
    return manifest


def scan_tardal(data_root, min_frames=4):
    """TarDAL paired frame directories (already frame-subsampled)."""
    root = Path(data_root) / "TarDAL"
    if not root.exists():
        return []

    candidates = [
        ("M3FD_Detection", root / "M3FD_Detection" / "vi", root / "M3FD_Detection" / "ir", 30.0),
        ("M3FD_Fusion", root / "M3FD_Fusion" / "Vis", root / "M3FD_Fusion" / "Ir", 30.0),
        ("roadscene", root / "roadscene" / "vi", root / "roadscene" / "ir", 25.0),
        ("tno", root / "tno" / "vi", root / "tno" / "ir", 25.0),
    ]

    manifest = []
    for name, rgb_dir, ir_dir, fps in candidates:
        if not rgb_dir.exists() or not ir_dir.exists():
            continue
        n_rgb = count_images(str(rgb_dir))
        n_ir = count_images(str(ir_dir))
        n = min(n_rgb, n_ir)
        if n < min_frames:
            continue
        files = list_images(str(rgb_dir))
        stride = estimate_stride_from_names(files)
        stride = max(5.0, stride)
        manifest.append(make_entry(
            dataset="tardal",
            modality="rgbir_paired",
            rgb_dir=str(rgb_dir),
            ir_dir=str(ir_dir),
            num_frames=n,
            fps=fps,
            sequence=name,
            sequence_stride=stride,
            already_subsampled=True,
            sample_min_delta=1,
            sample_max_delta=12,
        ))

    print(f"TarDAL: {len(manifest)} sequences")
    return manifest


# ============================================================================
# RGB-only datasets
# ============================================================================

def scan_aeroscapes(data_root, min_frames=4):
    """AeroScapes: JPEGImages/<seq>_<frame>.jpg"""
    root = Path(data_root) / "AeroScapes" / "JPEGImages"
    if not root.exists():
        return []

    files = list_images(str(root))
    groups = {}
    for fn in files:
        stem = Path(fn).stem
        m = re.match(r'(.+)_([0-9]+)$', stem)
        if not m:
            continue
        seq = m.group(1)
        groups.setdefault(seq, []).append(fn)

    manifest = []
    for seq, seq_files in sorted(groups.items()):
        n = len(seq_files)
        if n < min_frames:
            continue
        stride = estimate_stride_from_names(seq_files)
        manifest.append(make_entry(
            dataset="aeroscapes",
            modality="rgb_only",
            frame_dir=str(root),
            filename_prefix=f"{seq}_",
            num_frames=n,
            fps=30.0,
            sequence=seq,
            sequence_stride=max(1.0, stride),
            already_subsampled=bool(stride > 1.5),
        ))

    print(f"AeroScapes: {len(manifest)} sequences")
    return manifest


def scan_au_air(data_root, min_frames=4):
    """AU-AIR: auair2019data/frame_<session>_<view>_<frame>.jpg"""
    root = Path(data_root) / "AU-AIR" / "auair2019data"
    if not root.exists():
        return []

    files = list_images(str(root))
    groups = {}
    for fn in files:
        stem = Path(fn).stem
        m = re.match(r'(.+)_([0-9]+)$', stem)
        key = m.group(1) if m else stem
        groups.setdefault(key, []).append(fn)

    manifest = []
    for key, seq_files in sorted(groups.items()):
        n = len(seq_files)
        if n < min_frames:
            continue
        stride = estimate_stride_from_names(seq_files)
        manifest.append(make_entry(
            dataset="au_air",
            modality="rgb_only",
            frame_dir=str(root),
            filename_prefix=f"{key}_",
            num_frames=n,
            fps=30.0,
            sequence=key,
            sequence_stride=max(1.0, stride),
            already_subsampled=bool(stride > 1.5),
        ))

    print(f"AU-AIR: {len(manifest)} sequences")
    return manifest


def _scan_animaldrone_cut_dirs(base_dir):
    out = []
    if not os.path.isdir(base_dir):
        return out
    for cur, dirs, _ in os.walk(base_dir):
        dirs.sort(key=_natural_sort_key)
        n = count_images(cur)
        if n > 0:
            out.append(cur)
    return out


def scan_animaldrone(data_root, min_frames=4):
    """AnimalDrone: PartA flat frames + PartB cut sequences."""
    root = Path(data_root) / "AnimalDrone"
    if not root.exists():
        return []

    manifest = []

    # PartA as two large frame sequences.
    part_a_dirs = [
        root / "AnimalDrone_PartA" / "jpg-train" / "jpg",
        root / "AnimalDrone_PartA" / "jpg-test" / "jpg",
    ]
    for d in part_a_dirs:
        if not d.exists():
            continue
        files = list_images(str(d))
        n = len(files)
        if n < min_frames:
            continue
        stride = estimate_stride_from_names(files)
        manifest.append(make_entry(
            dataset="animaldrone",
            modality="rgb_only",
            frame_dir=str(d),
            num_frames=n,
            fps=30.0,
            sequence=d.name + "_" + d.parent.name,
            sequence_stride=max(1.0, stride),
            already_subsampled=bool(stride > 1.5),
        ))

    # PartB uses explicit cut directories, each cut is one video sequence.
    for sub in ("jpg", "jpg2", "jpg3", "jpg4"):
        cut_root = root / "AnimalDrone_PartB" / sub
        for cut_dir in _scan_animaldrone_cut_dirs(str(cut_root)):
            files = list_images(cut_dir)
            n = len(files)
            if n < min_frames:
                continue
            stride = estimate_stride_from_names(files)
            rel = os.path.relpath(cut_dir, str(root)).replace("/", "_")
            manifest.append(make_entry(
                dataset="animaldrone",
                modality="rgb_only",
                frame_dir=cut_dir,
                num_frames=n,
                fps=30.0,
                sequence=rel,
                sequence_stride=max(1.0, stride),
                already_subsampled=bool(stride > 1.5),
            ))

    print(f"AnimalDrone: {len(manifest)} sequences")
    return manifest

def scan_webuav_3m(data_root, min_frames=4):
    """WebUAV-3M: {Train,Val,Test}/<seq>/img/*.jpg"""
    root = Path(data_root) / "WebUAV-3M"
    if not root.exists():
        return []

    manifest = []
    for split in ('Train', 'Val', 'Test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            img_dir = seq_dir / "img"
            if not img_dir.exists():
                continue
            n = count_images(str(img_dir))
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "webuav_3m",
                "modality": "rgb_only",
                "frame_dir": str(img_dir),
                "num_frames": n,
                "fps": 30.0,
                "sequence": seq_dir.name,
            })

    print(f"WebUAV-3M: {len(manifest)} sequences")
    return manifest


def scan_sues200(data_root, min_frames=4):
    """SUES-200: drone_view_512/<id>/<scale>/*.jpg (already frame-subsampled)."""
    root = Path(data_root) / "SUES-200" / "drone_view_512"
    if not root.exists():
        return []

    manifest = []
    for id_dir in sorted(root.iterdir()):
        if not id_dir.is_dir():
            continue
        for scale_dir in sorted(id_dir.iterdir()):
            if not scale_dir.is_dir():
                continue
            n = count_images(str(scale_dir))
            if n < min_frames:
                continue
            files = list_images(str(scale_dir))
            stride = estimate_stride_from_names(files)
            stride = max(5.0, stride)
            manifest.append(make_entry(
                dataset="sues_200",
                modality="rgb_only",
                frame_dir=str(scale_dir),
                num_frames=n,
                fps=30.0,
                sequence=f"{id_dir.name}_{scale_dir.name}",
                sequence_stride=stride,
                already_subsampled=True,
                sample_min_delta=1,
                sample_max_delta=12,
            ))

    print(f"SUES-200: {len(manifest)} sequences")
    return manifest


def scan_uavid(data_root, min_frames=4):
    """UAVid: {train,val,test}/seq*/Images/*.png (already frame-subsampled)."""
    root = Path(data_root) / "UAVid"
    if not root.exists():
        return []

    manifest = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            img_dir = seq_dir / "Images"
            if not img_dir.exists():
                continue
            files = list_images(str(img_dir))
            n = len(files)
            if n < min_frames:
                continue
            stride = estimate_stride_from_names(files)
            stride = max(5.0, stride)
            manifest.append(make_entry(
                dataset="uavid",
                modality="rgb_only",
                frame_dir=str(img_dir),
                num_frames=n,
                fps=30.0,
                sequence=f"{split}_{seq_dir.name}",
                sequence_stride=stride,
                already_subsampled=True,
                sample_min_delta=1,
                sample_max_delta=8,
            ))

    print(f"UAVid: {len(manifest)} sequences")
    return manifest


def scan_uavscenes(data_root, min_frames=4):
    """UAVScenes: interval=5/interval5_CAM_LIDAR/*/interval5_CAM/*.jpg."""
    root = Path(data_root) / "UAVScenes" / "interval=5" / "interval5_CAM_LIDAR"
    if not root.exists():
        return []

    manifest = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        cam_dir = scene_dir / "interval5_CAM"
        if not cam_dir.exists():
            continue
        n = count_images(str(cam_dir))
        if n < min_frames:
            continue
        # interval=5 means the source stream is already decimated.
        manifest.append(make_entry(
            dataset="uavscenes",
            modality="rgb_only",
            frame_dir=str(cam_dir),
            num_frames=n,
            fps=2.0,
            sequence=scene_dir.name,
            sequence_stride=5.0,
            already_subsampled=True,
            sample_min_delta=1,
            sample_max_delta=8,
        ))

    print(f"UAVScenes: {len(manifest)} sequences")
    return manifest


def _scan_visdrone_sequences(seq_root, dataset_name, min_frames=4):
    """Generic scanner for VisDrone sequence directories."""
    if not seq_root.exists():
        return []
    manifest = []
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        n = count_images(str(seq_dir))
        if n < min_frames:
            continue
        manifest.append({
            "dataset": dataset_name,
            "modality": "rgb_only",
            "frame_dir": str(seq_dir),
            "num_frames": n,
            "fps": 30.0,
            "sequence": seq_dir.name,
        })
    return manifest


def scan_visdrone_sot(data_root, min_frames=4):
    """VisDrone SOT: Single-Object Tracking/*/VisDrone2019-SOT-*/sequences/<seq>/*.jpg"""
    root = Path(data_root) / "VisDrone" / "Single-Object Tracking"
    if not root.exists():
        return []

    manifest = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir():
            continue
        for sub_dir in sorted(part_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            seq_root = sub_dir / "sequences"
            if seq_root.exists():
                manifest.extend(_scan_visdrone_sequences(
                    seq_root, "visdrone_sot", min_frames))

    print(f"VisDrone-SOT: {len(manifest)} sequences")
    return manifest


def scan_visdrone_mot(data_root, min_frames=4):
    """VisDrone MOT: Multi-Object Tracking/VisDrone2019-MOT-*/sequences/<seq>/*.jpg"""
    root = Path(data_root) / "VisDrone" / "Multi-Object Tracking"
    if not root.exists():
        return []

    manifest = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir():
            continue
        seq_root = part_dir / "sequences"
        if seq_root.exists():
            manifest.extend(_scan_visdrone_sequences(
                seq_root, "visdrone_mot", min_frames))

    print(f"VisDrone-MOT: {len(manifest)} sequences")
    return manifest


def scan_visdrone_cc(data_root, min_frames=4):
    """VisDrone CC: Crowd Counting/sequences/<seq>/*.jpg"""
    root = Path(data_root) / "VisDrone" / "Crowd Counting" / "sequences"
    if not root.exists():
        return []

    manifest = _scan_visdrone_sequences(root, "visdrone_cc", min_frames)
    print(f"VisDrone-CC: {len(manifest)} sequences")
    return manifest


def scan_uav123(data_root, min_frames=4):
    """UAV123: data_seq/<seq>/*.jpg"""
    root = Path(data_root) / "UAV123" / "data_seq"
    if not root.exists():
        return []

    manifest = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        n = count_images(str(seq_dir))
        if n < min_frames:
            continue
        manifest.append({
            "dataset": "uav123",
            "modality": "rgb_only",
            "frame_dir": str(seq_dir),
            "num_frames": n,
            "fps": 30.0,
            "sequence": seq_dir.name,
        })

    print(f"UAV123: {len(manifest)} sequences")
    return manifest


def scan_uavdt(data_root, min_frames=4):
    """UAVDT: UAV-benchmark-{S,M}/<seq>/img*.jpg"""
    manifest = []
    for subset in ("UAV-benchmark-S", "UAV-benchmark-M"):
        root = Path(data_root) / "UAVDT" / subset
        if not root.exists():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            n = count_images(str(seq_dir))
            if n < min_frames:
                continue
            files = list_images(str(seq_dir))
            stride = estimate_stride_from_names(files)
            manifest.append(make_entry(
                dataset="uavdt",
                modality="rgb_only",
                frame_dir=str(seq_dir),
                num_frames=n,
                fps=30.0,
                sequence=f"{subset}_{seq_dir.name}",
                sequence_stride=max(1.0, stride),
                already_subsampled=bool(stride > 1.5),
            ))

    print(f"UAVDT: {len(manifest)} sequences")
    return manifest


def scan_dronecrowd(data_root, min_frames=4):
    """DroneCrowd: {train,val,test}_data/images + VisDrone2020-CC"""
    manifest = []
    for sub in ('train_data/images', 'val_data/images', 'test_data/images', 'VisDrone2020-CC'):
        img_dir = Path(data_root) / "DroneCrowd" / sub
        if not img_dir.exists():
            continue
        n = count_images(str(img_dir))
        if n < min_frames:
            continue
        files = list_images(str(img_dir))
        stride = estimate_stride_from_names(files)
        manifest.append(make_entry(
            dataset="dronecrowd",
            modality="rgb_only",
            frame_dir=str(img_dir),
            num_frames=n,
            fps=30.0,
            sequence=sub.replace('/', '_'),
            sequence_stride=max(1.0, stride),
            already_subsampled=bool(stride > 1.5),
        ))

    print(f"DroneCrowd: {len(manifest)} sequences")
    return manifest


def scan_dtb70(data_root, min_frames=4):
    """DTB70: <seq>/img/*.jpg"""
    root = Path(data_root) / "DTB70"
    if not root.exists():
        return []

    manifest = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        img_dir = seq_dir / "img"
        if not img_dir.exists():
            continue
        n = count_images(str(img_dir))
        if n < min_frames:
            continue
        manifest.append({
            "dataset": "dtb70",
            "modality": "rgb_only",
            "frame_dir": str(img_dir),
            "num_frames": n,
            "fps": 30.0,
            "sequence": seq_dir.name,
        })

    print(f"DTB70: {len(manifest)} sequences")
    return manifest


def scan_mdmt(data_root, min_frames=4):
    """MDMT: {train,val,test}/<seq>/*.jpg (may have nested subdirs)"""
    root = Path(data_root) / "MDMT"
    if not root.exists():
        return []

    manifest = []
    for split in ('train', 'val', 'test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            # MDMT has nested structure: train/1/ contains both .jpg and sub-dirs
            n = count_images(str(seq_dir))
            if n >= min_frames:
                manifest.append({
                    "dataset": "mdmt",
                    "modality": "rgb_only",
                    "frame_dir": str(seq_dir),
                    "num_frames": n,
                    "fps": 30.0,
                    "sequence": seq_dir.name,
                })
            # Also check sub-directories
            for sub in sorted(seq_dir.iterdir()):
                if not sub.is_dir():
                    continue
                n_sub = count_images(str(sub))
                if n_sub >= min_frames:
                    manifest.append({
                        "dataset": "mdmt",
                        "modality": "rgb_only",
                        "frame_dir": str(sub),
                        "num_frames": n_sub,
                        "fps": 30.0,
                        "sequence": f"{seq_dir.name}_{sub.name}",
                    })

    print(f"MDMT: {len(manifest)} sequences")
    return manifest


def scan_seadronessee_mot(data_root, min_frames=4):
    """SeaDronesSee-MOT: SeaDronesSee_MOT_*/SeaDronesSee_MOT/Compressed/{train,val,test}/<seq>/*.jpg"""
    root = (Path(data_root) / "SeaDronesSee" /
            "SeaDronesSee_MOT_Multi-Object Tracking" /
            "SeaDronesSee_MOT" / "Compressed")
    if not root.exists():
        return []

    manifest = []
    for split in ('train', 'val', 'test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            n = count_images(str(seq_dir))
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "seadronessee_mot",
                "modality": "rgb_only",
                "frame_dir": str(seq_dir),
                "num_frames": n,
                "fps": 30.0,
                "sequence": seq_dir.name,
            })

    print(f"SeaDronesSee-MOT: {len(manifest)} sequences")
    return manifest


# ============================================================================
# IR-only datasets
# ============================================================================

def scan_lsotb_tir(data_root, min_frames=4):
    """LSOTB-TIR: Training Dataset/TrainingData/<group>/<seq>/*.jpg"""
    root = Path(data_root) / "LSOTB-TIR" / "Training Dataset" / "TrainingData"
    if not root.exists():
        return []

    manifest = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        # Each group may contain sequences directly or be a sequence itself
        n_direct = count_images(str(group_dir))
        if n_direct >= min_frames:
            manifest.append({
                "dataset": "lsotb_tir",
                "modality": "ir_only",
                "frame_dir": str(group_dir),
                "num_frames": n_direct,
                "fps": 30.0,
                "sequence": group_dir.name,
            })
        # Check sub-sequences
        for seq_dir in sorted(group_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            n = count_images(str(seq_dir))
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "lsotb_tir",
                "modality": "ir_only",
                "frame_dir": str(seq_dir),
                "num_frames": n,
                "fps": 30.0,
                "sequence": f"{group_dir.name}_{seq_dir.name}",
            })

    print(f"LSOTB-TIR: {len(manifest)} sequences")
    return manifest


def scan_birdsai(data_root, min_frames=4):
    """BIRDSAI: TrainReal/images/<seq>/*.jpg"""
    manifest = []
    for split in ('TrainReal', 'TestReal'):
        root = Path(data_root) / "BIRDSAI" / split / "images"
        if not root.exists():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            n = count_images(str(seq_dir))
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "birdsai",
                "modality": "ir_only",
                "frame_dir": str(seq_dir),
                "num_frames": n,
                "fps": 30.0,
                "sequence": seq_dir.name,
            })

    print(f"BIRDSAI: {len(manifest)} sequences")
    return manifest


def scan_monet(data_root, min_frames=4):
    """MONET: monet_{train,val,test}/<name>/<name>/lwir/*.png"""
    root = Path(data_root) / "MONET"
    if not root.exists():
        return []

    manifest = []
    for split in ('monet_train', 'monet_val', 'monet_test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for outer_dir in sorted(split_dir.iterdir()):
            if not outer_dir.is_dir():
                continue
            # Structure: outer/<same_name>/lwir/
            inner = outer_dir / outer_dir.name
            if not inner.exists():
                inner = outer_dir
            lwir_dir = inner / "lwir"
            if not lwir_dir.exists():
                continue
            n = count_images(str(lwir_dir))
            if n < min_frames:
                continue
            manifest.append({
                "dataset": "monet",
                "modality": "ir_only",
                "frame_dir": str(lwir_dir),
                "num_frames": n,
                "fps": 30.0,
                "sequence": outer_dir.name,
            })

    print(f"MONET: {len(manifest)} sequences")
    return manifest


# ============================================================================
# Original dataset scanners (UAV-Human video, aerial-rgbt)
# ============================================================================

def generate_uav_human_manifest(root_dir, min_ir_frames=4):
    """Generate manifest for UAV-Human paired RGB+IR .avi videos."""
    root = Path(root_dir)
    rgb_dir = root / "RGBVideos"
    ir_dir = root / "NightvisionVideos"

    if not rgb_dir.exists() or not ir_dir.exists():
        print(f"UAV-Human: directory not found ({root})")
        return []

    ir_files = {f.name: f for f in ir_dir.iterdir() if f.suffix == '.avi'}

    manifest = []
    skipped = 0
    no_pair = 0

    for rgb_path in sorted(rgb_dir.iterdir()):
        if rgb_path.suffix != '.avi':
            continue
        ir_path = ir_files.get(rgb_path.name)
        if ir_path is None:
            no_pair += 1
            continue

        ir_fps, ir_frame_count = get_video_info(str(ir_path))
        if ir_frame_count is None or ir_frame_count < min_ir_frames:
            skipped += 1
            continue

        rgb_fps, rgb_frame_count = get_video_info(str(rgb_path))
        if rgb_frame_count is None:
            skipped += 1
            continue

        manifest.append({
            "rgb_video_path": str(rgb_path),
            "ir_video_path": str(ir_path),
            "rgb_fps": rgb_fps,
            "ir_fps": ir_fps,
            "rgb_frame_count": rgb_frame_count,
            "ir_frame_count": ir_frame_count,
            "dataset": "uav_human",
        })

    print(f"UAV-Human: {len(manifest)} paired clips, "
          f"{skipped} skipped, {no_pair} unpaired")
    return manifest


def generate_aerial_rgbt_manifest(root_dir, min_frames=4):
    """Generate manifest for aerial-rgbt image sequences."""
    root = Path(root_dir)
    if not root.exists():
        print(f"aerial-rgbt: {root} not found")
        return []

    manifest = []
    skipped = 0

    for location_dir in sorted(root.iterdir()):
        if not location_dir.is_dir():
            continue
        for seq_dir in sorted(location_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            color_dir = seq_dir / "color"
            thermal_dir = seq_dir / "thermal16"

            if not color_dir.exists() or not thermal_dir.exists():
                continue

            color_files = {}
            for f in color_dir.iterdir():
                m = re.match(r'pair-(\d+)\.png', f.name)
                if m:
                    color_files[int(m.group(1))] = str(f)

            thermal_files = {}
            for f in thermal_dir.iterdir():
                m = re.match(r'thermal-(\d+)\.tiff', f.name)
                if m:
                    thermal_files[int(m.group(1))] = str(f)

            common_indices = sorted(
                set(color_files.keys()) & set(thermal_files.keys()))

            if len(common_indices) < min_frames:
                skipped += 1
                continue

            paired_frames = []
            for idx in common_indices:
                paired_frames.append({
                    "rgb_path": color_files[idx],
                    "ir_path": thermal_files[idx],
                    "frame_index": idx,
                })

            manifest.append({
                "sequence_dir": str(seq_dir),
                "paired_frames": paired_frames,
                "num_frames": len(paired_frames),
                "dataset": "aerial_rgbt",
            })

    print(f"aerial-rgbt: {len(manifest)} sequences, {skipped} skipped")
    return manifest


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser("Generate video manifests for all datasets")
    parser.add_argument("--data_root",
                        default="/root/autodl-tmp/data",
                        help="Root directory containing all datasets")
    parser.add_argument("--output_dir",
                        default="/root/autodl-tmp/data/manifests",
                        help="Output directory for manifest JSON files")
    parser.add_argument("--min_frames", type=int, default=4,
                        help="Minimum frames per sequence")
    parser.add_argument("--include_original", action="store_true",
                        help="Also generate UAV-Human .avi and aerial-rgbt manifests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_root = args.data_root
    min_f = args.min_frames

    # ---- Frame-extracted datasets ----
    print("=" * 60)
    print("Scanning frame-extracted video datasets...")
    print("=" * 60)

    all_frame_seq = []

    # RGB-IR paired
    all_frame_seq.extend(scan_utuav(data_root, min_f))
    all_frame_seq.extend(scan_rgbt_tiny(data_root, min_f))
    all_frame_seq.extend(scan_vt_tiny_mot(data_root, min_f))
    all_frame_seq.extend(scan_m3ot(data_root, min_f))
    all_frame_seq.extend(scan_tardal(data_root, min_f))

    # RGB-only
    all_frame_seq.extend(scan_aeroscapes(data_root, min_f))
    all_frame_seq.extend(scan_animaldrone(data_root, min_f))
    all_frame_seq.extend(scan_au_air(data_root, min_f))
    all_frame_seq.extend(scan_webuav_3m(data_root, min_f))
    all_frame_seq.extend(scan_sues200(data_root, min_f))
    all_frame_seq.extend(scan_uavid(data_root, min_f))
    all_frame_seq.extend(scan_uavscenes(data_root, min_f))
    all_frame_seq.extend(scan_visdrone_sot(data_root, min_f))
    all_frame_seq.extend(scan_visdrone_mot(data_root, min_f))
    all_frame_seq.extend(scan_visdrone_cc(data_root, min_f))
    all_frame_seq.extend(scan_uav123(data_root, min_f))
    all_frame_seq.extend(scan_uavdt(data_root, min_f))
    all_frame_seq.extend(scan_dronecrowd(data_root, min_f))
    all_frame_seq.extend(scan_dtb70(data_root, min_f))
    all_frame_seq.extend(scan_mdmt(data_root, min_f))
    all_frame_seq.extend(scan_seadronessee_mot(data_root, min_f))

    # IR-only
    all_frame_seq.extend(scan_lsotb_tir(data_root, min_f))
    all_frame_seq.extend(scan_birdsai(data_root, min_f))
    all_frame_seq.extend(scan_monet(data_root, min_f))

    # Save frame-sequence manifest
    frame_seq_path = os.path.join(args.output_dir, "frame_sequences.json")
    with open(frame_seq_path, 'w') as f:
        json.dump(all_frame_seq, f, indent=2)

    # Summary
    from collections import Counter
    ds_counts = Counter(e['dataset'] for e in all_frame_seq)
    mod_counts = Counter(e['modality'] for e in all_frame_seq)
    total_frames = sum(e.get('num_frames', 0) for e in all_frame_seq)

    print(f"\n{'=' * 60}")
    print(f"Frame-sequence manifest: {frame_seq_path}")
    print(f"  Total sequences: {len(all_frame_seq)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  By modality: {dict(mod_counts)}")
    print(f"  By dataset: {dict(ds_counts)}")
    print(f"{'=' * 60}")

    # ---- Original video datasets (optional) ----
    if args.include_original:
        print("\nScanning original video datasets...")
        uav_manifest = generate_uav_human_manifest(
            os.path.join(data_root, "UAV-Human/ActionRecognition"),
            min_ir_frames=min_f)
        aerial_manifest = generate_aerial_rgbt_manifest(
            os.path.join(data_root, "aerial-rgbt/labeled_thermal_singles"),
            min_frames=min_f)

        # Save individual manifests
        for name, data in [("uav_human_video", uav_manifest),
                           ("aerial_rgbt_video", aerial_manifest)]:
            path = os.path.join(args.output_dir, f"{name}.json")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved {path} ({len(data)} entries)")

        # Combined video manifest (original + frame sequences)
        combined = uav_manifest + aerial_manifest + all_frame_seq
        combined_path = os.path.join(args.output_dir, "video_all.json")
        with open(combined_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"  Saved combined: {combined_path} ({len(combined)} entries)")
    else:
        # Combined = just frame sequences (can be extended later)
        combined_path = os.path.join(args.output_dir, "video_all.json")
        with open(combined_path, 'w') as f:
            json.dump(all_frame_seq, f, indent=2)
        print(f"\nSaved video_all.json: {combined_path} ({len(all_frame_seq)} entries)")


if __name__ == "__main__":
    main()

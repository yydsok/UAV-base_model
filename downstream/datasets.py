import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import RGBIR_MEAN, RGBIR_STD

try:
    from mmrotate.core.bbox.transforms import poly2obb_np as _poly2obb_np
except Exception:
    _poly2obb_np = None


DRONEVEHICLE_CLASSES = ["car", "truck", "bus", "feright car", "van"]
DRONEVEHICLE_LABEL_MAP = {name: idx + 1 for idx, name in enumerate(DRONEVEHICLE_CLASSES)}

# VisDrone-DET: 10 foreground categories (0=ignored, 11=others are skipped)
VISDRONE_DET_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VISDRONE_DET_CATEGORY_MAP = {i + 1: i + 1 for i in range(10)}  # cat_id 1-10 → label 1-10

# UAVDT: 3 categories
UAVDT_CLASSES = ["car", "truck", "bus"]
UAVDT_CATEGORY_MAP = {1: 1, 2: 2, 3: 3}
LLVIP_CLASSES = ["person"]
LLVIP_LABEL_MAP = {"person": 1}
HITUAV_CLASSES = ["person", "car", "bicycle", "other vehicle"]
HITUAV_LABEL_MAP = {name: idx + 1 for idx, name in enumerate(HITUAV_CLASSES)}
M3OT_CLASSES = ["vehicle"]
M3OT_LABEL_MAP = {"vehicle": 1}

UAVID_CLASSES = [
    "background",
    "building",
    "road",
    "tree",
    "low_vegetation",
    "moving_car",
    "static_car",
    "human",
]
UAVID_COLOR_MAP = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (128, 64, 128): 2,
    (0, 128, 0): 3,
    (128, 128, 0): 4,
    (64, 0, 128): 5,
    (192, 0, 192): 6,
    (64, 64, 0): 7,
}

UDD5_CLASSES = ["other", "vegetation", "road", "vehicle", "facade"]
UDD6_CLASSES = ["other", "vegetation", "road", "vehicle", "facade", "roof"]
UDD5_COLOR_MAP = {
    (0, 0, 0): 0,
    (107, 142, 35): 1,
    (128, 64, 128): 2,
    (0, 0, 142): 3,
    (102, 102, 156): 4,
}
UDD6_COLOR_MAP = {
    **UDD5_COLOR_MAP,
    (70, 70, 70): 5,
}

AEROSCAPES_CLASSES = [
    "background",
    "person",
    "bike",
    "car",
    "drone",
    "boat",
    "animal",
    "obstacle",
    "construction",
    "vegetation",
    "road",
    "sky",
]


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def collate_segmentation_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def collate_change_detection_batch(batch):
    image_a, image_b, labels = zip(*batch)
    return torch.stack(image_a, dim=0), torch.stack(image_b, dim=0), torch.stack(labels, dim=0)


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _read_ir(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read IR image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    if image.max() > 255:
        denom = max(image.max() - image.min(), 1.0)
        image = (image - image.min()) / denom
    else:
        image = image / 255.0
    return image


def _merge_rgb_ir(rgb_path=None, ir_path=None, modality="both"):
    rgb = None
    ir = None
    if rgb_path is not None and modality in ("both", "rgb_only"):
        rgb = _read_rgb(rgb_path)
    if ir_path is not None and modality in ("both", "ir_only"):
        ir = _read_ir(ir_path)

    if rgb is None and ir is None:
        raise ValueError("At least one modality must be available.")

    if rgb is not None:
        target_h, target_w = rgb.shape[:2]
    else:
        target_h, target_w = ir.shape[:2]

    if rgb is None:
        rgb = np.zeros((target_h, target_w, 3), dtype=np.float32)
    elif rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if ir is None:
        ir = np.zeros((target_h, target_w), dtype=np.float32)
    elif ir.shape[:2] != (target_h, target_w):
        ir = cv2.resize(ir, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    merged = np.concatenate([rgb, ir[..., None]], axis=2)
    return torch.from_numpy(merged).permute(2, 0, 1).float()


def _normalize_rgbir(image):
    mean = torch.tensor(RGBIR_MEAN, dtype=image.dtype).view(-1, 1, 1)
    std = torch.tensor(RGBIR_STD, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean) / std


def _decode_color_mask(mask_rgb, mapping, ignore_index=255):
    label = np.full(mask_rgb.shape[:2], ignore_index, dtype=np.uint8)
    for color, class_id in mapping.items():
        matches = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        label[matches] = class_id
    return label


def _parse_voc_annotation(path, label_map, polygon=False, ignore_labels=None):
    ignore_labels = ignore_labels or set()
    tree = ET.parse(path)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name in ignore_labels or name not in label_map:
            continue
        if polygon:
            poly = obj.find("polygon")
            if poly is None:
                continue
            xs = [float(poly.findtext(f"x{idx}", default="0")) for idx in range(1, 5)]
            ys = [float(poly.findtext(f"y{idx}", default="0")) for idx in range(1, 5)]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", default="0"))
            ymin = float(box.findtext("ymin", default="0"))
            xmax = float(box.findtext("xmax", default="0"))
            ymax = float(box.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[name])
    return boxes, labels


def _poly_to_obb(points, angle_version="le90"):
    points = np.asarray(points, dtype=np.float32).reshape(-1)
    if _poly2obb_np is not None:
        try:
            obb = _poly2obb_np(points, angle_version)
        except Exception:
            obb = None
        if obb is not None:
            cx, cy, w, h, angle = [float(x) for x in obb]
            return [cx, cy, max(w, 1e-3), max(h, 1e-3), float(angle)]

    pts = points.reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), deg = cv2.minAreaRect(pts)
    if w <= 0 or h <= 0:
        return None
    angle = np.deg2rad(deg)
    if w < h:
        w, h = h, w
        angle += np.pi / 2.0
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return [float(cx), float(cy), float(w), float(h), float(angle)]


def _obb_to_hbb(obb):
    cx, cy, w, h, angle = [float(x) for x in obb]
    rect = ((cx, cy), (w, h), np.rad2deg(angle))
    corners = cv2.boxPoints(rect)
    x_min = float(np.min(corners[:, 0]))
    y_min = float(np.min(corners[:, 1]))
    x_max = float(np.max(corners[:, 0]))
    y_max = float(np.max(corners[:, 1]))
    return [x_min, y_min, x_max, y_max]


def _parse_oriented_voc_annotation(path, label_map, angle_version="le90"):
    tree = ET.parse(path)
    root = tree.getroot()
    hboxes = []
    rbboxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in label_map:
            continue

        obb = None
        robnd = obj.find("robndbox")
        if robnd is not None:
            cx = float(robnd.findtext("cx", default="0"))
            cy = float(robnd.findtext("cy", default="0"))
            w = float(robnd.findtext("w", default="0"))
            h = float(robnd.findtext("h", default="0"))
            angle = float(robnd.findtext("angle", default="0"))
            if abs(angle) > np.pi:
                angle = np.deg2rad(angle)
            obb = [cx, cy, w, h, angle]

        if obb is None:
            poly = obj.find("polygon")
            if poly is not None:
                xs = [float(poly.findtext(f"x{idx}", default="0")) for idx in range(1, 5)]
                ys = [float(poly.findtext(f"y{idx}", default="0")) for idx in range(1, 5)]
                pts = list(zip(xs, ys))
                obb = _poly_to_obb(pts, angle_version=angle_version)

        if obb is None:
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", default="0"))
            ymin = float(box.findtext("ymin", default="0"))
            xmax = float(box.findtext("xmax", default="0"))
            ymax = float(box.findtext("ymax", default="0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            obb = [cx, cy, xmax - xmin, ymax - ymin, 0.0]

        if obb is None:
            continue
        if obb[2] <= 1e-3 or obb[3] <= 1e-3:
            continue

        hbb = _obb_to_hbb(obb)
        if hbb[2] <= hbb[0] or hbb[3] <= hbb[1]:
            continue
        hboxes.append(hbb)
        rbboxes.append([float(x) for x in obb])
        labels.append(label_map[name])
    return hboxes, rbboxes, labels


def _build_detection_target(index, boxes, labels):
    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
    iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": area,
        "iscrowd": iscrowd,
        "image_id": torch.tensor([index], dtype=torch.int64),
    }


def _build_oriented_detection_target(index, hboxes, rbboxes, labels):
    target = _build_detection_target(index, hboxes, labels)
    if rbboxes:
        rbboxes_tensor = torch.tensor(rbboxes, dtype=torch.float32)
    else:
        rbboxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
    target["rbboxes"] = rbboxes_tensor
    return target


class DroneVehicleDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="both", annotation_source="rgb"):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.class_names = DRONEVEHICLE_CLASSES
        self.num_classes = len(self.class_names)

        split_prefix = {"train": "train", "val": "val", "test": "test"}[split]
        self.rgb_dir = self.root / split / f"{split_prefix}img"
        self.ir_dir = self.root / split / f"{split_prefix}imgr"
        label_suffix = "labelr" if annotation_source == "ir" else "label"
        self.label_dir = self.root / split / f"{split_prefix}{label_suffix}"

        self.samples = []
        for label_path in sorted(self.label_dir.glob("*.xml")):
            stem = label_path.stem
            rgb_path = self.rgb_dir / f"{stem}.jpg"
            ir_path = self.ir_dir / f"{stem}.jpg"
            if rgb_path.exists() and ir_path.exists():
                self.samples.append((rgb_path, ir_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, ir_path, label_path = self.samples[index]
        image = _merge_rgb_ir(rgb_path, ir_path, modality=self.modality)
        boxes, labels = _parse_voc_annotation(label_path, DRONEVEHICLE_LABEL_MAP, polygon=True)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class VisDroneDETDataset(Dataset):
    """VisDrone-DET detection dataset (RGB-only, zero-padded IR channel).

    Expected directory layout::

        root/
        ├── VisDrone2019-DET-train/
        │   ├── images/   *.jpg
        │   └── annotations/  *.txt
        ├── VisDrone2019-DET-val/
        └── VisDrone2019-DET-test-dev/

    Annotation format per line:
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
    Categories 1-10 are kept; 0 (ignored) and 11 (others) are skipped.
    """

    def __init__(self, root, split="train", modality="rgb_only"):
        self.root = Path(root)
        self.modality = modality
        self.class_names = VISDRONE_DET_CLASSES
        self.num_classes = len(self.class_names)

        split_dir_map = {
            "train": "VisDrone2019-DET-train",
            "val": "VisDrone2019-DET-val",
            "test": "VisDrone2019-DET-test-dev",
        }
        split_dir = self.root / split_dir_map[split]
        img_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"

        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            ann_path = ann_dir / f"{img_path.stem}.txt"
            if ann_path.exists():
                self.samples.append((img_path, ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]
        image = _merge_rgb_ir(rgb_path=img_path, ir_path=None, modality=self.modality)

        boxes = []
        labels = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cat_id = int(parts[5])
                if cat_id < 1 or cat_id > 10 or w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(VISDRONE_DET_CATEGORY_MAP[cat_id])

        target = _build_detection_target(index, boxes, labels)
        return image, target


class UAVDTDetectionDataset(Dataset):
    """UAVDT detection dataset (RGB-only, zero-padded IR channel).

    Expected directory layout::

        root/
        ├── UAV-benchmark-S/          # image sequences
        │   ├── S0101/ img000001.jpg ...
        │   └── ...
        ├── UAV-benchmark-MOTD_v1.0/  # GT annotations
        │   └── GT/  M0101_gt_whole.txt ...
        └── M_attr/
            ├── train/  M0101_attr.txt ...
            └── test/   M0203_attr.txt ...

    GT format: <frame>,<id>,<x>,<y>,<w>,<h>,<out-of-view>,<occlusion>,<category>
    Categories: 1=car, 2=truck, 3=bus.  We flatten sequences to per-frame samples.
    """

    def __init__(self, root, split="train", modality="rgb_only"):
        self.root = Path(root)
        self.modality = modality
        self.class_names = UAVDT_CLASSES
        self.num_classes = len(self.class_names)

        attr_dir = self.root / "M_attr" / split
        seq_names = set()
        for attr_file in attr_dir.glob("*_attr.txt"):
            seq_names.add(attr_file.stem.replace("_attr", ""))

        gt_dir = self.root / "UAV-benchmark-MOTD_v1.0" / "GT"
        img_base = self.root / "UAV-benchmark-S"

        self.samples = []
        for seq in sorted(seq_names):
            s_seq = seq.replace("M", "S")
            seq_img_dir = img_base / s_seq
            if not seq_img_dir.exists():
                continue

            gt_path = gt_dir / f"{seq}_gt_whole.txt"
            if not gt_path.exists():
                gt_path = gt_dir / f"{seq}_gt.txt"
            if not gt_path.exists():
                continue

            gt_by_frame = defaultdict(list)
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 9:
                        continue
                    frame_id = int(float(parts[0]))
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cat_id = int(float(parts[8]))
                    if cat_id not in UAVDT_CATEGORY_MAP or w <= 1 or h <= 1:
                        continue
                    gt_by_frame[frame_id].append(([x, y, x + w, y + h], UAVDT_CATEGORY_MAP[cat_id]))

            for img_path in sorted(seq_img_dir.glob("*.jpg")):
                frame_num = int(img_path.stem.replace("img", ""))
                anns = gt_by_frame.get(frame_num, [])
                if anns:
                    frame_boxes = [a[0] for a in anns]
                    frame_labels = [a[1] for a in anns]
                    self.samples.append((img_path, frame_boxes, frame_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, boxes, labels = self.samples[index]
        image = _merge_rgb_ir(rgb_path=img_path, ir_path=None, modality=self.modality)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class VisDroneMOTTrackingDataset:
    """VisDrone-MOT tracking dataset.

    Expected directory layout::

        root/
        ├── VisDrone2019-MOT-train/
        │   ├── sequences/  <seq_name>/  *.jpg
        │   └── annotations/  <seq_name>.txt
        ├── VisDrone2019-MOT-val/
        └── VisDrone2019-MOT-test-dev/

    Annotation format per line:
        <frame>,<id>,<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>
    Categories 1-10 kept (same as VisDrone-DET).
    """

    def __init__(self, root, split="val"):
        self.root = Path(root)
        split_dir_map = {
            "train": "VisDrone2019-MOT-train",
            "val": "VisDrone2019-MOT-val",
            "test": "VisDrone2019-MOT-test-dev",
        }
        split_dir = self.root / split_dir_map[split]
        seq_dir = split_dir / "sequences"
        ann_dir = split_dir / "annotations"

        self.sequences = []
        for seq_path in sorted(d for d in seq_dir.iterdir() if d.is_dir()):
            seq_name = seq_path.name
            ann_path = ann_dir / f"{seq_name}.txt"
            if not ann_path.exists():
                continue

            gt_by_frame = defaultdict(list)
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    frame_id = int(float(parts[0]))
                    obj_id = int(float(parts[1]))
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cat_id = int(float(parts[7]))
                    if cat_id < 1 or cat_id > 10 or w <= 1 or h <= 1:
                        continue
                    gt_by_frame[frame_id].append({
                        "track_id": obj_id,
                        "label": cat_id,
                        "box": [x, y, x + w, y + h],
                    })

            img_files = sorted(seq_path.glob("*.jpg"))
            frames = []
            for img_path in img_files:
                frame_num = int(img_path.stem)
                frames.append({
                    "rgb_path": img_path,
                    "ir_path": None,
                    "frame_id": frame_num,
                    "annotations": gt_by_frame.get(frame_num, []),
                })

            self.sequences.append({
                "name": seq_name,
                "frames": frames,
                "class_names": VISDRONE_DET_CLASSES,
            })


class LLVIPDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="both"):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.class_names = LLVIP_CLASSES
        self.num_classes = len(self.class_names)
        self.rgb_dir = self.root / "visible" / split
        self.ir_dir = self.root / "infrared" / split
        self.ann_dir = self.root / "Annotations"

        self.samples = []
        for rgb_path in sorted(self.rgb_dir.glob("*.jpg")):
            stem = rgb_path.stem
            ir_path = self.ir_dir / f"{stem}.jpg"
            ann_path = self.ann_dir / f"{stem}.xml"
            if ir_path.exists() and ann_path.exists():
                self.samples.append((rgb_path, ir_path, ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, ir_path, ann_path = self.samples[index]
        image = _merge_rgb_ir(rgb_path, ir_path, modality=self.modality)
        boxes, labels = _parse_voc_annotation(ann_path, LLVIP_LABEL_MAP, polygon=False)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class RGBTTinyDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="both", subset="00"):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        ann_path = self.root / "annotations_coco" / f"instances_{subset}_{split}2017.json"
        with open(ann_path, "r") as f:
            data = json.load(f)

        categories = sorted(data["categories"], key=lambda item: item["id"])
        self.class_names = [item["name"] for item in categories]
        category_map = {item["id"]: idx + 1 for idx, item in enumerate(categories)}
        self.num_classes = len(self.class_names)

        images = {item["id"]: item for item in data["images"]}
        ann_index = defaultdict(list)
        for ann in data["annotations"]:
            ann_index[ann["image_id"]].append(ann)

        self.samples = []
        for image_id, meta in images.items():
            rgb_rel = meta["file_name"]
            rgb_path = self.root / "images" / rgb_rel
            ir_rel = rgb_rel.replace("/00/", "/01/")
            ir_path = self.root / "images" / ir_rel
            boxes = []
            labels = []
            for ann in ann_index[image_id]:
                x, y, w, h = ann["bbox"]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(category_map[ann["category_id"]])
            if rgb_path.exists() and ir_path.exists() and boxes:
                self.samples.append((rgb_path, ir_path, boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, ir_path, boxes, labels = self.samples[index]
        image = _merge_rgb_ir(rgb_path, ir_path, modality=self.modality)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class HITUAVDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="ir_only"):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.class_names = HITUAV_CLASSES
        self.num_classes = len(self.class_names)

        img_dir = self.root / split / "img"
        ann_dir = self.root / split / "ann"
        self.samples = []
        for ann_path in sorted(ann_dir.glob("*.json")):
            stem = ann_path.name.replace(".json", "")
            img_path = img_dir / stem
            if img_path.exists():
                self.samples.append((img_path, ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]
        image = _merge_rgb_ir(rgb_path=None, ir_path=img_path, modality=self.modality)

        with open(ann_path, "r") as f:
            ann = json.load(f)
        boxes = []
        labels = []
        for obj in ann.get("objects", []):
            label = obj.get("classTitle")
            if label == "dontcare" or label not in HITUAV_LABEL_MAP:
                continue
            points = obj.get("points", {}).get("exterior", [])
            if len(points) != 2:
                continue
            (xmin, ymin), (xmax, ymax) = points
            xmin, xmax = sorted((float(xmin), float(xmax)))
            ymin, ymax = sorted((float(ymin), float(ymax)))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(HITUAV_LABEL_MAP[label])

        target = _build_detection_target(index, boxes, labels)
        return image, target


class M3OTDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="both"):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.class_names = M3OT_CLASSES
        self.num_classes = len(self.class_names)

        rgb_split_dir = self.root / "2" / "rgb" / split
        ir_split_dir = self.root / "2" / "ir" / split
        self.samples = []

        for rgb_seq_dir in sorted(d for d in rgb_split_dir.iterdir() if d.is_dir()):
            ir_seq_dir = ir_split_dir / f"{rgb_seq_dir.name}T"
            gt_path = rgb_seq_dir / "gt" / "gt.txt"
            if not ir_seq_dir.exists() or not gt_path.exists():
                continue

            gt_by_frame = defaultdict(list)
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame_id = int(float(parts[0]))
                    xmin = float(parts[2])
                    ymin = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    gt_by_frame[frame_id].append([xmin, ymin, xmin + width, ymin + height])

            rgb_frames = sorted((rgb_seq_dir / "img1").glob("*"))
            ir_frames = sorted((ir_seq_dir / "img1").glob("*"))
            pair_count = min(len(rgb_frames), len(ir_frames))
            for frame_idx in range(pair_count):
                boxes = gt_by_frame.get(frame_idx + 1, [])
                if not boxes:
                    continue
                labels = [1] * len(boxes)
                self.samples.append((rgb_frames[frame_idx], ir_frames[frame_idx], boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, ir_path, boxes, labels = self.samples[index]
        image = _merge_rgb_ir(rgb_path, ir_path, modality=self.modality)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class OrientedXmlDetectionDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        modality="rgb_only",
        image_dir=None,
        ann_dir=None,
        split_file=None,
        class_names=None,
        angle_version="le90",
    ):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.angle_version = angle_version
        self.image_dir = self._resolve_image_dir(image_dir)
        self.ann_dir = self._resolve_ann_dir(ann_dir)
        self.samples = self._build_samples(split_file=split_file)

        if class_names is None:
            discovered = sorted({item["name"] for sample in self.samples for item in sample["objects"]})
            self.class_names = discovered
        else:
            self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.label_map = {name: idx + 1 for idx, name in enumerate(self.class_names)}

        filtered = []
        for sample in self.samples:
            has_known = any(obj["name"] in self.label_map for obj in sample["objects"])
            if has_known:
                filtered.append(sample)
        self.samples = filtered

    def _resolve_image_dir(self, image_dir):
        candidates = []
        if image_dir is not None:
            candidates.append(self.root / image_dir)
        candidates.extend(
            [
                self.root / "images" / self.split,
                self.root / "images",
                self.root / "JPEGImages" / self.split,
                self.root / "JPEGImages-trainval",
                self.root / "JPEGImages",
            ]
        )
        for path in candidates:
            if path.exists() and path.is_dir():
                return path
        raise FileNotFoundError(f"Could not resolve image directory under: {self.root}")

    def _resolve_ann_dir(self, ann_dir):
        candidates = []
        if ann_dir is not None:
            candidates.append(self.root / ann_dir)
        candidates.extend(
            [
                self.root / "annotations" / self.split,
                self.root / "annotations",
                self.root / "annfiles" / self.split,
                self.root / "annfiles",
                self.root / "Annotations" / "Oriented Bounding Boxes",
                self.root / "Annotations",
            ]
        )
        for path in candidates:
            if path.exists() and path.is_dir():
                return path
        raise FileNotFoundError(f"Could not resolve annotation directory under: {self.root}")

    def _resolve_split_file(self, split_file):
        candidates = []
        if split_file is not None:
            candidates.append(self.root / split_file)
        candidates.extend(
            [
                self.root / "splits" / f"{self.split}.txt",
                self.root / "ImageSets" / "Main" / f"{self.split}.txt",
                self.root / "ImageSets" / f"{self.split}.txt",
            ]
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def _parse_object_names(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        names = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            if name:
                names.append(name)
        return names

    def _find_image_path(self, stem):
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            path = self.image_dir / f"{stem}{ext}"
            if path.exists():
                return path
        return None

    def _build_samples(self, split_file=None):
        split_path = self._resolve_split_file(split_file)
        names = []
        if split_path is not None:
            for line in split_path.read_text().splitlines():
                entry = line.strip().split()
                if not entry:
                    continue
                names.append(Path(entry[0]).stem)
        else:
            names = [path.stem for path in sorted(self.ann_dir.glob("*.xml"))]

        samples = []
        for stem in names:
            ann_path = self.ann_dir / f"{stem}.xml"
            if not ann_path.exists():
                continue
            image_path = self._find_image_path(stem)
            if image_path is None:
                continue
            object_names = self._parse_object_names(ann_path)
            if not object_names:
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "ann_path": ann_path,
                    "objects": [{"name": name} for name in object_names],
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = _merge_rgb_ir(sample["image_path"], None, modality=self.modality)
        hboxes, rbboxes, labels = _parse_oriented_voc_annotation(
            sample["ann_path"],
            self.label_map,
            angle_version=self.angle_version,
        )
        target = _build_oriented_detection_target(index, hboxes, rbboxes, labels)
        return image, target


class M3OTTrackingDataset:
    def __init__(self, root, split="val"):
        self.root = Path(root)
        self.split = split
        self.sequences = []

        rgb_split_dir = self.root / "2" / "rgb" / split
        ir_split_dir = self.root / "2" / "ir" / split
        for rgb_seq_dir in sorted(d for d in rgb_split_dir.iterdir() if d.is_dir()):
            ir_seq_dir = ir_split_dir / f"{rgb_seq_dir.name}T"
            if not ir_seq_dir.exists():
                continue

            gt_path = rgb_seq_dir / "gt" / "gt.txt"
            gt_by_frame = defaultdict(list)
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame_id = int(float(parts[0]))
                    obj_id = int(float(parts[1]))
                    xmin = float(parts[2])
                    ymin = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    gt_by_frame[frame_id].append(
                        {
                            "track_id": obj_id,
                            "label": 1,
                            "box": [xmin, ymin, xmin + width, ymin + height],
                        }
                    )

            rgb_frames = sorted((rgb_seq_dir / "img1").glob("*"))
            ir_frames = sorted((ir_seq_dir / "img1").glob("*"))
            pair_count = min(len(rgb_frames), len(ir_frames))
            frames = []
            for frame_idx in range(pair_count):
                frames.append(
                    {
                        "rgb_path": rgb_frames[frame_idx],
                        "ir_path": ir_frames[frame_idx],
                        "frame_id": frame_idx + 1,
                        "annotations": gt_by_frame.get(frame_idx + 1, []),
                    }
                )
            self.sequences.append(
                {
                    "name": rgb_seq_dir.name,
                    "frames": frames,
                    "class_names": M3OT_CLASSES,
                }
            )


class VTTinyMOTDetectionDataset(Dataset):
    def __init__(self, root, split="train", modality="both"):
        self.root = Path(root)
        self.split = split
        self.modality = modality

        ann_path = self.root / "annotations" / f"instances_{split}2017.json"
        with open(ann_path, "r") as f:
            data = json.load(f)

        categories = sorted(data["categories"], key=lambda item: item["id"])
        self.class_names = [item["name"] for item in categories]
        self.num_classes = len(self.class_names)
        category_map = {item["id"]: idx + 1 for idx, item in enumerate(categories)}

        images = {item["id"]: item for item in data["images"]}
        ann_index = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("ignore", 0) or ann.get("iscrowd", 0):
                continue
            ann_index[ann["image_id"]].append(ann)

        split_dir = self.root / f"{split}2017"
        self.samples = []
        for image_id, meta in images.items():
            rgb_rel = meta["file_name"]
            if "/00/" not in rgb_rel:
                continue
            ir_rel = rgb_rel.replace("/00/", "/01/")
            rgb_path = split_dir / rgb_rel
            ir_path = split_dir / ir_rel
            boxes = []
            labels = []
            for ann in ann_index.get(image_id, []):
                x, y, w, h = ann["bbox"]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(category_map[ann["category_id"]])
            if rgb_path.exists() and ir_path.exists() and boxes:
                self.samples.append((rgb_path, ir_path, boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, ir_path, boxes, labels = self.samples[index]
        image = _merge_rgb_ir(rgb_path, ir_path, modality=self.modality)
        target = _build_detection_target(index, boxes, labels)
        return image, target


class VTTinyMOTTrackingDataset:
    def __init__(self, root, split="test"):
        self.root = Path(root)
        self.split = split
        ann_path = self.root / "annotations" / f"instances_{split}2017.json"
        with open(ann_path, "r") as f:
            data = json.load(f)

        categories = sorted(data["categories"], key=lambda item: item["id"])
        self.class_names = [item["name"] for item in categories]
        category_map = {item["id"]: idx + 1 for idx, item in enumerate(categories)}

        images_by_id = {item["id"]: item for item in data["images"]}
        ann_index = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("ignore", 0) or ann.get("iscrowd", 0):
                continue
            ann_index[ann["image_id"]].append(ann)

        images_by_video = defaultdict(list)
        for meta in data["images"]:
            if "/00/" not in meta["file_name"]:
                continue
            images_by_video[meta["video_id"]].append(meta)

        videos = {item["id"]: item for item in data.get("videos", [])}
        split_dir = self.root / f"{split}2017"
        self.sequences = []
        for video_id in sorted(images_by_video):
            frames = []
            ordered_images = sorted(
                images_by_video[video_id],
                key=lambda item: (item.get("frame_id", 0), item.get("mot_frame_id", 0), item["file_name"]),
            )
            for meta in ordered_images:
                rgb_rel = meta["file_name"]
                ir_rel = rgb_rel.replace("/00/", "/01/")
                rgb_path = split_dir / rgb_rel
                ir_path = split_dir / ir_rel
                if not rgb_path.exists() or not ir_path.exists():
                    continue
                frame_annotations = []
                for ann in ann_index.get(meta["id"], []):
                    x, y, w, h = ann["bbox"]
                    if w <= 1 or h <= 1:
                        continue
                    frame_annotations.append(
                        {
                            "track_id": int(ann["track_id"]),
                            "label": category_map[ann["category_id"]],
                            "box": [x, y, x + w, y + h],
                        }
                    )
                frames.append(
                    {
                        "rgb_path": rgb_path,
                        "ir_path": ir_path,
                        "frame_id": int(meta.get("mot_frame_id", meta.get("frame_id", 0))),
                        "annotations": frame_annotations,
                    }
                )
            if not frames:
                continue
            video_meta = videos.get(video_id, {})
            self.sequences.append(
                {
                    "name": video_meta.get("name", str(video_id)),
                    "frames": frames,
                    "class_names": self.class_names,
                }
            )


class RetrievalImageDataset(Dataset):
    def __init__(self, samples, image_size=224):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def _load_image(self, sample):
        image = _merge_rgb_ir(
            rgb_path=sample.get("rgb_path"),
            ir_path=sample.get("ir_path"),
            modality=sample.get("modality", "rgb_only"),
        )
        if self.image_size is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return _normalize_rgbir(image)

    def __getitem__(self, index):
        sample = self.samples[index]
        return self._load_image(sample), int(sample["label"]), index


class DroneVehicleRetrievalDataset(RetrievalImageDataset):
    def __init__(self, root, split="val", view="rgb", image_size=224, max_samples=None):
        root = Path(root)
        rgb_dir = root / split / f"{split}img"
        ir_dir = root / split / f"{split}imgr"
        samples = []
        label = 0
        for rgb_path in sorted(rgb_dir.glob("*")):
            if rgb_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            stem = rgb_path.stem
            ir_path = ir_dir / f"{stem}.jpg"
            if not ir_path.exists():
                ir_path = ir_dir / f"{stem}.png"
            if not ir_path.exists():
                continue
            samples.append(
                {
                    "rgb_path": rgb_path if view == "rgb" else None,
                    "ir_path": ir_path if view == "ir" else None,
                    "modality": "rgb_only" if view == "rgb" else "ir_only",
                    "label": label,
                }
            )
            label += 1
            if max_samples is not None and len(samples) >= max_samples:
                break
        super().__init__(samples, image_size=image_size)


class LLVIPRetrievalDataset(RetrievalImageDataset):
    def __init__(self, root, split="test", view="rgb", image_size=224, max_samples=None):
        root = Path(root)
        rgb_dir = root / "visible" / split
        ir_dir = root / "infrared" / split
        rgb_files = sorted([path for path in rgb_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        ir_lookup = {path.stem: path for path in ir_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}

        samples = []
        label = 0
        for rgb_path in rgb_files:
            ir_path = ir_lookup.get(rgb_path.stem)
            if ir_path is None:
                continue
            samples.append(
                {
                    "rgb_path": rgb_path if view == "rgb" else None,
                    "ir_path": ir_path if view == "ir" else None,
                    "modality": "rgb_only" if view == "rgb" else "ir_only",
                    "label": label,
                }
            )
            label += 1
            if max_samples is not None and len(samples) >= max_samples:
                break
        super().__init__(samples, image_size=image_size)


class SUES200RetrievalDataset(RetrievalImageDataset):
    def __init__(
        self,
        root,
        view="drone",
        image_size=224,
        max_samples=None,
        altitudes=("150", "200", "250", "300"),
        query_stride=1,
    ):
        root = Path(root)
        samples = []
        locations = sorted(path.name for path in (root / "satellite-view").iterdir() if path.is_dir())
        label_map = {location: idx for idx, location in enumerate(locations)}

        if view == "satellite":
            for location in locations:
                image_path = root / "satellite-view" / location / "0.png"
                if not image_path.exists():
                    continue
                samples.append(
                    {
                        "rgb_path": image_path,
                        "ir_path": None,
                        "modality": "rgb_only",
                        "label": label_map[location],
                    }
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        else:
            altitudes = tuple(str(value) for value in altitudes)
            for location in locations:
                for altitude in altitudes:
                    view_dir = root / "drone_view_512" / location / altitude
                    if not view_dir.exists():
                        continue
                    frame_paths = sorted(
                        [path for path in view_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
                    )
                    for frame_idx, image_path in enumerate(frame_paths):
                        if query_stride > 1 and frame_idx % query_stride != 0:
                            continue
                        samples.append(
                            {
                                "rgb_path": image_path,
                                "ir_path": None,
                                "modality": "rgb_only",
                                "label": label_map[location],
                            }
                        )
                        if max_samples is not None and len(samples) >= max_samples:
                            break
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                if max_samples is not None and len(samples) >= max_samples:
                    break
        super().__init__(samples, image_size=image_size)


class CVOGLRetrievalDataset(RetrievalImageDataset):
    def __init__(self, root, view="drone", image_size=224, max_samples=None):
        root = Path(root)
        drone_dir = root / "CVOGL_DroneAerial" / "query"
        svi_dir = root / "CVOGL_SVI" / "query"
        drone_lookup = {path.name: path for path in drone_dir.glob("*.jpg")}
        svi_lookup = {path.name: path for path in svi_dir.glob("*.jpg")}
        common_names = sorted(set(drone_lookup) & set(svi_lookup))

        samples = []
        for label, name in enumerate(common_names):
            image_path = drone_lookup[name] if view == "drone" else svi_lookup[name]
            samples.append(
                {
                    "rgb_path": image_path,
                    "ir_path": None,
                    "modality": "rgb_only",
                    "label": label,
                }
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
        super().__init__(samples, image_size=image_size)


class SceneFolderClassificationDataset(Dataset):
    """Generic scene classification dataset with deterministic per-class splits."""

    def __init__(
        self,
        root,
        split="train",
        image_size=224,
        train_ratio=0.2,
        val_ratio=0.1,
        seed=42,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.class_names = []
        self.num_classes = 0
        self.samples = self._build_samples()

    def _iter_class_dirs(self, base_dir):
        return sorted([path for path in base_dir.iterdir() if path.is_dir()])

    def _list_images(self, class_dir):
        return sorted(
            [
                path for path in class_dir.glob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )

    def _build_samples_from_pre_split(self, split_dir):
        class_dirs = self._iter_class_dirs(split_dir)
        self.class_names = [path.name for path in class_dirs]
        self.num_classes = len(self.class_names)
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        samples = []
        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            for image_path in self._list_images(class_dir):
                samples.append((image_path, label))
        return samples

    def _build_samples_from_full_dir(self):
        class_dirs = self._iter_class_dirs(self.root)
        self.class_names = [path.name for path in class_dirs]
        self.num_classes = len(self.class_names)
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        rng = random.Random(self.seed)
        samples = []

        for class_dir in class_dirs:
            image_paths = self._list_images(class_dir)
            if not image_paths:
                continue
            image_paths = image_paths.copy()
            rng.shuffle(image_paths)

            total = len(image_paths)
            n_train = max(1, int(round(total * self.train_ratio)))
            n_train = min(n_train, total - 1) if total > 1 else 1
            remaining = max(total - n_train, 0)
            n_val = int(round(total * self.val_ratio))
            if remaining > 1:
                n_val = max(1, n_val)
            n_val = min(n_val, remaining)

            train_paths = image_paths[:n_train]
            val_paths = image_paths[n_train:n_train + n_val]
            test_paths = image_paths[n_train + n_val:]

            if self.split == "train":
                selected = train_paths
            elif self.split == "val":
                selected = val_paths
            else:
                selected = test_paths

            if not selected:
                continue
            label = class_to_idx[class_dir.name]
            for image_path in selected:
                samples.append((image_path, label))
        return samples

    def _build_samples(self):
        split_dir = self.root / self.split
        if split_dir.exists() and split_dir.is_dir():
            return self._build_samples_from_pre_split(split_dir)
        return self._build_samples_from_full_dir()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = _read_rgb(image_path)
        if self.image_size is not None:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        ir = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
        merged = np.concatenate([image, ir], axis=2)
        tensor = torch.from_numpy(merged).permute(2, 0, 1).float()
        tensor = _normalize_rgbir(tensor)
        return tensor, label


class PairChangeDetectionDataset(Dataset):
    """Change-detection dataset with A/B paired images and binary labels."""

    def __init__(
        self,
        root,
        split="train",
        image_size=256,
        random_flip=True,
        list_dir="list",
        image_a_dir="A",
        image_b_dir="B",
        label_dir="label",
        label_suffix=".png",
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.random_flip = random_flip
        self.image_a_dir = self.root / image_a_dir
        self.image_b_dir = self.root / image_b_dir
        self.label_dir = self.root / label_dir
        self.label_suffix = label_suffix
        self.samples = self._build_samples(list_dir=list_dir)

    def _build_samples(self, list_dir):
        list_file = self.root / list_dir / f"{self.split}.txt"
        names = []
        if list_file.exists():
            for line in list_file.read_text().splitlines():
                entry = line.strip().split()
                if entry:
                    names.append(entry[0])
        else:
            names = sorted([path.name for path in self.image_a_dir.glob("*") if path.is_file()])

        samples = []
        for name in names:
            image_a = self.image_a_dir / name
            image_b = self.image_b_dir / name
            label_name = Path(name).stem + self.label_suffix
            label_path = self.label_dir / label_name
            if image_a.exists() and image_b.exists() and label_path.exists():
                samples.append((image_a, image_b, label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _prepare_rgbir(self, rgb):
        if self.image_size is not None:
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        ir = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        merged = np.concatenate([rgb, ir], axis=2)
        tensor = torch.from_numpy(merged).permute(2, 0, 1).float()
        return _normalize_rgbir(tensor)

    def __getitem__(self, index):
        image_a_path, image_b_path, label_path = self.samples[index]
        image_a = _read_rgb(image_a_path)
        image_b = _read_rgb(image_b_path)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to read change mask: {label_path}")

        if self.image_size is not None:
            label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        label = (label > 0).astype(np.int64)

        if self.random_flip and random.random() < 0.5:
            image_a = np.flip(image_a, axis=1).copy()
            image_b = np.flip(image_b, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        image_a = self._prepare_rgbir(image_a)
        image_b = self._prepare_rgbir(image_b)
        label = torch.from_numpy(label)
        return image_a, image_b, label


class BaseSegmentationDataset(Dataset):
    def __init__(self, image_size=512, modality="rgb_only", random_flip=False):
        self.image_size = image_size
        self.modality = modality
        self.random_flip = random_flip

    def _load_image(self, rgb_path):
        image = _merge_rgb_ir(rgb_path=rgb_path, ir_path=None, modality=self.modality)
        if self.image_size is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return _normalize_rgbir(image)

    def _resize_label(self, label):
        if self.image_size is None:
            return label
        resized = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return resized

    def _maybe_flip(self, image, label):
        if self.random_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            label = np.flip(label, axis=1).copy()
        return image, label


class UAVidSegmentationDataset(BaseSegmentationDataset):
    def __init__(self, root, split="train", image_size=512, random_flip=True):
        super().__init__(image_size=image_size, modality="rgb_only", random_flip=random_flip)
        self.root = Path(root)
        self.class_names = UAVID_CLASSES
        self.num_classes = len(self.class_names)
        self.samples = []
        for seq_dir in sorted((self.root / split).iterdir()):
            if not seq_dir.is_dir():
                continue
            img_dir = seq_dir / "Images"
            label_dir = seq_dir / "Labels"
            for img_path in sorted(img_dir.glob("*")):
                label_path = label_dir / f"{img_path.stem}.png"
                if label_path.exists():
                    self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = self._load_image(img_path)
        label_rgb = cv2.cvtColor(cv2.imread(str(label_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        label = _decode_color_mask(label_rgb, UAVID_COLOR_MAP)
        label = self._resize_label(label)
        image, label = self._maybe_flip(image, label)
        return image, torch.from_numpy(label.astype(np.int64))


class UDDSegmentationDataset(BaseSegmentationDataset):
    def __init__(self, root, subset="UDD6", split="train", image_size=512, random_flip=True):
        super().__init__(image_size=image_size, modality="rgb_only", random_flip=random_flip)
        self.root = Path(root) / subset / split
        self.class_names = UDD6_CLASSES if subset == "UDD6" else UDD5_CLASSES
        self.num_classes = len(self.class_names)
        color_map = UDD6_COLOR_MAP if subset == "UDD6" else UDD5_COLOR_MAP
        self.color_map = color_map
        self.samples = []
        for img_path in sorted((self.root / "src").glob("*")):
            label_path = self.root / "gt" / f"{img_path.stem}.png"
            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = self._load_image(img_path)
        label_rgb = cv2.cvtColor(cv2.imread(str(label_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        label = _decode_color_mask(label_rgb, self.color_map)
        label = self._resize_label(label)
        image, label = self._maybe_flip(image, label)
        return image, torch.from_numpy(label.astype(np.int64))


class AeroScapesSegmentationDataset(BaseSegmentationDataset):
    def __init__(self, root, split="train", image_size=512, random_flip=True):
        super().__init__(image_size=image_size, modality="rgb_only", random_flip=random_flip)
        self.root = Path(root)
        self.class_names = AEROSCAPES_CLASSES
        self.num_classes = len(self.class_names)
        split_file = self.root / "ImageSets" / ("trn.txt" if split == "train" else "val.txt")
        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.samples = []
        for name in names:
            img_path = self.root / "JPEGImages" / f"{name}.jpg"
            label_path = self.root / "SegmentationClass" / f"{name}.png"
            if img_path.exists() and label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = self._load_image(img_path)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = self._resize_label(label)
        image, label = self._maybe_flip(image, label)
        return image, torch.from_numpy(label.astype(np.int64))


def build_detection_dataset(name, root, split="train", modality="both", **kwargs):
    name = name.lower()
    if name == "dronevehicle":
        return DroneVehicleDetectionDataset(root, split=split, modality=modality, **kwargs)
    if name == "llvip":
        return LLVIPDetectionDataset(root, split=split, modality=modality)
    if name == "rgbt_tiny":
        return RGBTTinyDetectionDataset(root, split=split, modality=modality, **kwargs)
    if name == "vt_tiny_mot":
        return VTTinyMOTDetectionDataset(root, split=split, modality=modality)
    if name == "hit_uav":
        return HITUAVDetectionDataset(root, split=split, modality=modality)
    if name == "m3ot":
        return M3OTDetectionDataset(root, split=split, modality=modality)
    if name == "visdrone_det":
        return VisDroneDETDataset(root, split=split, modality=modality if modality != "both" else "rgb_only")
    if name == "uavdt":
        return UAVDTDetectionDataset(root, split=split, modality=modality if modality != "both" else "rgb_only")
    raise ValueError(f"Unsupported detection dataset '{name}'.")


def build_oriented_detection_dataset(
    name,
    root,
    split="train",
    modality="rgb_only",
    image_dir=None,
    ann_dir=None,
    split_file=None,
    class_names=None,
    angle_version="le90",
):
    name = name.lower()
    if name in {"dior_r", "fair1m", "custom"}:
        return OrientedXmlDetectionDataset(
            root=root,
            split=split,
            modality=modality,
            image_dir=image_dir,
            ann_dir=ann_dir,
            split_file=split_file,
            class_names=class_names,
            angle_version=angle_version,
        )
    raise ValueError(f"Unsupported oriented detection dataset '{name}'.")


def build_segmentation_dataset(name, root, split="train", image_size=512, **kwargs):
    name = name.lower()
    if name == "uavid":
        return UAVidSegmentationDataset(root, split=split, image_size=image_size, **kwargs)
    if name == "udd5":
        return UDDSegmentationDataset(root, subset="UDD5", split=split, image_size=image_size, **kwargs)
    if name == "udd6":
        return UDDSegmentationDataset(root, subset="UDD6", split=split, image_size=image_size, **kwargs)
    if name == "aeroscapes":
        return AeroScapesSegmentationDataset(root, split=split, image_size=image_size, **kwargs)
    raise ValueError(f"Unsupported segmentation dataset '{name}'.")


def build_tracking_dataset(name, root, split="val"):
    name = name.lower()
    if name == "m3ot":
        return M3OTTrackingDataset(root, split=split)
    if name == "vt_tiny_mot":
        return VTTinyMOTTrackingDataset(root, split=split)
    if name == "visdrone_mot":
        return VisDroneMOTTrackingDataset(root, split=split)
    raise ValueError(f"Unsupported tracking dataset '{name}'.")


def build_retrieval_dataset(name, root, view, split=None, image_size=224, max_samples=None, **kwargs):
    name = name.lower()
    if name == "dronevehicle":
        return DroneVehicleRetrievalDataset(
            root,
            split=split or "val",
            view=view,
            image_size=image_size,
            max_samples=max_samples,
        )
    if name == "llvip":
        return LLVIPRetrievalDataset(
            root,
            split=split or "test",
            view=view,
            image_size=image_size,
            max_samples=max_samples,
        )
    if name == "sues_200":
        return SUES200RetrievalDataset(
            root,
            view=view,
            image_size=image_size,
            max_samples=max_samples,
            **kwargs,
        )
    if name == "cvogl":
        return CVOGLRetrievalDataset(
            root,
            view=view,
            image_size=image_size,
            max_samples=max_samples,
        )
    raise ValueError(f"Unsupported retrieval dataset '{name}'.")


def build_scene_classification_dataset(
    name,
    root,
    split="train",
    image_size=224,
    train_ratio=None,
    val_ratio=0.1,
    seed=42,
):
    name = name.lower()
    if name == "aid":
        default_train_ratio = 0.2
    elif name == "resisc45":
        default_train_ratio = 0.1
    elif name == "imagefolder":
        default_train_ratio = 0.8
    else:
        raise ValueError(f"Unsupported scene classification dataset '{name}'.")

    ratio = default_train_ratio if train_ratio is None else float(train_ratio)
    return SceneFolderClassificationDataset(
        root=root,
        split=split,
        image_size=image_size,
        train_ratio=ratio,
        val_ratio=val_ratio,
        seed=seed,
    )


def build_change_detection_dataset(name, root, split="train", image_size=256):
    name = name.lower()
    if name in {"levir_cd", "cdd", "oscd", "dsifn_cd", "custom"}:
        return PairChangeDetectionDataset(
            root=root,
            split=split,
            image_size=image_size,
            random_flip=(split == "train"),
        )
    raise ValueError(f"Unsupported change detection dataset '{name}'.")

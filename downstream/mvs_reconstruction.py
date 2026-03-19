"""DINO-MM integration with Cas-MVSNet for 3D reconstruction.

This module provides:
  1. DinoMMMVSFeatureNet: An adapter that replaces Cas-MVSNet's FeatureNet with
     DINO-MM backbone features, projecting them to the expected channel dims.
  2. build_dino_mm_casmvsnet: Builds a CascadeMVSNet with DINO-MM feature adapter.
  3. MVSDataset: A generic multi-view stereo dataset loader for LuoJia-MVS / WHU.
  4. Training and evaluation entry points.

RingMo-Aerial paper uses Cas-MVSNet with 3 stages (ndepths=48,32,8).
Metrics: MAE, <0.6m accuracy, <3-interval accuracy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. DINO-MM Feature Adapter for Cas-MVSNet
# ---------------------------------------------------------------------------

class DinoMMMVSFeatureNet(nn.Module):
    """Replaces Cas-MVSNet FeatureNet with DINO-MM multi-scale features.

    Cas-MVSNet FeatureNet outputs (base_channels=8, num_stage=3):
        stage1: [B, 32, H/4, W/4]   (coarsest)
        stage2: [B, 16, H/2, W/2]
        stage3: [B, 8,  H,   W]     (finest)

    This adapter extracts features from DINO-MM backbone and projects them
    to match the exact channel dimensions and spatial resolutions.
    """

    def __init__(self, backbone, target_channels=(32, 16, 8)):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim
        self.target_channels = target_channels

        # 1x1 convs to project DINO-MM features to target channel dims
        self.proj_stage1 = nn.Sequential(
            nn.Conv2d(embed_dim, target_channels[0], 1, bias=False),
            nn.BatchNorm2d(target_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.proj_stage2 = nn.Sequential(
            nn.Conv2d(embed_dim, target_channels[1], 1, bias=False),
            nn.BatchNorm2d(target_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.proj_stage3 = nn.Sequential(
            nn.Conv2d(embed_dim, target_channels[2], 1, bias=False),
            nn.BatchNorm2d(target_channels[2]),
            nn.ReLU(inplace=True),
        )

        # out_channels must match Cas-MVSNet expectation
        self.out_channels = list(target_channels)

    def _extract_feature_map(self, x):
        """Run DINO-MM backbone and return patch-level feature map."""
        tokens = self.backbone.prepare_tokens(x)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)

        # Remove CLS token, reshape to spatial
        patch_tokens = tokens[:, 1:, :]  # [B, N_patches, D]
        B, N, D = patch_tokens.shape
        patch_size = getattr(self.backbone.patch_embed, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        h = x.shape[2] // patch_size
        w = x.shape[3] // patch_size
        return patch_tokens.transpose(1, 2).reshape(B, D, h, w)

    def forward(self, x):
        """Forward pass matching FeatureNet output format.

        Args:
            x: [B, C, H, W] input image (RGB 3ch or RGBIR 4ch)

        Returns:
            dict with keys "stage1", "stage2", "stage3"
        """
        H, W = x.shape[2], x.shape[3]
        feat = self._extract_feature_map(x)  # [B, D, H/ps, W/ps] where ps=patch_size

        # stage1: 1/4 resolution
        stage1 = F.interpolate(feat, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        stage1 = self.proj_stage1(stage1)

        # stage2: 1/2 resolution
        stage2 = F.interpolate(feat, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
        stage2 = self.proj_stage2(stage2)

        # stage3: full resolution
        stage3 = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        stage3 = self.proj_stage3(stage3)

        return {"stage1": stage1, "stage2": stage2, "stage3": stage3}


# ---------------------------------------------------------------------------
# 2. Build DINO-MM CasMVSNet
# ---------------------------------------------------------------------------

def _import_cascade_mvsnet():
    """Import CascadeMVSNet from baselines.

    module.py does ``from utils import local_pcd`` at the top, so the
    CasMVSNet repo root must be on sys.path *before* the module is loaded.
    We add it temporarily and clean up afterwards.
    """
    cas_root = Path("/autodl-fs/data/baselines/cas_mvsnet/CasMVSNet")
    cas_root_str = str(cas_root)

    already_on_path = cas_root_str in sys.path
    if not already_on_path:
        sys.path.insert(0, cas_root_str)

    try:
        import importlib
        import types

        models_dir = cas_root / "models"

        # Load module.py (needs ``utils`` importable from cas_root)
        spec_m = importlib.util.spec_from_file_location(
            "_casmvs_module", models_dir / "module.py"
        )
        mod_m = importlib.util.module_from_spec(spec_m)
        spec_m.loader.exec_module(mod_m)

        # Create a fake package so the relative import in cas_mvsnet.py works
        pkg = types.ModuleType("_casmvs_models")
        pkg.__path__ = [str(models_dir)]
        pkg.__package__ = "_casmvs_models"
        pkg.module = mod_m
        sys.modules["_casmvs_models"] = pkg
        sys.modules["_casmvs_models.module"] = mod_m

        # Load cas_mvsnet.py
        spec_c = importlib.util.spec_from_file_location(
            "_casmvs_models.cas_mvsnet",
            models_dir / "cas_mvsnet.py",
            submodule_search_locations=[],
        )
        mod_c = importlib.util.module_from_spec(spec_c)
        mod_c.__package__ = "_casmvs_models"
        sys.modules["_casmvs_models.cas_mvsnet"] = mod_c
        spec_c.loader.exec_module(mod_c)

        return mod_c.CascadeMVSNet
    finally:
        if not already_on_path and cas_root_str in sys.path:
            sys.path.remove(cas_root_str)


def build_dino_mm_casmvsnet(
    checkpoint_path,
    checkpoint_key="teacher",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    ndepths=(48, 32, 8),
    depth_interals_ratio=(4, 2, 1),
    cr_base_chs=(8, 8, 8),
    device="cpu",
):
    """Build CascadeMVSNet with DINO-MM feature extractor."""
    from downstream.common import build_backbone_from_checkpoint, freeze_backbone

    CascadeMVSNet = _import_cascade_mvsnet()

    # Build DINO-MM backbone
    backbone, _, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device="cpu",
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)

    # Build CasMVSNet with default FeatureNet
    model = CascadeMVSNet(
        ndepths=list(ndepths),
        depth_interals_ratio=list(depth_interals_ratio),
        cr_base_chs=list(cr_base_chs),
        arch_mode="fpn",
        grad_method="detach",
    )

    # Replace FeatureNet with DINO-MM adapter
    dino_feature = DinoMMMVSFeatureNet(backbone, target_channels=(32, 16, 8))
    model.feature = dino_feature

    model.to(device)
    meta["task"] = "3d_reconstruction"
    return model, meta


# ---------------------------------------------------------------------------
# 3. MVS Dataset for LuoJia-MVS / WHU
# ---------------------------------------------------------------------------

def _read_pfm(filename):
    """Read PFM depth map file."""
    with open(filename, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {filename}")

        dim_match = f.readline().decode("ascii").strip().split()
        width, height = int(dim_match[0]), int(dim_match[1])
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data, scale


def _read_cam_file(filename):
    """Read camera parameters from Cas-MVSNet-style or WHU sub-block camera files."""
    with open(filename, "r") as f:
        raw_lines = f.readlines()

    lines = [line.strip() for line in raw_lines if line.strip()]
    if lines and lines[0].lower() == "extrinsic":
        extrinsic = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            extrinsic[i] = [float(v) for v in lines[1 + i].split()]

        intrinsic_values = [float(v) for v in lines[5].split()]
        if len(intrinsic_values) == 3:
            focal, cx, cy = intrinsic_values
            intrinsic = np.array(
                [[-focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        elif len(intrinsic_values) == 9:
            intrinsic = np.array(intrinsic_values, dtype=np.float32).reshape(3, 3)
        else:
            raise ValueError(f"Unsupported intrinsic format in camera file: {filename}")

        depth_info = [float(v) for v in lines[6].split()]
        depth_min = depth_info[0]
        depth_max = depth_info[1] if len(depth_info) > 1 else None
        depth_interval = depth_info[2] if len(depth_info) > 2 else None
        if depth_interval is None:
            depth_interval = 1.0
        return extrinsic, intrinsic, depth_min, depth_interval, depth_max

    # Cas-MVSNet / DTU style camera file.
    extrinsic = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        values = raw_lines[1 + i].strip().split()
        extrinsic[i] = [float(v) for v in values]

    intrinsic = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        values = raw_lines[7 + i].strip().split()
        intrinsic[i] = [float(v) for v in values]

    depth_info = raw_lines[11].strip().split()
    depth_min = float(depth_info[0])
    depth_interval = float(depth_info[1])
    return extrinsic, intrinsic, depth_min, depth_interval, None


def _read_depth_map(filename, png_scale=1.0):
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == ".pfm":
        depth, _ = _read_pfm(str(path))
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth map: {filename}")
    depth = depth.astype(np.float32)
    if png_scale not in (0.0, 1.0):
        depth = depth / png_scale
    return depth


class MVSDataset(Dataset):
    """Generic MVS dataset for LuoJia-MVS / WHU / DTU format.

    Expected directory layout::

        root/
        ├── images/         or Rectified/
        │   └── {scene}/
        │       └── *.png / *.jpg
        ├── cams/           or Cameras/
        │   └── {scene}/
        │       └── *_cam.txt
        ├── depths/         or Depths/
        │   └── {scene}/
        │       └── *.pfm
        └── pair.txt        (per scene or global)

    WHU cropped sub-block layout is also supported::

        root/
        ├── train/ or test/
        │   ├── Images/{scene}/{view_id}/{frame_id}.png
        │   ├── Cams/{scene}/{view_id}/{frame_id}.txt
        │   ├── Depths/{scene}/{view_id}/{frame_id}.png
        │   ├── index.txt
        │   └── pair.txt
    """

    def __init__(
        self,
        root,
        split="train",
        num_views=5,
        num_depth=192,
        interval_scale=1.06,
        max_h=512,
        max_w=640,
        scene_list=None,
    ):
        self.root = Path(root)
        self.split = split
        self.num_views = num_views
        self.num_depth = num_depth
        self.interval_scale = interval_scale
        self.max_h = max_h
        self.max_w = max_w
        self.data_root = self._resolve_split_root(split)
        self.dataset_kind = self._detect_dataset_kind()

        if scene_list is not None:
            with open(scene_list, "r") as f:
                self.scenes = [line.strip() for line in f if line.strip()]
        elif self.dataset_kind == "whu_mvs":
            index_file = self.data_root / "index.txt"
            if index_file.exists():
                self.scenes = [line.strip() for line in index_file.read_text().splitlines() if line.strip()]
            else:
                images_root = self.data_root / "Images"
                self.scenes = sorted([path.name for path in images_root.iterdir() if path.is_dir()])
        else:
            img_root = self._find_dir("images", "Images", "Rectified")
            if img_root and img_root.is_dir():
                self.scenes = sorted([d.name for d in img_root.iterdir() if d.is_dir()])
            else:
                self.scenes = ["."]

        self.samples = self._build_samples()

    def _resolve_split_root(self, split):
        split_aliases = {
            "train": ("train",),
            "val": ("val", "test"),
            "test": ("test", "val"),
        }
        aliases = split_aliases.get(split, (split,))
        if self.root.name.lower() in aliases:
            return self.root
        for alias in aliases:
            candidate = self.root / alias
            if candidate.exists():
                return candidate
        return self.root

    def _detect_dataset_kind(self):
        if (self.data_root / "Images").exists() and (self.data_root / "Cams").exists():
            return "whu_mvs"
        return "generic"

    def _build_samples(self):
        if self.dataset_kind == "whu_mvs":
            return self._build_whu_samples()
        return self._build_generic_samples()

    def _build_generic_samples(self):
        samples = []
        for scene in self.scenes:
            pair_path = self._find_pair_file(scene)
            if pair_path is None:
                continue
            pairs = self._parse_pair_file(pair_path)
            for ref_view, src_views in pairs:
                if len(src_views) < self.num_views - 1:
                    continue
                samples.append(
                    {
                        "scene": scene,
                        "ref_view": ref_view,
                        "src_views": src_views[: self.num_views - 1],
                        "frame_id": None,
                    }
                )
        return samples

    def _build_whu_samples(self):
        pair_path = self.data_root / "pair.txt"
        if not pair_path.exists():
            return []

        pairs = self._parse_pair_file(pair_path)
        samples = []
        for scene in self.scenes:
            for ref_view, src_views in pairs:
                selected_views = [ref_view] + src_views[: self.num_views - 1]
                if len(selected_views) != self.num_views:
                    continue
                frame_ids = self._list_frame_ids(scene, ref_view)
                for frame_id in frame_ids:
                    if self._frame_exists(scene, selected_views, frame_id):
                        samples.append(
                            {
                                "scene": scene,
                                "ref_view": ref_view,
                                "src_views": selected_views[1:],
                                "frame_id": frame_id,
                            }
                        )
        return samples

    def _find_dir(self, *candidates):
        for name in candidates:
            path = self.data_root / name
            if path.exists():
                return path
        return None

    def _find_pair_file(self, scene):
        for name in ["pair.txt", "Cameras/pair.txt"]:
            p = self.data_root / scene / name
            if p.exists():
                return p
            p = self.data_root / name
            if p.exists():
                return p
        return None

    def _parse_pair_file(self, path):
        """Parse pair.txt with or without source-view scores."""
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        pairs = []
        idx = 1  # skip first line (num viewpoints)
        while idx < len(lines):
            ref_view = int(lines[idx])
            idx += 1
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if not parts:
                idx += 1
                continue
            num_src = int(parts[0])
            values = parts[1:]
            src_views = []
            if len(values) >= num_src * 2:
                for i in range(num_src):
                    src_views.append(int(values[2 * i]))
            else:
                for value in values[:num_src]:
                    src_views.append(int(value))
            pairs.append((ref_view, src_views))
            idx += 1
        return pairs

    def _list_frame_ids(self, scene, ref_view):
        view_dir = self.data_root / "Images" / scene / str(ref_view)
        if not view_dir.exists():
            return []
        return sorted(path.stem for path in view_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def _frame_exists(self, scene, view_ids, frame_id):
        for view_id in view_ids:
            if self._find_image(scene, view_id, frame_id) is None:
                return False
            if self._find_cam(scene, view_id, frame_id) is None:
                return False
        return self._find_depth(scene, view_ids[0], frame_id) is not None

    def _find_image(self, scene, view_id, frame_id=None):
        if self.dataset_kind == "whu_mvs":
            for ext in (".png", ".jpg", ".jpeg"):
                path = self.data_root / "Images" / scene / str(view_id) / f"{frame_id}{ext}"
                if path.exists():
                    return path
            return None

        for pattern in [
            f"{scene}/images/{view_id:08d}.jpg",
            f"{scene}/images/{view_id:08d}.png",
            f"{scene}/Images/{view_id:08d}.jpg",
            f"{scene}/Images/{view_id:08d}.png",
            f"images/{scene}/{view_id:08d}.jpg",
            f"images/{scene}/{view_id:08d}.png",
            f"Images/{scene}/{view_id:08d}.jpg",
            f"Images/{scene}/{view_id:08d}.png",
            f"Rectified/{scene}/rect_{view_id:03d}_3_r5000.png",
        ]:
            path = self.data_root / pattern
            if path.exists():
                return path
        return None

    def _find_cam(self, scene, view_id, frame_id=None):
        if self.dataset_kind == "whu_mvs":
            path = self.data_root / "Cams" / scene / str(view_id) / f"{frame_id}.txt"
            return path if path.exists() else None

        for pattern in [
            f"{scene}/cams/{view_id:08d}_cam.txt",
            f"{scene}/Cams/{view_id:08d}_cam.txt",
            f"cams/{scene}/{view_id:08d}_cam.txt",
            f"Cams/{scene}/{view_id:08d}_cam.txt",
            f"Cameras/{view_id:08d}_cam.txt",
        ]:
            path = self.data_root / pattern
            if path.exists():
                return path
        return None

    def _find_depth(self, scene, view_id, frame_id=None):
        if self.dataset_kind == "whu_mvs":
            path = self.data_root / "Depths" / scene / str(view_id) / f"{frame_id}.png"
            return path if path.exists() else None

        for pattern in [
            f"{scene}/depths/{view_id:08d}.pfm",
            f"depths/{scene}/{view_id:08d}.pfm",
            f"Depths/{scene}/depth_map_{view_id:04d}.pfm",
            f"Depths/{scene}/{view_id:08d}.pfm",
            f"Depths/{scene}/{view_id:08d}.png",
        ]:
            path = self.data_root / pattern
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        scene = sample["scene"]
        frame_id = sample.get("frame_id")
        view_ids = [sample["ref_view"]] + sample["src_views"]

        imgs = []
        proj_matrices = {"stage1": [], "stage2": [], "stage3": []}
        ref_depth_min = None
        ref_depth_max = None
        ref_depth_interval = None

        for view_id in view_ids:
            # Load image
            img_path = self._find_image(scene, view_id, frame_id=frame_id)
            if img_path is None:
                raise FileNotFoundError(f"Image not found: scene={scene}, view={view_id}")
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            img = cv2.resize(img, (self.max_w, self.max_h))
            img = img.astype(np.float32) / 255.0
            ir = np.zeros((self.max_h, self.max_w, 1), dtype=np.float32)
            imgs.append(np.concatenate([img, ir], axis=2))

            # Load camera
            cam_path = self._find_cam(scene, view_id, frame_id=frame_id)
            if cam_path is None:
                raise FileNotFoundError(f"Camera not found: scene={scene}, view={view_id}")
            extrinsic, intrinsic, depth_min, depth_interval, depth_max = _read_cam_file(str(cam_path))

            scale_x = self.max_w / float(orig_w)
            scale_y = self.max_h / float(orig_h)
            intrinsic_resized = intrinsic.copy()
            intrinsic_resized[0, 0] *= scale_x
            intrinsic_resized[0, 2] *= scale_x
            intrinsic_resized[1, 1] *= scale_y
            intrinsic_resized[1, 2] *= scale_y

            if view_id == sample["ref_view"]:
                ref_depth_min = depth_min
                ref_depth_max = depth_max
                ref_depth_interval = depth_interval

            for stage_key, scale in [("stage1", 4.0), ("stage2", 2.0), ("stage3", 1.0)]:
                intrinsic_scaled = intrinsic_resized.copy()
                intrinsic_scaled[0, 0] /= scale
                intrinsic_scaled[0, 2] /= scale
                intrinsic_scaled[1, 1] /= scale
                intrinsic_scaled[1, 2] /= scale
                proj_mat = np.zeros((2, 4, 4), dtype=np.float32)
                proj_mat[0] = extrinsic
                proj_mat[1, :3, :3] = intrinsic_scaled
                proj_mat[1, 3, 3] = 1.0
                proj_matrices[stage_key].append(torch.from_numpy(proj_mat))

        # Stack RGB plus a zero IR channel: [N, 4, H, W]
        imgs_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()

        # Stack proj matrices: [N, 2, 4, 4]
        for key in proj_matrices:
            proj_matrices[key] = torch.stack(proj_matrices[key])

        # Load depth GT for ref view
        depth_path = self._find_depth(scene, sample["ref_view"], frame_id=frame_id)
        if depth_path is not None:
            png_scale = 64.0 if self.dataset_kind == "whu_mvs" else 1.0
            depth_gt = _read_depth_map(str(depth_path), png_scale=png_scale)
            depth_gt = cv2.resize(depth_gt, (self.max_w, self.max_h), interpolation=cv2.INTER_NEAREST)
            depth_gt = torch.from_numpy(depth_gt).float()
            mask = (depth_gt > 0).float()
        else:
            depth_gt = torch.zeros(self.max_h, self.max_w, dtype=torch.float32)
            mask = torch.zeros(self.max_h, self.max_w, dtype=torch.float32)

        # Build multi-scale depth GT and mask
        depth_gt_ms = {}
        mask_ms = {}
        for stage_key, scale in [("stage1", 4.0), ("stage2", 2.0), ("stage3", 1.0)]:
            sh, sw = int(self.max_h / scale), int(self.max_w / scale)
            depth_gt_ms[stage_key] = F.interpolate(
                depth_gt.unsqueeze(0).unsqueeze(0), size=(sh, sw), mode="nearest"
            ).squeeze()
            mask_ms[stage_key] = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(sh, sw), mode="nearest"
            ).squeeze()

        # Build depth_values (uniform sampling)
        if ref_depth_min is None or ref_depth_interval is None:
            cam_path_ref = self._find_cam(scene, sample["ref_view"], frame_id=frame_id)
            _, _, ref_depth_min, ref_depth_interval, ref_depth_max = _read_cam_file(str(cam_path_ref))
        ref_depth_interval *= self.interval_scale
        if ref_depth_max is not None:
            depth_values = torch.linspace(
                ref_depth_min,
                ref_depth_max,
                steps=self.num_depth,
                dtype=torch.float32,
            )
        else:
            depth_values = (
                torch.arange(0, self.num_depth, dtype=torch.float32) * ref_depth_interval + ref_depth_min
            )

        return {
            "imgs": imgs_tensor,
            "proj_matrices": proj_matrices,
            "depth_values": depth_values,
            "depth_gt_ms": depth_gt_ms,
            "mask_ms": mask_ms,
            "filename": f"{scene}/{frame_id}" if frame_id is not None else f"{scene}/{sample['ref_view']:08d}",
        }


# ---------------------------------------------------------------------------
# 4. Loss and Metrics
# ---------------------------------------------------------------------------

def mvs_loss(outputs, depth_gt_ms, mask_ms, dlossw=(0.5, 1.0, 2.0)):
    """Multi-stage depth loss (same as cas_mvsnet_loss)."""
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device)
    for stage_idx, stage_key in enumerate(["stage1", "stage2", "stage3"]):
        if stage_key not in outputs:
            continue
        depth_est = outputs[stage_key]["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5
        if mask.sum() == 0:
            continue
        loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction="mean")
        total_loss = total_loss + dlossw[stage_idx] * loss
    return total_loss


def compute_mvs_metrics(depth_pred, depth_gt, mask, interval=1.0):
    """Compute MAE, <0.6m accuracy, <3-interval accuracy."""
    valid = mask > 0.5
    if valid.sum() == 0:
        return {"mae": 0.0, "acc_06m": 0.0, "acc_3interval": 0.0}

    diff = torch.abs(depth_pred[valid] - depth_gt[valid])
    mae = diff.mean().item()
    acc_06m = (diff < 0.6).float().mean().item() * 100.0
    acc_3interval = (diff < 3.0 * interval).float().mean().item() * 100.0
    return {"mae": mae, "acc_06m": acc_06m, "acc_3interval": acc_3interval}

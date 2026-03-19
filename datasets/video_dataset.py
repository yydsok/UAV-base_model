"""
Video dataset classes for temporal pretraining.

UAVHumanVideoDataset: paired RGB+IR .avi video clips from UAV-Human
AerialRGBTSequenceDataset: paired RGB png + thermal16 tiff image sequences
FrameSequenceDataset: generic frame-extracted video sequences (RGB-IR / RGB / IR)
collate_video: batches (frames, masks, timestamps) tuples

Each __getitem__ returns:
    frames: [T, 4, H, W] float tensor (normalized)
    modality_mask: [2] float tensor ([1, 1] = both modalities present)
    timestamps: [T] float tensor (seconds)
"""

import json
import os
import sys
import re
import random
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transforms_video import VideoAugmentation
from models.temporal_module import VideoFrameSampler

logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
_INT_RE = re.compile(r'(\d+)')


def _load_video_entries(manifest_path):
    """Load video entries from legacy or split-schema manifests."""
    with open(manifest_path, 'r') as f:
        raw_manifest = json.load(f)

    if isinstance(raw_manifest, list):
        return raw_manifest
    if isinstance(raw_manifest, dict):
        if 'video_sequences' in raw_manifest:
            return raw_manifest['video_sequences']
        return raw_manifest.get('samples', raw_manifest.get('data', []))
    raise ValueError(f"Unsupported manifest format: {type(raw_manifest).__name__}")


def _natural_sort_key(name):
    """Sort strings like 1, 2, 10 instead of lexical 1, 10, 2."""
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


def _extract_last_int(stem):
    matches = _INT_RE.findall(stem)
    if not matches:
        return None
    return int(matches[-1])


class UAVHumanVideoDataset(Dataset):
    """
    Paired RGB+IR video dataset from UAV-Human.

    Each sample is a pair of .avi files (RGB 1920x1080@60fps, IR 640x480@25fps).
    Sampling is done on the IR timeline (lower fps), with RGB frames computed
    proportionally.
    """

    def __init__(self, manifest_path, num_frames=4, target_size=224,
                 min_delta=1, max_delta=30):
        all_entries = _load_video_entries(manifest_path)

        # Filter to entries with paired video files (uav_human, llvip, etc.)
        self.entries = [e for e in all_entries
                        if 'rgb_video_path' in e
                        and 'ir_video_path' in e]

        self.num_frames = num_frames
        self.target_size = target_size
        self.transform = VideoAugmentation(target_size=target_size)
        self.sampler = VideoFrameSampler(
            num_frames=num_frames, min_delta=min_delta,
            max_delta=max_delta, fps=25.0)  # IR fps as base

        logger.info(f"UAVHumanVideoDataset: {len(self.entries)} clips, "
                    f"num_frames={num_frames}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        rgb_path = entry['rgb_video_path']
        ir_path = entry['ir_video_path']
        ir_fps = entry.get('ir_fps', 25.0)
        rgb_fps = entry.get('rgb_fps', 60.0)
        ir_frame_count = entry['ir_frame_count']

        # Sample frame indices on IR timeline
        ir_indices, timestamps = self.sampler.sample(ir_frame_count)

        # Compute corresponding RGB frame indices (proportional mapping)
        fps_ratio = rgb_fps / ir_fps if ir_fps > 0 else 2.4
        rgb_indices = [min(int(i * fps_ratio), entry.get('rgb_frame_count', 999999) - 1)
                       for i in ir_indices]

        # Read frames
        frames = self._read_paired_frames(
            rgb_path, ir_path, rgb_indices, ir_indices)

        # Apply augmentation (shared flips + sensor drop across frames)
        frames, aug_mask = self.transform(frames, base_modality_mask=[1.0, 1.0])

        # Stack: list of [4,H,W] -> [T, 4, H, W]
        frames_tensor = torch.stack(frames, dim=0)
        modality_mask = torch.tensor(aug_mask, dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

        return frames_tensor, modality_mask, timestamps_tensor

    def _read_paired_frames(self, rgb_path, ir_path, rgb_indices, ir_indices):
        """Read and pair RGB+IR frames, returning list of [H,W,4] numpy arrays."""
        frames = []

        rgb_cap = cv2.VideoCapture(rgb_path)
        ir_cap = cv2.VideoCapture(ir_path)

        for rgb_idx, ir_idx in zip(rgb_indices, ir_indices):
            # Read RGB frame
            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
            ret_rgb, rgb_frame = rgb_cap.read()
            if not ret_rgb:
                rgb_frame = np.zeros((self.target_size, self.target_size, 3),
                                     dtype=np.uint8)
            else:
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(
                    rgb_frame, (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR)

            # Read IR frame
            ir_cap.set(cv2.CAP_PROP_POS_FRAMES, ir_idx)
            ret_ir, ir_frame = ir_cap.read()
            if not ret_ir:
                ir_frame = np.zeros((self.target_size, self.target_size, 1),
                                    dtype=np.uint8)
            else:
                if len(ir_frame.shape) == 3:
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
                ir_frame = cv2.resize(
                    ir_frame, (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR)
                ir_frame = ir_frame[:, :, np.newaxis]

            # Concatenate: [H, W, 4]
            combined = np.concatenate([rgb_frame, ir_frame], axis=2)
            frames.append(combined)

        rgb_cap.release()
        ir_cap.release()

        return frames


class AerialRGBTSequenceDataset(Dataset):
    """
    Paired RGB+thermal image sequence dataset from aerial-rgbt.

    Each sample is a sequence directory with matched color/*.png and
    thermal16/*.tiff frames.
    """

    def __init__(self, manifest_path, num_frames=4, target_size=224):
        all_entries = _load_video_entries(manifest_path)

        # Filter to aerial_rgbt entries only
        self.entries = [e for e in all_entries
                        if e.get('dataset') == 'aerial_rgbt'
                        and 'paired_frames' in e]

        self.num_frames = num_frames
        self.target_size = target_size
        self.transform = VideoAugmentation(target_size=target_size)

        logger.info(f"AerialRGBTSequenceDataset: {len(self.entries)} sequences, "
                    f"num_frames={num_frames}")

    def __len__(self):
        return len(self.entries)

    def _empty_sequence(self):
        zero = np.zeros((self.target_size, self.target_size, 4), dtype=np.uint8)
        frames, aug_mask = self.transform(
            [zero.copy() for _ in range(self.num_frames)],
            base_modality_mask=[1.0, 1.0])
        return (torch.stack(frames, dim=0),
                torch.tensor(aug_mask, dtype=torch.float32),
                torch.zeros(self.num_frames, dtype=torch.float32))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        paired_frames = entry['paired_frames']

        if len(paired_frames) == 0:
            return self._empty_sequence()

        # Sample frames from available indices
        if len(paired_frames) >= self.num_frames:
            selected = sorted(random.sample(range(len(paired_frames)),
                                            self.num_frames))
        else:
            # Pad by repeating last frame
            selected = list(range(len(paired_frames)))
            while len(selected) < self.num_frames:
                selected.append(selected[-1])

        frames = []
        timestamps = []

        for i in selected:
            pair = paired_frames[i]
            rgb_path = pair['rgb_path']
            ir_path = pair['ir_path']
            frame_index = pair['frame_index']

            # Read RGB
            rgb_frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_frame is None:
                rgb_frame = np.zeros((self.target_size, self.target_size, 3),
                                     dtype=np.uint8)
            else:
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(
                    rgb_frame, (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR)

            # Read thermal16 TIFF (16-bit) -> normalize to uint8
            ir_frame = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
            if ir_frame is None:
                ir_frame = np.zeros((self.target_size, self.target_size),
                                    dtype=np.uint8)
            else:
                # Normalize 16-bit to 8-bit
                if ir_frame.dtype == np.uint16:
                    ir_min, ir_max = ir_frame.min(), ir_frame.max()
                    if ir_max > ir_min:
                        ir_frame = ((ir_frame - ir_min) / (ir_max - ir_min) * 255).astype(np.uint8)
                    else:
                        ir_frame = np.zeros_like(ir_frame, dtype=np.uint8)
                if len(ir_frame.shape) == 3:
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
                ir_frame = cv2.resize(
                    ir_frame, (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR)

            ir_frame = ir_frame[:, :, np.newaxis]

            # Concatenate: [H, W, 4]
            combined = np.concatenate([rgb_frame, ir_frame], axis=2)
            frames.append(combined)

            # Timestamps: use frame_index * 6 seconds (approx interval)
            timestamps.append(frame_index * 6.0 / 180.0)

        # Apply augmentation
        frames, aug_mask = self.transform(frames, base_modality_mask=[1.0, 1.0])

        frames_tensor = torch.stack(frames, dim=0)  # [T, 4, H, W]
        modality_mask = torch.tensor(aug_mask, dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

        return frames_tensor, modality_mask, timestamps_tensor


class FrameSequenceDataset(Dataset):
    """
    Generic frame-extracted video sequence dataset.

    Supports three modality types:
    - rgbir_paired: matched RGB + IR frame directories
    - rgb_only: RGB frames only (IR channel filled with zeros)
    - ir_only: IR frames only (RGB channels filled with zeros)

    Manifest entry format:
    {
        "dataset": "<dataset_name>",
        "modality": "rgbir_paired" | "rgb_only" | "ir_only",
        "frame_dir": "<path>",              # for rgb_only / ir_only
        "rgb_dir": "<path>",                # for rgbir_paired
        "ir_dir": "<path>",                 # for rgbir_paired
        "num_frames": <int>,                # total frames in sequence
        "fps": <float>                      # estimated fps (default 30)
    }
    """

    # Datasets that use this class
    SUPPORTED_DATASETS = {
        'utuav', 'rgbt_tiny', 'vt_tiny_mot', 'm3ot',
        'webuav_3m', 'visdrone_sot', 'visdrone_mot', 'visdrone_cc',
        'visdrone_det',
        'uav123', 'uavdt', 'dronecrowd', 'dtb70', 'mdmt',
        'seadronessee_mot', 'uav_human_pose',
        'monet',
        'aeroscapes', 'animaldrone', 'au_air',
        'sues_200', 'uavid', 'uavscenes', 'tardal',
        'udd', 'manipal_uav',
    }

    def __init__(self, manifest_path, num_frames=4, target_size=224,
                 min_delta=1, max_delta=30, alignment_offsets_path=None):
        all_entries = _load_video_entries(manifest_path)

        # Filter to frame_sequence entries with sufficient frames
        self.entries = [e for e in all_entries
                        if e.get('dataset', '') in self.SUPPORTED_DATASETS
                        and e.get('modality') in ('rgbir_paired', 'rgb_only', 'ir_only')
                        and e.get('num_frames', 0) >= 2]

        self.num_frames = num_frames
        self.target_size = target_size
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.transform = VideoAugmentation(target_size=target_size)
        self._frame_cache = {}
        self._frame_cache_max = 10000

        # Load alignment offsets for RGB-IR registration
        self._alignment_per_image = {}
        self._alignment_per_seq = {}
        self._entry_offset_cache = {}
        if alignment_offsets_path and os.path.exists(alignment_offsets_path):
            import json as _json
            with open(alignment_offsets_path, 'r') as f:
                align_data = _json.load(f)
            self._alignment_per_image = align_data.get('per_image_offsets', {})
            self._alignment_per_seq = align_data.get('sequence_offsets', {})
            logger.info(f"FrameSequenceDataset: alignment offsets loaded "
                        f"({len(self._alignment_per_seq)} seq, "
                        f"{len(self._alignment_per_image)} per-image)")

        logger.info(f"FrameSequenceDataset: {len(self.entries)} sequences, "
                    f"num_frames={num_frames}")

    @staticmethod
    def _sorted_image_files(directory, name_prefix=None, name_regex=None):
        """List and sort image files in a directory."""
        pattern = re.compile(name_regex) if name_regex else None
        try:
            files = []
            for f in os.listdir(directory):
                if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
                    continue
                if name_prefix and not f.startswith(name_prefix):
                    continue
                if pattern and not pattern.match(f):
                    continue
                files.append(f)
        except OSError:
            return []
        files.sort(key=_natural_sort_key)
        return [os.path.join(directory, f) for f in files]

    @staticmethod
    def _infer_frame_values(file_paths):
        """
        Infer frame indices (or timestamps) from file names.
        Returns a list aligned with file_paths, or None if parsing is unreliable.
        """
        if not file_paths:
            return None

        stems = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

        # Priority 1: pure numeric timestamp/frame id (e.g. 000123, 1658137057.64)
        pure_numeric = []
        pure_ok = True
        for s in stems:
            if re.fullmatch(r'\d+(?:\.\d+)?', s):
                pure_numeric.append(float(s))
            else:
                pure_ok = False
                break
        if pure_ok:
            return pure_numeric

        # Priority 2: last integer token (e.g. frame_x_000123)
        values = []
        for s in stems:
            v = _extract_last_int(s)
            if v is None:
                return None
            values.append(float(v))
        return values

    @staticmethod
    def _estimate_stride(frame_values):
        """Estimate nominal frame stride from parsed frame values."""
        if frame_values is None or len(frame_values) < 2:
            return 1.0
        diffs = []
        for a, b in zip(frame_values, frame_values[1:]):
            d = b - a
            if d > 0:
                diffs.append(d)
        if not diffs:
            return 1.0
        diffs.sort()
        mid = len(diffs) // 2
        if len(diffs) % 2 == 1:
            return float(diffs[mid])
        return float((diffs[mid - 1] + diffs[mid]) / 2.0)

    def _list_frames(self, entry):
        """Build sorted frame path lists for a manifest entry (called per __getitem__)."""
        cache_key = (
            entry.get('modality'),
            entry.get('rgb_dir'),
            entry.get('ir_dir'),
            entry.get('frame_dir'),
        )
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        # Evict oldest half when cache exceeds limit
        if len(self._frame_cache) >= self._frame_cache_max:
            keys = list(self._frame_cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._frame_cache[k]

        modality = entry['modality']
        if modality == 'rgbir_paired':
            rgb_files = self._sorted_image_files(
                entry['rgb_dir'],
                name_prefix=entry.get('rgb_prefix'),
                name_regex=entry.get('rgb_regex'))
            ir_files = self._sorted_image_files(
                entry['ir_dir'],
                name_prefix=entry.get('ir_prefix'),
                name_regex=entry.get('ir_regex'))
            n = min(len(rgb_files), len(ir_files))
            rgb_files = rgb_files[:n]
            ir_files = ir_files[:n]
            frame_values = self._infer_frame_values(rgb_files)
            out = {'rgb': rgb_files, 'ir': ir_files, 'frame_values': frame_values}
        else:
            frame_files = self._sorted_image_files(
                entry['frame_dir'],
                name_prefix=entry.get('filename_prefix'),
                name_regex=entry.get('filename_regex'))
            frame_values = self._infer_frame_values(frame_files)
            out = {'frames': frame_files, 'frame_values': frame_values}

        self._frame_cache[cache_key] = out
        return out

    def __len__(self):
        return len(self.entries)

    def _empty_sequence(self, modality):
        zero = np.zeros((self.target_size, self.target_size, 4), dtype=np.uint8)
        base_mask = {
            'rgbir_paired': torch.tensor([1.0, 1.0], dtype=torch.float32),
            'rgb_only': torch.tensor([1.0, 0.0], dtype=torch.float32),
            'ir_only': torch.tensor([0.0, 1.0], dtype=torch.float32),
        }.get(modality, torch.tensor([1.0, 1.0], dtype=torch.float32))
        frames, aug_mask = self.transform(
            [zero.copy() for _ in range(self.num_frames)],
            base_modality_mask=base_mask)
        modality_mask = torch.tensor(aug_mask, dtype=torch.float32)
        return (torch.stack(frames, dim=0),
                modality_mask,
                torch.zeros(self.num_frames, dtype=torch.float32))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        modality = entry['modality']
        fps = entry.get('fps', 30.0)
        fl = self._list_frames(entry)

        if modality == 'rgbir_paired':
            return self._get_paired(entry, fl, fps)
        elif modality == 'rgb_only':
            return self._get_single(entry, fl, fps, is_rgb=True)
        else:  # ir_only
            return self._get_single(entry, fl, fps, is_rgb=False)

    def _sample_positions(self, total, entry, frame_values):
        """Sample position indices on available frames, adapting by sequence stride."""
        if total <= 0:
            return [0] * self.num_frames

        if total < self.num_frames:
            pos = list(range(total))
            while len(pos) < self.num_frames:
                pos.append(pos[-1])
            return pos

        stride = float(entry.get('sequence_stride') or 1.0)
        est = self._estimate_stride(frame_values)
        if est > 0 and (stride <= 1.0):
            stride = est

        # Use manifest hints first; fallback to global constructor defaults.
        min_raw = float(entry.get('sample_min_delta', self.min_delta))
        max_raw = float(entry.get('sample_max_delta', self.max_delta))
        already_subsampled = bool(entry.get('already_subsampled', False))

        # Convert raw-frame delta to available-frame delta.
        denom = max(1.0, stride)
        min_step = max(1, int(round(min_raw / denom)))
        max_step = max(min_step, int(round(max_raw / denom)))

        # If dataset has been frame-subsampled already, keep hops conservative.
        if already_subsampled:
            max_step = min(max_step, max(2, self.num_frames * 2))

        max_step = min(max_step, max(1, total - 1))
        min_step = min(min_step, max_step)

        # Build a monotonic sequence with random temporal gaps.
        start_upper = max(0, total - 1 - (self.num_frames - 1) * min_step)
        positions = [random.randint(0, start_upper)]
        for _ in range(self.num_frames - 1):
            d = random.randint(min_step, max_step)
            positions.append(min(total - 1, positions[-1] + d))
        return positions

    def _build_timestamps(self, positions, frame_values, fps):
        """Build relative timestamps (seconds) from sampled positions."""
        if not positions:
            return []

        # Fallback: positional timestamps.
        if frame_values is None or len(frame_values) == 0:
            return [float(p) / float(max(fps, 1e-6)) for p in positions]

        vals = [frame_values[min(max(p, 0), len(frame_values) - 1)] for p in positions]
        base = vals[0]
        if base is None:
            return [float(p) / float(max(fps, 1e-6)) for p in positions]

        # Pure numeric timestamps with decimals are treated as seconds (e.g. UAVScenes).
        has_decimal = any(abs(v - round(v)) > 1e-6 for v in vals)
        if has_decimal:
            return [float(v - base) for v in vals]

        # Otherwise treat as frame ids and convert with fps.
        fps_safe = float(max(fps, 1e-6))
        return [float(v - base) / fps_safe for v in vals]

    def _read_rgb(self, path):
        """Read an RGB image, return [H, W, 3] uint8."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_ir(self, path):
        """Read an IR image, return [H, W, 1] uint8."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((self.target_size, self.target_size, 1), dtype=np.uint8)
        # Handle 16-bit
        if img.dtype == np.uint16:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = ((img.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        # Multi-channel -> grayscale
        if img.ndim == 3 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img

    def _get_entry_offset(self, entry):
        """Get alignment offset (dx, dy) for a video sequence entry.

        For UTUAV: uses sequence_offsets directly.
        For others: samples per-image offsets from the IR directory, takes median.
        """
        entry_id = id(entry)
        if entry_id in self._entry_offset_cache:
            return self._entry_offset_cache[entry_id]

        dx, dy = 0.0, 0.0
        dataset = entry.get('dataset', '')
        seq_name = entry.get('sequence', '')

        # 1. Try sequence offsets (UTUAV)
        # Map dataset name variants to canonical form
        ds_map = {'utuav': 'UTUAV'}
        canonical = ds_map.get(dataset, '')
        if canonical and seq_name:
            key = f"{canonical}/{seq_name}"
            if key in self._alignment_per_seq:
                dx, dy = self._alignment_per_seq[key]
                self._entry_offset_cache[entry_id] = (dx, dy)
                return (dx, dy)

        # 2. Try sampling per-image offsets from ir_dir
        ir_dir = entry.get('ir_dir', '')
        if ir_dir and self._alignment_per_image:
            # Sample a few IR file paths to get the median offset
            try:
                ir_files = self._sorted_image_files(ir_dir)[:20]
            except Exception:
                ir_files = []
            dxs, dys = [], []
            for f in ir_files:
                if f in self._alignment_per_image:
                    ox, oy = self._alignment_per_image[f]
                    dxs.append(ox)
                    dys.append(oy)
            if len(dxs) >= 3:
                dx = float(np.median(dxs))
                dy = float(np.median(dys))

        self._entry_offset_cache[entry_id] = (dx, dy)
        return (dx, dy)

    def _apply_frame_alignment(self, rgb, ir, dx, dy):
        """Apply alignment offset to one frame's IR and crop intersection."""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return rgb, ir

        h, w = rgb.shape[:2]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        ir_2d = ir[:, :, 0] if ir.ndim == 3 else ir
        ir_2d = cv2.warpAffine(ir_2d, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Crop to intersection
        x0 = int(np.ceil(max(0, dx)))
        x1 = w - int(np.ceil(max(0, -dx)))
        y0 = int(np.ceil(max(0, dy)))
        y1 = h - int(np.ceil(max(0, -dy)))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        if x1 - x0 < 64 or y1 - y0 < 64:
            return rgb, ir  # offset too large, skip

        rgb = rgb[y0:y1, x0:x1]
        ir = ir_2d[y0:y1, x0:x1, np.newaxis]
        return rgb, ir

    def _get_paired(self, entry, fl, fps):
        """Get a paired RGB+IR sequence."""
        rgb_files = fl['rgb']
        ir_files = fl['ir']
        frame_values = fl.get('frame_values')
        n = min(len(rgb_files), len(ir_files))
        if n == 0:
            return self._empty_sequence('rgbir_paired')
        indices = self._sample_positions(n, entry, frame_values)
        timestamps = self._build_timestamps(indices, frame_values, fps)

        # Get per-sequence alignment offset (consistent across all frames)
        dx, dy = self._get_entry_offset(entry)

        frames = []
        for i in indices:
            rgb = self._read_rgb(rgb_files[min(i, len(rgb_files) - 1)])
            ir = self._read_ir(ir_files[min(i, len(ir_files) - 1)])
            # Ensure same spatial size before alignment
            if rgb.shape[:2] != ir.shape[:2]:
                ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
                if ir.ndim == 2:
                    ir = ir[:, :, np.newaxis]
            # Apply alignment offset before resize
            rgb, ir = self._apply_frame_alignment(rgb, ir, dx, dy)
            # Resize to target
            rgb = cv2.resize(rgb, (self.target_size, self.target_size))
            ir = cv2.resize(ir, (self.target_size, self.target_size))
            if ir.ndim == 2:
                ir = ir[:, :, np.newaxis]
            frames.append(np.concatenate([rgb, ir], axis=2))  # [H, W, 4]

        frames, aug_mask = self.transform(frames, base_modality_mask=[1.0, 1.0])
        return (torch.stack(frames, dim=0),
                torch.tensor(aug_mask, dtype=torch.float32),
                torch.tensor(timestamps, dtype=torch.float32))

    def _get_single(self, entry, fl, fps, is_rgb=True):
        """Get an RGB-only or IR-only sequence."""
        file_list = fl['frames']
        frame_values = fl.get('frame_values')
        if len(file_list) == 0:
            return self._empty_sequence('rgb_only' if is_rgb else 'ir_only')
        indices = self._sample_positions(len(file_list), entry, frame_values)
        timestamps = self._build_timestamps(indices, frame_values, fps)

        frames = []
        for i in indices:
            path = file_list[min(i, len(file_list) - 1)]
            if is_rgb:
                rgb = self._read_rgb(path)
                rgb = cv2.resize(rgb, (self.target_size, self.target_size))
                ir = np.zeros((self.target_size, self.target_size, 1), dtype=np.uint8)
            else:
                ir = self._read_ir(path)
                ir = cv2.resize(ir, (self.target_size, self.target_size))
                if ir.ndim == 2:
                    ir = ir[:, :, np.newaxis]
                rgb = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            frames.append(np.concatenate([rgb, ir], axis=2))

        mask = [1.0, 0.0] if is_rgb else [0.0, 1.0]
        frames, aug_mask = self.transform(frames, base_modality_mask=mask)
        return (torch.stack(frames, dim=0),
                torch.tensor(aug_mask, dtype=torch.float32),
                torch.tensor(timestamps, dtype=torch.float32))


def build_video_dataset(manifest_path, num_frames=4, target_size=224,
                        alignment_offsets_path=None):
    """
    Build a combined video dataset from a manifest JSON.

    The manifest can contain entries from uav_human, aerial_rgbt,
    and frame-extracted sequence datasets.
    Returns a ConcatDataset if multiple types are present.
    """
    all_entries = _load_video_entries(manifest_path)

    has_uav = any(e.get('dataset') == 'uav_human' or 'rgb_video_path' in e
                  for e in all_entries)
    has_aerial = any(e.get('dataset') == 'aerial_rgbt' for e in all_entries)
    has_frame_seq = any(
        e.get('dataset', '') in FrameSequenceDataset.SUPPORTED_DATASETS
        and e.get('modality') in ('rgbir_paired', 'rgb_only', 'ir_only')
        for e in all_entries)

    datasets = []
    if has_uav:
        datasets.append(UAVHumanVideoDataset(
            manifest_path, num_frames=num_frames, target_size=target_size))
    if has_aerial:
        datasets.append(AerialRGBTSequenceDataset(
            manifest_path, num_frames=num_frames, target_size=target_size))
    if has_frame_seq:
        datasets.append(FrameSequenceDataset(
            manifest_path, num_frames=num_frames, target_size=target_size,
            alignment_offsets_path=alignment_offsets_path))

    if len(datasets) == 0:
        raise ValueError(f"No valid video entries found in {manifest_path}")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def collate_video(batch):
    """
    Collate function for video datasets.

    Input: list of (frames [T,4,H,W], mask [2], timestamps [T])
    Output: (frames [B,T,4,H,W], masks [B,2], timestamps [B,T])
    """
    frames_list, masks_list, timestamps_list = zip(*batch)

    frames = torch.stack(frames_list, dim=0)       # [B, T, 4, H, W]
    masks = torch.stack(masks_list, dim=0)          # [B, 2]
    timestamps = torch.stack(timestamps_list, dim=0)  # [B, T]

    return frames, masks, timestamps

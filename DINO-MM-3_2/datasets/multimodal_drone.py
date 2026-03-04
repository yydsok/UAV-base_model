"""
Multi-modal drone dataset for RGB+IR DINO pretraining.
Supports:
- RGB+IR paired images
- Pure RGB images (IR channel filled with zeros)
- LMDB storage for fast random access
- On-the-fly image file loading (fallback)
"""

import os
import io
import json
import pickle
import random
import logging
import threading
from collections import Counter

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

logger = logging.getLogger(__name__)

# ============================================================================
# Field / type aliases accepted by manifest normalization
# ============================================================================
_RGB_PATH_KEYS = ('rgb_path', 'rgb', 'visible_path', 'visible', 'img_path', 'image_path')
_IR_PATH_KEYS = ('ir_path', 'ir', 'infrared_path', 'infrared', 'thermal_path', 'thermal')
_TYPE_KEYS = ('type', 'modality_type', 'sample_type', 'modality')
_TYPE_MAP = {
    'paired': 'paired', 'pair': 'paired', 'rgb_ir': 'paired', 'rgbir': 'paired',
    'rgb_only': 'rgb_only', 'rgb': 'rgb_only', 'visible': 'rgb_only', 'visible_only': 'rgb_only',
    'ir_only': 'ir_only', 'ir': 'ir_only', 'infrared': 'ir_only',
    'thermal_only': 'ir_only', 'thermal': 'ir_only',
}


def _find_key(d, candidates):
    """Return the value for the first matching key."""
    for k in candidates:
        if k in d:
            return d[k]
    return None


def _normalize_sample(raw, manifest_dir):
    """Normalize a raw manifest sample dict into (rgb_path, ir_path, type).

    - Maps alternative field names to canonical ones.
    - Resolves relative paths against *manifest_dir*.
    - Infers type from path presence when not explicitly provided.
    """
    rgb_path = _find_key(raw, _RGB_PATH_KEYS)
    ir_path = _find_key(raw, _IR_PATH_KEYS)

    # Resolve relative paths
    if rgb_path and not os.path.isabs(rgb_path):
        rgb_path = os.path.join(manifest_dir, rgb_path)
    if ir_path and not os.path.isabs(ir_path):
        ir_path = os.path.join(manifest_dir, ir_path)

    # Determine type
    raw_type = _find_key(raw, _TYPE_KEYS)
    if raw_type is not None:
        sample_type = _TYPE_MAP.get(str(raw_type).lower().strip())
        if sample_type is None:
            sample_type = _infer_type(rgb_path, ir_path)
    else:
        sample_type = _infer_type(rgb_path, ir_path)

    return {
        'rgb_path': rgb_path,
        'ir_path': ir_path,
        'type': sample_type,
        'dataset': raw.get('dataset', ''),
    }


def _infer_type(rgb_path, ir_path):
    has_rgb = rgb_path is not None and rgb_path != ''
    has_ir = ir_path is not None and ir_path != ''
    if has_rgb and has_ir:
        return 'paired'
    elif has_rgb:
        return 'rgb_only'
    elif has_ir:
        return 'ir_only'
    return 'rgb_only'  # fallback


def _read_ir(path):
    """Read an IR image with auto bit-depth handling.

    Supports 8-bit and 16-bit IR images. 16-bit images are scaled to 0-255 uint8.
    Returns [H, W, 1] uint8 array.
    """
    ir_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if ir_raw is None:
        return None, 'decode_fail'

    # Handle different bit depths
    if ir_raw.dtype == np.uint16:
        # 16-bit → scale to 8-bit
        mn, mx = ir_raw.min(), ir_raw.max()
        if mx > mn:
            ir_raw = ((ir_raw.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            ir_raw = np.zeros_like(ir_raw, dtype=np.uint8)
        bit_depth = 16
    elif ir_raw.dtype == np.uint8:
        bit_depth = 8
    else:
        # float or other → clip and convert
        ir_raw = np.clip(ir_raw * 255, 0, 255).astype(np.uint8) if ir_raw.max() <= 1.0 \
            else np.clip(ir_raw, 0, 255).astype(np.uint8)
        bit_depth = -1

    # If multi-channel (e.g. 3-channel pseudo-color IR), convert to gray
    if ir_raw.ndim == 3 and ir_raw.shape[2] > 1:
        ir_raw = cv2.cvtColor(ir_raw, cv2.COLOR_BGR2GRAY)

    if ir_raw.ndim == 2:
        ir_raw = ir_raw[:, :, np.newaxis]

    return ir_raw, bit_depth


class _LoadStats:
    """Thread-safe load failure statistics with rate-limited logging."""

    def __init__(self, log_first_n=20):
        self._lock = threading.Lock()
        self.fail_count = 0
        self.total_count = 0
        self.fail_by_reason = Counter()
        self._log_first_n = log_first_n
        self._ir_bit_depths = Counter()

    def record_success(self):
        with self._lock:
            self.total_count += 1

    def record_ir_bit_depth(self, depth):
        with self._lock:
            self._ir_bit_depths[depth] += 1

    def record_failure(self, index, reason, path=''):
        with self._lock:
            self.fail_count += 1
            self.total_count += 1
            self.fail_by_reason[reason] += 1
            if self.fail_count <= self._log_first_n:
                logger.warning(f"Load fail #{self.fail_count} idx={index} "
                               f"reason={reason} path={path}")
            elif self.fail_count == self._log_first_n + 1:
                logger.warning(f"Suppressing further per-sample warnings "
                               f"(already {self.fail_count} failures). "
                               f"Summary will be printed via get_summary().")

    @property
    def fail_ratio(self):
        if self.total_count == 0:
            return 0.0
        return self.fail_count / self.total_count

    def get_summary(self):
        with self._lock:
            lines = [
                f"Load stats: {self.total_count} total, "
                f"{self.fail_count} failures "
                f"({self.fail_ratio:.4%})",
            ]
            if self.fail_by_reason:
                lines.append(f"  Failures by reason: {dict(self.fail_by_reason)}")
            if self._ir_bit_depths:
                lines.append(f"  IR bit depths: {dict(self._ir_bit_depths)}")
            return '\n'.join(lines)


class MultiModalDroneDataset(Dataset):
    """
    Dataset for RGB+IR multi-modal drone imagery.

    Data can be loaded from:
    1. LMDB database (recommended for training performance)
    2. Direct image files (for prototyping / small datasets)

    Each sample returns:
    - image: numpy array [H, W, 4] (RGB 3ch + IR 1ch), uint8
    - metadata: dict with 'dataset_name', 'type' (paired/rgb_only/ir_only)
    - modality_mask: [2] binary array (rgb_available, ir_available)
    """
    def __init__(self, data_source, transform=None, mode='lmdb',
                 strict_loading=False, max_load_fail_ratio=0.01,
                 dry_run_samples=0):
        """
        Args:
            data_source: path to LMDB database or JSON manifest file
            transform: DataAugmentationDINO_RGBIR instance
            mode: 'lmdb' or 'file'
            strict_loading: if True, raise on any load failure (precheck mode)
            max_load_fail_ratio: abort if failure ratio exceeds this (0 = no limit)
            dry_run_samples: if > 0, only use first N samples (for precheck)
        """
        self.transform = transform
        self.mode = mode
        self.strict_loading = strict_loading
        self.max_load_fail_ratio = max_load_fail_ratio
        self.dry_run_samples = dry_run_samples
        self.load_stats = _LoadStats()

        if mode == 'lmdb':
            self._init_lmdb(data_source)
        elif mode == 'file':
            self._init_file(data_source)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_lmdb(self, lmdb_path):
        """Initialize LMDB-based dataset."""
        assert HAS_LMDB, "lmdb package not installed. Install with: pip install lmdb"
        self.lmdb_path = lmdb_path
        self.env = None  # Lazy initialization for multi-worker compatibility

        # Read metadata from a separate JSON
        meta_path = os.path.join(os.path.dirname(lmdb_path), 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            self.num_samples = len(self.metadata['samples'])
        else:
            # Fallback: count LMDB entries
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                self.num_samples = txn.stat()['entries']
            env.close()
            self.metadata = None

    def _init_file(self, manifest_path):
        """Initialize file-based dataset from a JSON manifest.

        Accepts manifests with various field naming conventions.
        Relative paths are resolved against the manifest's parent directory.
        """
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, 'r') as f:
            raw_manifest = json.load(f)

        raw_samples = raw_manifest.get('samples', raw_manifest.get('data', []))

        # Normalize every sample
        self.samples = [_normalize_sample(s, manifest_dir) for s in raw_samples]
        self.num_samples = len(self.samples)

        if self.dry_run_samples > 0:
            self.num_samples = min(self.num_samples, self.dry_run_samples)
            self.samples = self.samples[:self.num_samples]

        # Print data config summary (rank-0 only, checked by caller)
        type_dist = Counter(s['type'] for s in self.samples)
        logger.info(f"Manifest loaded: {manifest_path}")
        logger.info(f"  Samples: {self.num_samples}  "
                     f"Type distribution: {dict(type_dist)}")
        if self.dry_run_samples > 0:
            logger.info(f"  [DRY-RUN] limited to first {self.dry_run_samples} samples")

    def _get_lmdb_env(self):
        """Lazy LMDB initialization (for multi-worker DataLoader)."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False)
        return self.env

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.mode == 'lmdb':
            image, modality_mask = self._load_lmdb(index)
        else:
            image, modality_mask = self._load_file(index)

        # Apply transforms
        if self.transform is not None:
            crops, view_crop = self.transform(image)
        else:
            # Default: just convert to tensor
            from models.transforms_rgbir import ToTensor4Ch
            tensor = ToTensor4Ch()(image)
            crops = [tensor]
            view_crop = None

        # Compute actual modality mask after augmentation (might be modified by SensorDrop)
        # The SensorDrop is applied per crop, so mask is approximate here
        modality_mask = torch.tensor(modality_mask, dtype=torch.float32)

        return crops, view_crop, modality_mask

    def _load_lmdb(self, index):
        """Load a sample from LMDB."""
        env = self._get_lmdb_env()
        with env.begin(write=False) as txn:
            raw = txn.get(str(index).encode())

        if raw is None:
            raise IndexError(f"Sample {index} not found in LMDB")

        data = pickle.loads(raw)

        # Expected format: (rgb_bytes, rgb_shape, ir_bytes, ir_shape, meta_dict)
        if len(data) == 5:
            rgb_bytes, rgb_shape, ir_bytes, ir_shape, meta = data
            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(rgb_shape)
            if ir_bytes is not None:
                ir = np.frombuffer(ir_bytes, dtype=np.uint8).reshape(ir_shape)
                modality_mask = [1.0, 1.0]
            else:
                ir = np.zeros((*rgb_shape[:2], 1), dtype=np.uint8)
                modality_mask = [1.0, 0.0]
        elif len(data) == 3:
            # Simpler format: (rgb_bytes, ir_bytes_or_none, meta)
            rgb_bytes, ir_bytes, meta = data
            rgb = cv2.imdecode(
                np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if ir_bytes is not None:
                ir = cv2.imdecode(
                    np.frombuffer(ir_bytes, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                ir = ir[:, :, np.newaxis]
                modality_mask = [1.0, 1.0]
            else:
                ir = np.zeros((*rgb.shape[:2], 1), dtype=np.uint8)
                modality_mask = [1.0, 0.0]
        else:
            raise ValueError(f"Unknown LMDB data format with {len(data)} elements")

        # Concatenate: [H, W, 4]
        image = np.concatenate([rgb, ir], axis=2)
        return image, modality_mask

    def _load_file(self, index):
        """Load a sample from image files."""
        sample = self.samples[index]

        try:
            if sample['type'] == 'paired':
                rgb = cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR)
                if rgb is None:
                    raise IOError(f"Failed to read RGB: {sample['rgb_path']}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                ir, ir_depth = _read_ir(sample['ir_path'])
                if ir is None:
                    raise IOError(f"Failed to read IR: {sample['ir_path']}")
                self.load_stats.record_ir_bit_depth(ir_depth)

                # Ensure same spatial size
                if rgb.shape[:2] != ir.shape[:2]:
                    ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
                    if ir.ndim == 2:
                        ir = ir[:, :, np.newaxis]

                modality_mask = [1.0, 1.0]

            elif sample['type'] == 'rgb_only':
                rgb = cv2.imread(sample['rgb_path'], cv2.IMREAD_COLOR)
                if rgb is None:
                    raise IOError(f"Failed to read RGB: {sample['rgb_path']}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                ir = np.zeros((*rgb.shape[:2], 1), dtype=np.uint8)
                modality_mask = [1.0, 0.0]

            elif sample['type'] == 'ir_only':
                ir, ir_depth = _read_ir(sample['ir_path'])
                if ir is None:
                    raise IOError(f"Failed to read IR: {sample['ir_path']}")
                self.load_stats.record_ir_bit_depth(ir_depth)
                rgb = np.zeros((*ir.shape[:2], 3), dtype=np.uint8)
                modality_mask = [0.0, 1.0]

            else:
                raise ValueError(f"Unknown sample type: {sample['type']}")

            self.load_stats.record_success()

        except Exception as e:
            reason = type(e).__name__
            path_info = sample.get('rgb_path') or sample.get('ir_path') or ''
            self.load_stats.record_failure(index, reason, path_info)

            if self.strict_loading:
                raise RuntimeError(
                    f"Strict loading: failed on sample {index} "
                    f"({sample}): {e}"
                ) from e

            # Check fail ratio threshold
            if (self.max_load_fail_ratio > 0 and
                    self.load_stats.fail_ratio > self.max_load_fail_ratio and
                    self.load_stats.total_count >= 100):
                raise RuntimeError(
                    f"Load failure ratio {self.load_stats.fail_ratio:.4%} "
                    f"exceeds threshold {self.max_load_fail_ratio:.4%} "
                    f"after {self.load_stats.total_count} samples. "
                    f"Aborting. {self.load_stats.get_summary()}"
                )

            # Fallback: return a small dummy image so training doesn't crash
            rgb = np.zeros((512, 640, 3), dtype=np.uint8)
            ir = np.zeros((512, 640, 1), dtype=np.uint8)
            modality_mask = [0.0, 0.0]

        # Concatenate: [H, W, 4]
        image = np.concatenate([rgb, ir], axis=2)
        return image, modality_mask


class WeightedMultiDatasetSampler(torch.utils.data.Sampler):
    """
    Weighted sampler for multi-dataset training.
    Uses sqrt(dataset_size) weighting to balance large and small datasets.
    """
    def __init__(self, dataset_sizes, total_samples_per_epoch=None):
        """
        Args:
            dataset_sizes: dict {dataset_name: num_samples}
            total_samples_per_epoch: total samples per epoch (default: sum of all)
        """
        self.dataset_sizes = dataset_sizes
        self.total = total_samples_per_epoch or sum(dataset_sizes.values())

        # Compute sqrt-weighted probabilities
        sqrt_sizes = {k: np.sqrt(v) for k, v in dataset_sizes.items()}
        total_sqrt = sum(sqrt_sizes.values())
        self.weights = {k: v / total_sqrt for k, v in sqrt_sizes.items()}

        # Build cumulative ranges for each dataset
        self.ranges = {}
        offset = 0
        for name, size in dataset_sizes.items():
            self.ranges[name] = (offset, offset + size)
            offset += size

    def __iter__(self):
        indices = []
        dataset_names = list(self.weights.keys())
        probs = [self.weights[n] for n in dataset_names]

        for _ in range(self.total):
            # Choose dataset
            ds = random.choices(dataset_names, weights=probs, k=1)[0]
            # Choose sample within dataset
            start, end = self.ranges[ds]
            idx = random.randint(start, end - 1)
            indices.append(idx)

        return iter(indices)

    def __len__(self):
        return self.total


def collate_multimodal(batch):
    """
    Custom collate function for multi-modal crops.

    Each sample is: (crops_list, view_crop, modality_mask)
    where crops_list is a list of tensors of potentially different sizes.

    Returns:
        crops: list of stacked tensors (one per crop type)
        view_crops: stacked tensor or None
        modality_masks: [B, 2] tensor
    """
    crops_list, view_list, mask_list = zip(*batch)

    # Stack crops by position (all global crop 1's together, etc.)
    num_crops = len(crops_list[0])
    crops = [torch.stack([sample[i] for sample in crops_list])
             for i in range(num_crops)]

    # Stack view crops
    if view_list[0] is not None:
        view_crops = torch.stack(view_list)
    else:
        view_crops = None

    # Stack modality masks
    modality_masks = torch.stack(mask_list)

    return crops, view_crops, modality_masks

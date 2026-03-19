"""
Video augmentation pipeline for temporal training.

Reuses transforms from transforms_rgbir for consistency with image pipeline.
All frames in a clip share the same spatial/sensor decisions for temporal consistency.
Per-frame: GaussianBlur (independent).
"""

import random
import numpy as np
import cv2
import torch

from models.transforms_rgbir import (
    ToTensor4Ch, MultiModalNormalize, RandomSensorDrop_RGBIR,
    GaussianBlur, RandomResizedCrop4Ch,
)


class VideoAugmentation:
    """
    Augmentation pipeline for video clips (list of T frames).

    Shared across all frames (temporal consistency):
    - RandomResizedCrop (same crop params)
    - Horizontal/vertical flip
    - SensorDrop (same modality dropped for all frames)

    Per-frame (independent):
    - GaussianBlur (p=0.3)

    Args:
        target_size: output spatial resolution (square)
        crop_scale: scale range for RandomResizedCrop
        sensor_drop_p: probability of dropping a modality
        blur_p: per-frame probability of GaussianBlur
    """

    def __init__(self, target_size=224, crop_scale=(0.6, 1.0),
                 sensor_drop_p=0.5, blur_p=0.3):
        self.target_size = target_size
        self.crop = RandomResizedCrop4Ch(target_size, scale=crop_scale)
        self.to_tensor = ToTensor4Ch()
        self.normalize = MultiModalNormalize()
        self.sensor_drop = RandomSensorDrop_RGBIR(p_drop=sensor_drop_p)
        self.gaussian_blur = GaussianBlur(sigma=(0.1, 2.0))
        self.blur_p = blur_p

    def __call__(self, frames, base_modality_mask=None):
        """
        Args:
            frames: list of T numpy arrays, each [H, W, 4] uint8

        Returns:
            tuple: (frames_list, modality_mask)
                frames_list: list of T tensors, each [4, target_size, target_size]
                modality_mask: [2] list, [rgb_available, ir_available]
        """
        if not frames:
            if base_modality_mask is None:
                return [], [1.0, 1.0]
            base = self.sensor_drop._normalize_mask(base_modality_mask)
            return [], [float(base[0]), float(base[1])]

        if base_modality_mask is None:
            base_mask = self.sensor_drop.infer_raw_modality_mask(frames[0])
        else:
            rgb_avail, ir_avail = self.sensor_drop._normalize_mask(base_modality_mask)
            base_mask = [float(rgb_avail), float(ir_avail)]
        output_mask = self.sensor_drop.choose_output_mask(base_mask)

        # --- Sample shared decisions once for all frames ---
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5

        # Shared crop params (computed from first frame dimensions)
        h, w = frames[0].shape[:2]
        crop_params = self.crop.get_params(h, w)  # (i, j, new_h, new_w)

        result = []
        for frame in frames:
            # Shared RandomResizedCrop (same region for all frames)
            i, j, new_h, new_w = crop_params
            cropped = frame[i:i+new_h, j:j+new_w]
            frame = cv2.resize(cropped, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)

            # Shared spatial flips
            if do_hflip:
                frame = np.flip(frame, axis=1).copy()
            if do_vflip:
                frame = np.flip(frame, axis=0).copy()

            # Per-frame GaussianBlur
            if random.random() < self.blur_p:
                frame = self.gaussian_blur(frame)

            # To tensor and normalize
            tensor = self.to_tensor(frame)
            tensor = self.normalize(tensor)
            tensor = self.sensor_drop.apply_mask(tensor, output_mask)

            result.append(tensor)
        return result, output_mask

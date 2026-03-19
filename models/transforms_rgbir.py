"""
Data transforms for RGB+IR multi-modal DINO pretraining.

Includes:
- RandomSensorDrop_RGBIR: Random modality dropout
- AffineViewAugmentation: Perspective transforms for viewpoint invariance (RingMo-Aerial ATCL)
- MultiModalNormalize: Per-modality normalization
- DataAugmentationDINO_RGBIR: Full multi-crop augmentation pipeline
"""

import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps


# ============================================================================
# Multi-modal specific transforms
# ============================================================================

class RandomSensorDrop_RGBIR:
    """
    Randomly drop one modality's channels to enable single- and multi-modal learning.
    Input: numpy array [H, W, 4] or torch tensor [4, H, W]
    Channels [0:3] = RGB, Channel [3] = IR

    Probability:
    - 50%: keep all channels (multi-modal learning)
    - 25%: zero out IR -> RGB-only
    - 25%: zero out RGB -> IR-only
    """
    def __init__(self, p_drop=0.5, p_rgb_only=0.5):
        self.p_drop = p_drop
        self.p_rgb_only = p_rgb_only

    @staticmethod
    def _normalize_mask(mask):
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().tolist()
        if len(mask) != 2:
            raise ValueError(f"Expected mask of length 2, got: {mask}")
        return [float(mask[0]) > 0.5, float(mask[1]) > 0.5]

    def infer_raw_modality_mask(self, sample):
        """Infer modality availability from an unnormalized sample."""
        is_tensor = isinstance(sample, torch.Tensor)
        if is_tensor:
            rgb_active = sample[:3].abs().sum() > 0
            ir_active = sample[3:].abs().sum() > 0
        else:
            rgb_active = np.abs(sample[:, :, :3]).sum() > 0
            ir_active = np.abs(sample[:, :, 3:]).sum() > 0
        return [float(rgb_active), float(ir_active)]

    def choose_output_mask(self, base_mask):
        """Choose the output modality mask after sensor drop.

        If the source sample only has one modality, sensor drop becomes a no-op
        so we never fabricate a fully empty sample.
        """
        rgb_avail, ir_avail = self._normalize_mask(base_mask)
        if not (rgb_avail and ir_avail):
            return [float(rgb_avail), float(ir_avail)]

        if random.random() > self.p_drop:
            return [1.0, 1.0]
        if random.random() < self.p_rgb_only:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def apply_mask(self, sample, modality_mask):
        rgb_avail, ir_avail = self._normalize_mask(modality_mask)
        is_tensor = isinstance(sample, torch.Tensor)
        if not rgb_avail:
            if is_tensor:
                sample[:3, :, :] = 0
            else:
                sample[:, :, :3] = 0
        if not ir_avail:
            if is_tensor:
                sample[3:, :, :] = 0
            else:
                sample[:, :, 3:] = 0
        return sample

    def __call__(self, sample, base_mask=None):
        if base_mask is None:
            base_mask = self.infer_raw_modality_mask(sample)
        return self.apply_mask(sample, self.choose_output_mask(base_mask))

    def get_modality_mask(self, sample):
        return self.infer_raw_modality_mask(sample)


class AffineViewAugmentation:
    """
    Affine/perspective transformation for viewpoint invariance.
    Simulates different UAV viewing angles (nadir, oblique, side-view).

    Reference: RingMo-Aerial CLAF (Contrastive Learning Affine Framework)

    The transformation coefficient alpha controls the degree of perspective change:
    - alpha close to 0: minimal perspective distortion
    - alpha close to 0.5: strong perspective distortion
    """
    def __init__(self, alpha_range=(0.05, 0.35), p=0.5):
        self.alpha_range = alpha_range
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: PIL Image or numpy array [H, W, C]

        Returns:
            Perspective-transformed image
        """
        if random.random() > self.p:
            return img

        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img_np = np.array(img)
        else:
            img_np = img.copy()

        h, w = img_np.shape[:2]
        alpha = random.uniform(*self.alpha_range)

        # Source points (corners)
        src_pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])

        # Destination points with perspective distortion
        direction = random.choice(['top', 'bottom', 'left', 'right', 'mixed'])

        if direction == 'top':
            dst_pts = np.float32([
                [w * alpha, h * alpha],
                [w * (1 - alpha), h * alpha],
                [w, h], [0, h]
            ])
        elif direction == 'bottom':
            dst_pts = np.float32([
                [0, 0], [w, 0],
                [w * (1 - alpha), h * (1 - alpha)],
                [w * alpha, h * (1 - alpha)]
            ])
        elif direction == 'left':
            dst_pts = np.float32([
                [w * alpha, h * alpha],
                [w, 0],
                [w, h],
                [w * alpha, h * (1 - alpha)]
            ])
        elif direction == 'right':
            dst_pts = np.float32([
                [0, 0],
                [w * (1 - alpha), h * alpha],
                [w * (1 - alpha), h * (1 - alpha)],
                [0, h]
            ])
        else:  # mixed
            offset = w * alpha
            dst_pts = np.float32([
                [random.uniform(0, offset), random.uniform(0, offset)],
                [w - random.uniform(0, offset), random.uniform(0, offset)],
                [w - random.uniform(0, offset), h - random.uniform(0, offset)],
                [random.uniform(0, offset), h - random.uniform(0, offset)]
            ])

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img_np, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101)

        if is_pil:
            return Image.fromarray(warped)
        return warped


class MultiModalNormalize:
    """
    Per-modality normalization.
    RGB channels: ImageNet statistics
    IR channel: configurable statistics
    """
    def __init__(self, rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225),
                 ir_mean=(0.5,), ir_std=(0.5,)):
        self.rgb_mean = torch.tensor(rgb_mean).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std).view(3, 1, 1)
        self.ir_mean = torch.tensor(ir_mean).view(1, 1, 1)
        self.ir_std = torch.tensor(ir_std).view(1, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor: [4, H, W] float tensor in range [0, 1]
        """
        # Normalize RGB channels
        tensor[:3] = (tensor[:3] - self.rgb_mean.to(tensor.device)) / self.rgb_std.to(tensor.device)
        # Normalize IR channel
        tensor[3:] = (tensor[3:] - self.ir_mean.to(tensor.device)) / self.ir_std.to(tensor.device)
        return tensor


class GaussianBlur:
    """Gaussian blur augmentation (for multi-channel images)."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        if isinstance(img, Image.Image):
            sigma = random.uniform(*self.sigma)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif isinstance(img, np.ndarray):
            sigma = random.uniform(*self.sigma)
            ksize = int(np.ceil(sigma * 3) * 2 + 1)
            return cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img


class Solarize:
    """Solarize augmentation."""
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return ImageOps.solarize(img, self.threshold)
        elif isinstance(img, np.ndarray):
            result = img.copy()
            result[result >= self.threshold] = 255 - result[result >= self.threshold]
            return result
        return img


class RandomBrightness:
    """Random brightness adjustment for multi-channel images."""
    def __init__(self, max_delta=0.4):
        self.max_delta = max_delta

    def __call__(self, img):
        factor = 1.0 + random.uniform(-self.max_delta, self.max_delta)
        if isinstance(img, np.ndarray):
            return np.clip(img * factor, 0, 255).astype(img.dtype)
        return img


class RandomContrast:
    """Random contrast adjustment for multi-channel images."""
    def __init__(self, max_delta=0.4):
        self.max_delta = max_delta

    def __call__(self, img):
        factor = 1.0 + random.uniform(-self.max_delta, self.max_delta)
        if isinstance(img, np.ndarray):
            mean = img.mean()
            return np.clip((img - mean) * factor + mean, 0, 255).astype(img.dtype)
        return img


class ToTensor4Ch:
    """Convert 4-channel numpy [H,W,4] uint8 to torch [4,H,W] float32 in [0,1]."""
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # HWC -> CHW
            tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
            tensor = tensor / 255.0
            return tensor
        return img


class RandomResizedCrop4Ch:
    """Random resized crop for 4-channel numpy images."""
    def __init__(self, size, scale=(0.4, 1.0), ratio=(3./4., 4./3.)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, h, w):
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < new_w <= w and 0 < new_h <= h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)
                return i, j, new_h, new_w

        # Fallback to center crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.ratio):
            new_w = w
            new_h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            new_h = h
            new_w = int(round(h * max(self.ratio)))
        else:
            new_w = w
            new_h = h
        i = (h - new_h) // 2
        j = (w - new_w) // 2
        return i, j, new_h, new_w

    def __call__(self, img):
        """
        Args:
            img: numpy array [H, W, 4]
        Returns:
            numpy array [self.size[0], self.size[1], 4]
        """
        h, w = img.shape[:2]
        i, j, new_h, new_w = self.get_params(h, w)
        crop = img[i:i+new_h, j:j+new_w]
        resized = cv2.resize(crop, self.size, interpolation=cv2.INTER_LINEAR)
        return resized


class RandomHorizontalFlip4Ch:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, axis=1).copy()
        return img


class RandomVerticalFlip4Ch:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, axis=0).copy()
        return img


class RandomRotation90:
    """Random 90-degree rotation (useful for remote sensing / aerial images)."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            return np.rot90(img, k, axes=(0, 1)).copy()
        return img


class Compose4Ch:
    """Compose transforms for 4-channel images."""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


# ============================================================================
# Main data augmentation pipeline
# ============================================================================

class DataAugmentationDINO_RGBIR:
    """
    Full DINO-style multi-crop augmentation for RGB+IR (4-channel) images.

    Outputs:
    - 2 global crops at global_crop_size (e.g., 224)
    - local_crops_number local crops at local_crop_size (e.g., 96)
    - Each crop independently gets RandomSensorDrop

    Also produces:
    - Affine view augmented version (for viewpoint invariance loss L_view)
    """
    def __init__(self, global_crop_size=224, local_crop_size=96,
                 global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 local_crops_number=8, use_view_augmentation=True,
                 return_pair_anchor=False):

        self.local_crops_number = local_crops_number
        self.use_view_augmentation = use_view_augmentation
        self.return_pair_anchor = return_pair_anchor

        # Spatial augmentations (applied before ToTensor)
        flip_and_jitter = Compose4Ch([
            RandomHorizontalFlip4Ch(p=0.5),
            RandomVerticalFlip4Ch(p=0.5),
            RandomRotation90(p=0.5),
        ])

        # Normalize + SensorDrop (applied after ToTensor)
        to_tensor = ToTensor4Ch()
        normalize = MultiModalNormalize()
        self.sensor_drop = RandomSensorDrop_RGBIR(p_drop=0.5)

        # Keep references for post-processing helpers
        self.to_tensor = to_tensor
        self.normalize = normalize

        # Global crop 1: always Gaussian blur
        self.global_transfo1 = Compose4Ch([
            RandomResizedCrop4Ch(global_crop_size, scale=global_crops_scale),
            flip_and_jitter,
            GaussianBlur(sigma=(0.1, 2.0)),
        ])
        self.global_post1 = [to_tensor, normalize]

        # Global crop 2: less Gaussian blur, sometimes Solarize
        self.global_transfo2 = Compose4Ch([
            RandomResizedCrop4Ch(global_crop_size, scale=global_crops_scale),
            flip_and_jitter,
        ])
        self.global_post2 = [to_tensor, normalize]
        self.blur2_p = 0.1
        self.solarize2_p = 0.2

        # Local crops
        self.local_transfo = Compose4Ch([
            RandomResizedCrop4Ch(local_crop_size, scale=local_crops_scale),
            flip_and_jitter,
        ])
        self.local_post = [to_tensor, normalize]
        self.blur_local_p = 0.5

        # Viewpoint augmentation (ATCL from RingMo-Aerial)
        self.view_aug = AffineViewAugmentation(alpha_range=(0.05, 0.35), p=1.0)

    def _apply_post(self, img, post_transforms, blur_p=0.0, solarize_p=0.0):
        """Apply blur/solarize probabilistically, then post-transforms."""
        if blur_p > 0 and random.random() < blur_p:
            img = GaussianBlur(sigma=(0.1, 2.0))(img)
        if solarize_p > 0 and random.random() < solarize_p:
            img = Solarize(128)(img)
        for t in post_transforms:
            img = t(img)
        return img

    def _resolve_base_mask(self, image, base_modality_mask=None):
        if base_modality_mask is not None:
            rgb_avail, ir_avail = self.sensor_drop._normalize_mask(base_modality_mask)
            return [float(rgb_avail), float(ir_avail)]
        return self.sensor_drop.infer_raw_modality_mask(image)

    def _finalize_crop(self, img, post_transforms, base_mask,
                       blur_p=0.0, solarize_p=0.0, preserve_full_modal=False):
        img = self._apply_post(img, post_transforms, blur_p=blur_p, solarize_p=solarize_p)
        full_modal = img.clone() if preserve_full_modal else None
        final_mask = self.sensor_drop.choose_output_mask(base_mask)
        img = self.sensor_drop.apply_mask(img, final_mask)
        if preserve_full_modal:
            return img, torch.tensor(final_mask, dtype=torch.float32), full_modal
        return img, torch.tensor(final_mask, dtype=torch.float32)

    def __call__(self, image, base_modality_mask=None):
        """
        Args:
            image: numpy array [H, W, 4] (RGB 3ch + IR 1ch), uint8

        Returns:
            crops: list of tensors [4, crop_H, crop_W]
            view_crop: tensor [4, H, W] affine-augmented version (for L_view)
                       None if use_view_augmentation=False
            pair_anchor: tensor [4, H, W] full-modality global crop before
                         sensor drop, or None when disabled
        """
        crops = []
        crop_masks = []
        base_mask = self._resolve_base_mask(image, base_modality_mask)
        pair_anchor = None

        # Global crop 1
        g1 = self.global_transfo1(image)
        if self.return_pair_anchor:
            g1, g1_mask, pair_anchor = self._finalize_crop(
                g1, self.global_post1, base_mask, blur_p=1.0,
                preserve_full_modal=True)
        else:
            g1, g1_mask = self._finalize_crop(
                g1, self.global_post1, base_mask, blur_p=1.0)
        crops.append(g1)
        crop_masks.append(g1_mask)

        # Global crop 2
        g2 = self.global_transfo2(image)
        g2, g2_mask = self._finalize_crop(
            g2, self.global_post2, base_mask,
            blur_p=self.blur2_p, solarize_p=self.solarize2_p)
        crops.append(g2)
        crop_masks.append(g2_mask)

        # Local crops
        for _ in range(self.local_crops_number):
            lc = self.local_transfo(image)
            lc, lc_mask = self._finalize_crop(
                lc, self.local_post, base_mask, blur_p=self.blur_local_p)
            crops.append(lc)
            crop_masks.append(lc_mask)

        # Viewpoint-augmented version for L_view
        view_crop = None
        view_mask = None
        if self.use_view_augmentation:
            view_img = self.view_aug(image)
            # Apply same global crop pipeline to the view-augmented image
            view_crop = Compose4Ch([
                RandomResizedCrop4Ch(crops[0].shape[-1],
                                     scale=(0.4, 1.0)),
                RandomHorizontalFlip4Ch(p=0.5),
            ])(view_img)
            for t in [ToTensor4Ch(), MultiModalNormalize()]:
                view_crop = t(view_crop)
            view_mask_values = self.sensor_drop.choose_output_mask(base_mask)
            view_crop = self.sensor_drop.apply_mask(view_crop, view_mask_values)
            view_mask = torch.tensor(view_mask_values, dtype=torch.float32)

        return crops, view_crop, crop_masks, view_mask, pair_anchor

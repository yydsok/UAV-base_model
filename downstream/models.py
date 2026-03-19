import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import RGBIR_MEAN, RGBIR_STD, build_backbone_from_checkpoint, freeze_backbone
from downstream.strong_heads import DinoMMUPerNetSegmentor


class DinoMMDenseBackbone(nn.Module):
    def __init__(self, backbone, feature_dim=256, out_indices=None):
        super().__init__()
        self.backbone = backbone
        self.patch_size = getattr(backbone.patch_embed, "patch_size", 16)
        self.feature_dim = feature_dim
        self.out_indices = tuple(out_indices or self._default_out_indices())
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(backbone.embed_dim, feature_dim, kernel_size=1) for _ in self.out_indices]
        )
        self.out_channels = feature_dim

    def _default_out_indices(self):
        depth = len(self.backbone.blocks)
        candidates = [depth // 4 - 1, depth // 2 - 1, (3 * depth) // 4 - 1, depth - 1]
        unique = []
        for idx in candidates:
            idx = max(0, min(depth - 1, idx))
            if idx not in unique:
                unique.append(idx)
        while len(unique) < 4:
            unique.append(depth - 1)
        return unique[:4]

    def _prepare_tokens(self, x):
        if hasattr(self.backbone, "prepare_tokens"):
            return self.backbone.prepare_tokens(x)
        raise AttributeError("Backbone does not implement prepare_tokens.")

    def _tokens_to_map(self, tokens, height, width):
        batch_size, _, channels = tokens.shape
        patch_h = max(1, height // self.patch_size)
        patch_w = max(1, width // self.patch_size)
        patch_tokens = tokens[:, 1:, :]
        return patch_tokens.transpose(1, 2).reshape(batch_size, channels, patch_h, patch_w)

    def forward_feature_list(self, x):
        height, width = x.shape[-2:]
        tokens = self._prepare_tokens(x)
        captured = []
        for idx, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            if idx in self.out_indices:
                captured.append(self.backbone.norm(tokens))

        if len(captured) != len(self.out_indices):
            raise RuntimeError("Failed to capture the requested transformer blocks.")

        maps = []
        for token_map, proj in zip(captured, self.proj_layers):
            fmap = self._tokens_to_map(token_map, height, width)
            maps.append(proj(fmap))

        p2 = F.interpolate(maps[0], scale_factor=2.0, mode="bilinear", align_corners=False)
        p3 = maps[1]
        p4 = F.avg_pool2d(maps[2], kernel_size=2, stride=2)
        p5 = F.avg_pool2d(maps[3], kernel_size=4, stride=4)
        return [p2, p3, p4, p5]

    def forward(self, x):
        features = self.forward_feature_list(x)
        return OrderedDict((str(idx), feat) for idx, feat in enumerate(features))


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        outputs = [x]
        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            outputs.append(pooled)
        return torch.cat(outputs, dim=1)


class LiteUPerHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=8, dropout=0.1):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels, in_channels // 2)
        ppm_out = in_channels + 4 * (in_channels // 2)
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
            nn.ReLU(inplace=True),
        )
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, features):
        feats = list(features)
        feats[-1] = self.ppm_bottleneck(self.ppm(feats[-1]))

        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, feats[:-1])]
        laterals.append(feats[-1])

        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fpn_outs = []
        for feat, conv in zip(laterals, self.fpn_convs):
            fpn_outs.append(conv(feat))

        target_size = fpn_outs[0].shape[-2:]
        fused = []
        for feat in fpn_outs:
            fused.append(F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False))
        fused = torch.cat(fused, dim=1)
        return self.fuse(fused)


class DinoMMSegmentor(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.decode_head = LiteUPerHead(in_channels=backbone.out_channels, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone.forward_feature_list(x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ZeroBasedLabelDetectorWrapper(nn.Module):
    """Adapts torchvision dense detectors that expect 0-based labels."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def _shift_targets(self, targets):
        shifted = []
        for target in targets:
            adapted = dict(target)
            adapted["labels"] = target["labels"].clone() - 1
            shifted.append(adapted)
        return shifted

    def _shift_outputs(self, outputs):
        shifted = []
        for output in outputs:
            adapted = dict(output)
            adapted["labels"] = output["labels"] + 1
            shifted.append(adapted)
        return shifted

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("Targets are required during training.")
            return self.detector(images, self._shift_targets(targets))
        outputs = self.detector(images)
        return self._shift_outputs(outputs)


class DinoMMClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x, return_all_tokens=False)
        return self.head(self.dropout(features))


class DinoMMRetrievalModel(nn.Module):
    def __init__(self, backbone, projection_dim=256, dropout=0.1, temperature=0.07):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(backbone.embed_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def encode(self, x):
        features = self.backbone(x, return_all_tokens=False)
        features = self.projection(self.dropout(features))
        return F.normalize(features, dim=1)

    def forward(self, x):
        return self.encode(x)


def _build_dense_detection_backbone(
    checkpoint_path,
    checkpoint_key,
    device,
    init_mode,
    arch,
    patch_size,
    in_chans,
    fusion,
    trainable_blocks,
    feature_dim,
):
    backbone, temporal_attn, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)
    dense_backbone = DinoMMDenseBackbone(backbone, feature_dim=feature_dim)
    dense_backbone.to(device)
    return dense_backbone, meta


def _build_torchvision_fasterrcnn(dense_backbone, num_classes, min_size, max_size):
    anchor_sizes = ((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256))
    aspect_ratios = ((0.5, 1.0, 2.0),) * 4
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        dense_backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=max_size,
        image_mean=RGBIR_MEAN,
        image_std=RGBIR_STD,
    )


def _build_torchvision_fcos(dense_backbone, num_classes, min_size, max_size, score_thresh):
    anchor_sizes = ((8,), (16,), (32,), (64,))
    aspect_ratios = ((1.0,),) * 4
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    detector = FCOS(
        dense_backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        image_mean=RGBIR_MEAN,
        image_std=RGBIR_STD,
        anchor_generator=anchor_generator,
        score_thresh=score_thresh,
    )
    return ZeroBasedLabelDetectorWrapper(detector)


def build_detection_model(
    checkpoint_path,
    num_classes,
    checkpoint_key="teacher",
    device="cpu",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    feature_dim=256,
    min_size=640,
    max_size=1333,
    score_thresh=0.05,
    framework="fasterrcnn",
):
    dense_backbone, meta = _build_dense_detection_backbone(
        checkpoint_path=checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        init_mode=init_mode,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        trainable_blocks=trainable_blocks,
        feature_dim=feature_dim,
    )

    if framework == "fasterrcnn":
        model = _build_torchvision_fasterrcnn(dense_backbone, num_classes, min_size=min_size, max_size=max_size)
    elif framework == "fcos":
        model = _build_torchvision_fcos(
            dense_backbone,
            num_classes,
            min_size=min_size,
            max_size=max_size,
            score_thresh=score_thresh,
        )
    elif framework == "cascade_rcnn":
        from downstream.openmmlab import build_mmdet_cascade_rcnn

        model = build_mmdet_cascade_rcnn(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            checkpoint_key=checkpoint_key,
            init_mode=init_mode,
            arch=arch,
            patch_size=patch_size,
            in_chans=in_chans,
            fusion=fusion,
            trainable_blocks=trainable_blocks,
            feature_dim=feature_dim,
            score_thresh=score_thresh,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported detection framework '{framework}'.")
    model.to(device)
    meta["framework"] = framework
    return model, meta


def build_segmentation_model(
    checkpoint_path,
    num_classes,
    checkpoint_key="teacher",
    device="cpu",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    feature_dim=256,
    framework="upernet",
):
    backbone, temporal_attn, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)
    dense_backbone = DinoMMDenseBackbone(backbone, feature_dim=feature_dim)
    dense_backbone.to(device)
    if framework == "lite_uper":
        model = DinoMMSegmentor(dense_backbone, num_classes=num_classes)
    elif framework == "upernet":
        model = DinoMMUPerNetSegmentor(
            dense_backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
    else:
        raise ValueError(f"Unsupported segmentation framework '{framework}'.")
    model.to(device)
    meta["framework"] = framework
    return model, meta


def build_classification_model(
    checkpoint_path,
    num_classes,
    checkpoint_key="teacher",
    device="cpu",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=12,
):
    backbone, temporal_attn, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)
    model = DinoMMClassifier(backbone, num_classes=num_classes)
    model.to(device)
    return model, meta


def build_retrieval_model(
    checkpoint_path,
    checkpoint_key="teacher",
    device="cpu",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=12,
    projection_dim=256,
    dropout=0.1,
    temperature=0.07,
):
    backbone, temporal_attn, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)
    model = DinoMMRetrievalModel(
        backbone,
        projection_dim=projection_dim,
        dropout=dropout,
        temperature=temperature,
    )
    model.to(device)
    meta["projection_dim"] = projection_dim
    return model, meta

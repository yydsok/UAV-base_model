import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import RGBIR_MEAN, RGBIR_STD, build_backbone_from_checkpoint, freeze_backbone


def _ensure_mplconfigdir():
    mpl_dir = os.environ.get("MPLCONFIGDIR")
    if not mpl_dir:
        mpl_dir = "/tmp/mpl-skysence"
        os.environ["MPLCONFIGDIR"] = mpl_dir
    Path(mpl_dir).mkdir(parents=True, exist_ok=True)
    return mpl_dir


def require_mmdet_runtime(require_mmtrack=False):
    _ensure_mplconfigdir()
    try:
        import mmcv

        if getattr(mmcv, "__version__", "").startswith("2.2."):
            mmcv.__version__ = "2.1.0"
        import mmdet  # noqa: F401
        if require_mmtrack:
            import mmtrack  # noqa: F401
    except AssertionError as exc:
        raise RuntimeError(
            "OpenMMLab detection requires an mmdet/mmcv runtime compatible with mmcv<2.2.0."
        ) from exc
    except ModuleNotFoundError as exc:
        missing = str(exc)
        if "mmcv._ext" in missing:
            raise RuntimeError(
                "OpenMMLab detection requires compiled mmcv ops. The current environment only has mmcv-lite."
            ) from exc
        if require_mmtrack:
            raise RuntimeError("MMTracking requires mmtrack, mmdet, and compiled mmcv ops.") from exc
        raise RuntimeError("MMDetection requires mmdet and compiled mmcv ops.") from exc


def require_mmrotate_runtime():
    require_mmdet_runtime()
    try:
        import mmrotate  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("MMRotate requires mmrotate, mmdet, and compiled mmcv ops.") from exc


def _register_mmdet_backbone():
    require_mmdet_runtime()
    from mmdet.models.builder import BACKBONES

    if "DinoMMMMDetBackbone" in BACKBONES.module_dict:
        return BACKBONES.module_dict["DinoMMMMDetBackbone"]

    class DinoMMMMDetBackbone(nn.Module):
        def __init__(
            self,
            checkpoint_path=None,
            checkpoint_key="teacher",
            init_mode="pretrained",
            arch=None,
            patch_size=None,
            in_chans=None,
            fusion=None,
            trainable_blocks=4,
            feature_dim=256,
        ):
            super().__init__()
            from downstream.models import DinoMMDenseBackbone

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
            self.backbone = DinoMMDenseBackbone(backbone, feature_dim=feature_dim)
            self.out_channels = [feature_dim] * 4
            self.meta = meta

        def init_weights(self, pretrained=None):
            return None

        def forward(self, x):
            return tuple(self.backbone.forward_feature_list(x))

    BACKBONES.register_module()(DinoMMMMDetBackbone)
    return DinoMMMMDetBackbone


def _register_mmrotate_backbone():
    require_mmrotate_runtime()
    from mmrotate.models.builder import ROTATED_BACKBONES

    if "DinoMMMMRotateBackbone" in ROTATED_BACKBONES.module_dict:
        return ROTATED_BACKBONES.module_dict["DinoMMMMRotateBackbone"]

    class DinoMMMMRotateBackbone(nn.Module):
        def __init__(
            self,
            checkpoint_path=None,
            checkpoint_key="teacher",
            init_mode="pretrained",
            arch=None,
            patch_size=None,
            in_chans=None,
            fusion=None,
            trainable_blocks=4,
            feature_dim=256,
        ):
            super().__init__()
            from downstream.models import DinoMMDenseBackbone

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
            self.backbone = DinoMMDenseBackbone(backbone, feature_dim=feature_dim)
            self.out_channels = [feature_dim] * 4
            self.meta = meta

        def init_weights(self, pretrained=None):
            return None

        def forward(self, x):
            return tuple(self.backbone.forward_feature_list(x))

    ROTATED_BACKBONES.register_module()(DinoMMMMRotateBackbone)
    return DinoMMMMRotateBackbone


def _build_cascade_rcnn_cfg(
    checkpoint_path,
    num_classes,
    checkpoint_key,
    init_mode,
    arch,
    patch_size,
    in_chans,
    fusion,
    trainable_blocks,
    feature_dim,
    score_thresh,
):
    return dict(
        type="CascadeRCNN",
        backbone=dict(
            type="DinoMMMMDetBackbone",
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            init_mode=init_mode,
            arch=arch,
            patch_size=patch_size,
            in_chans=in_chans,
            fusion=fusion,
            trainable_blocks=trainable_blocks,
            feature_dim=feature_dim,
        ),
        neck=dict(
            type="FPN",
            in_channels=[feature_dim] * 4,
            out_channels=feature_dim,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=feature_dim,
            feat_channels=feature_dim,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        ),
        roi_head=dict(
            type="CascadeRoIHead",
            num_stages=3,
            stage_loss_weights=[1.0, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=feature_dim,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=[
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=feature_dim,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=feature_dim,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.05, 0.05, 0.1, 0.1],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=feature_dim,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.033, 0.033, 0.067, 0.067],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
            ],
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=[
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    pos_weight=-1,
                    debug=False,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    pos_weight=-1,
                    debug=False,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    pos_weight=-1,
                    debug=False,
                ),
            ],
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=score_thresh,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=100,
            ),
        ),
    )


def build_mmdet_cascade_rcnn(
    checkpoint_path,
    num_classes,
    checkpoint_key="teacher",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    feature_dim=256,
    score_thresh=0.05,
    device="cpu",
):
    require_mmdet_runtime()
    _register_mmdet_backbone()
    from mmcv import ConfigDict
    from mmdet.models import build_detector

    cfg = _build_cascade_rcnn_cfg(
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
    )
    cfg = ConfigDict(cfg)
    model = build_detector(cfg)
    model.init_weights()
    model.to(device)
    return model


def _normalize_image(image):
    mean = image.new_tensor(RGBIR_MEAN).view(-1, 1, 1)
    std = image.new_tensor(RGBIR_STD).view(-1, 1, 1)
    return (image - mean) / std


def pack_mmdet_images(images, device, size_divisor=32):
    normalized = [_normalize_image(image.to(device)) for image in images]
    max_h = max(image.shape[-2] for image in normalized)
    max_w = max(image.shape[-1] for image in normalized)
    pad_h = int(math.ceil(max_h / size_divisor) * size_divisor)
    pad_w = int(math.ceil(max_w / size_divisor) * size_divisor)
    batch = normalized[0].new_zeros((len(normalized), normalized[0].shape[0], pad_h, pad_w))
    img_metas = []
    for idx, image in enumerate(normalized):
        height, width = image.shape[-2:]
        batch[idx, :, :height, :width] = image
        img_metas.append(
            dict(
                img_shape=(height, width, image.shape[0]),
                ori_shape=(height, width, image.shape[0]),
                pad_shape=(pad_h, pad_w, image.shape[0]),
                batch_input_shape=(pad_h, pad_w),
                scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                flip=False,
                img_norm_cfg=dict(
                    mean=np.array(RGBIR_MEAN, dtype=np.float32),
                    std=np.array(RGBIR_STD, dtype=np.float32),
                    to_rgb=False,
                ),
            )
        )
    return batch, img_metas


def pack_mmdet_targets(targets, device):
    gt_bboxes = []
    gt_labels = []
    for target in targets:
        gt_bboxes.append(target["boxes"].to(device))
        gt_labels.append((target["labels"] - 1).to(device))
    return gt_bboxes, gt_labels


def train_mmdet_detector(model, images, targets):
    device = next(model.parameters()).device
    batch, img_metas = pack_mmdet_images(images, device=device)
    gt_bboxes, gt_labels = pack_mmdet_targets(targets, device=device)
    losses = model(
        img=batch,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True,
    )
    loss, log_vars = model._parse_losses(losses)
    return loss, log_vars


def _result_to_prediction(result, image_id):
    if isinstance(result, tuple):
        result = result[0]

    boxes = []
    scores = []
    labels = []
    for class_idx, class_result in enumerate(result):
        if class_result is None or len(class_result) == 0:
            continue
        class_tensor = torch.as_tensor(class_result, dtype=torch.float32)
        boxes.append(class_tensor[:, :4])
        scores.append(class_tensor[:, 4])
        labels.append(torch.full((class_tensor.shape[0],), class_idx + 1, dtype=torch.int64))

    if boxes:
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes.cpu(),
        "scores": scores.cpu(),
        "labels": labels.cpu(),
        "image_id": image_id.cpu() if isinstance(image_id, torch.Tensor) else torch.tensor([int(image_id)], dtype=torch.int64),
    }


@torch.no_grad()
def infer_mmdet_detector(model, images, image_ids=None):
    device = next(model.parameters()).device
    batch, img_metas = pack_mmdet_images(images, device=device)
    results = model(
        img=[batch],
        img_metas=[img_metas],
        return_loss=False,
        rescale=True,
    )

    if image_ids is None:
        image_ids = [torch.tensor([idx], dtype=torch.int64) for idx in range(len(results))]
    return [_result_to_prediction(result, image_id) for result, image_id in zip(results, image_ids)]


def _build_oriented_rcnn_cfg(
    checkpoint_path,
    num_classes,
    checkpoint_key,
    init_mode,
    arch,
    patch_size,
    in_chans,
    fusion,
    trainable_blocks,
    feature_dim,
    score_thresh,
    angle_version,
):
    return dict(
        type="OrientedRCNN",
        backbone=dict(
            type="DinoMMMMRotateBackbone",
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            init_mode=init_mode,
            arch=arch,
            patch_size=patch_size,
            in_chans=in_chans,
            fusion=fusion,
            trainable_blocks=trainable_blocks,
            feature_dim=feature_dim,
        ),
        neck=dict(
            type="FPN",
            in_channels=[feature_dim] * 4,
            out_channels=feature_dim,
            num_outs=5,
        ),
        rpn_head=dict(
            type="OrientedRPNHead",
            in_channels=feature_dim,
            feat_channels=feature_dim,
            version=angle_version,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="MidpointOffsetCoder",
                angle_range=angle_version,
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        ),
        roi_head=dict(
            type="OrientedStandardRoIHead",
            bbox_roi_extractor=dict(
                type="RotatedSingleRoIExtractor",
                roi_layer=dict(type="RoIAlignRotated", out_size=7, sample_num=2, clockwise=True),
                out_channels=feature_dim,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=dict(
                type="RotatedShared2FCBBoxHead",
                in_channels=feature_dim,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHAOBBoxCoder",
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=0,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.8),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    iou_calculator=dict(type="RBboxOverlaps2D"),
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RRandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.8),
                min_bbox_size=0,
            ),
            rcnn=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=score_thresh,
                nms=dict(iou_thr=0.1),
                max_per_img=2000,
            ),
        ),
    )


def build_mmrotate_oriented_rcnn(
    checkpoint_path,
    num_classes,
    checkpoint_key="teacher",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    feature_dim=256,
    score_thresh=0.05,
    angle_version="le90",
    device="cpu",
):
    require_mmrotate_runtime()
    _register_mmrotate_backbone()
    from mmcv import ConfigDict
    from mmrotate.models import build_detector

    cfg = _build_oriented_rcnn_cfg(
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
        angle_version=angle_version,
    )
    cfg = ConfigDict(cfg)
    model = build_detector(cfg)
    model.init_weights()
    model.to(device)
    return model


def pack_mmrotate_targets(targets, device):
    gt_bboxes = []
    gt_labels = []
    for target in targets:
        if "rbboxes" in target:
            rbboxes = target["rbboxes"].to(device)
        else:
            boxes = target["boxes"].to(device)
            if boxes.numel() == 0:
                rbboxes = boxes.new_zeros((0, 5))
            else:
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
                wh = (boxes[:, 2:4] - boxes[:, :2]).clamp(min=1e-3)
                angles = boxes.new_zeros((boxes.shape[0], 1))
                rbboxes = torch.cat([centers, wh, angles], dim=1)
        gt_bboxes.append(rbboxes)
        gt_labels.append((target["labels"] - 1).to(device))
    return gt_bboxes, gt_labels


def train_mmrotate_detector(model, images, targets):
    device = next(model.parameters()).device
    batch, img_metas = pack_mmdet_images(images, device=device)
    gt_bboxes, gt_labels = pack_mmrotate_targets(targets, device=device)
    losses = model(
        img=batch,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True,
    )
    loss, log_vars = model._parse_losses(losses)
    return loss, log_vars


def _oriented_to_hbb(obb):
    if obb.numel() == 0:
        return obb.new_zeros((0, 4))
    angle = obb[:, 4]
    if torch.max(torch.abs(angle)).item() > math.pi + 1e-3:
        angle = torch.deg2rad(angle)
    cos = torch.abs(torch.cos(angle))
    sin = torch.abs(torch.sin(angle))
    half_w = 0.5 * (obb[:, 2] * cos + obb[:, 3] * sin)
    half_h = 0.5 * (obb[:, 2] * sin + obb[:, 3] * cos)
    x1 = obb[:, 0] - half_w
    y1 = obb[:, 1] - half_h
    x2 = obb[:, 0] + half_w
    y2 = obb[:, 1] + half_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _oriented_result_to_prediction(result, image_id):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, np.ndarray):
        result = [result]

    boxes = []
    rbboxes = []
    scores = []
    labels = []
    for class_idx, class_result in enumerate(result):
        if class_result is None or len(class_result) == 0:
            continue
        class_tensor = torch.as_tensor(class_result, dtype=torch.float32)
        if class_tensor.ndim == 1:
            class_tensor = class_tensor.unsqueeze(0)
        if class_tensor.shape[1] >= 6:
            class_rbboxes = class_tensor[:, :5]
            class_scores = class_tensor[:, 5]
        elif class_tensor.shape[1] == 5:
            class_rbboxes = class_tensor[:, :5]
            class_scores = torch.ones((class_tensor.shape[0],), dtype=torch.float32)
        else:
            class_boxes = class_tensor[:, :4]
            centers = (class_boxes[:, :2] + class_boxes[:, 2:4]) / 2.0
            wh = (class_boxes[:, 2:4] - class_boxes[:, :2]).clamp(min=1e-3)
            angles = torch.zeros((class_boxes.shape[0], 1), dtype=torch.float32)
            class_rbboxes = torch.cat([centers, wh, angles], dim=1)
            class_scores = (
                class_tensor[:, 4]
                if class_tensor.shape[1] > 4
                else torch.ones((class_tensor.shape[0],), dtype=torch.float32)
            )
        rbboxes.append(class_rbboxes)
        boxes.append(_oriented_to_hbb(class_rbboxes))
        scores.append(class_scores)
        labels.append(torch.full((class_rbboxes.shape[0],), class_idx + 1, dtype=torch.int64))

    if boxes:
        boxes = torch.cat(boxes, dim=0)
        rbboxes = torch.cat(rbboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        rbboxes = torch.zeros((0, 5), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes.cpu(),
        "rbboxes": rbboxes.cpu(),
        "scores": scores.cpu(),
        "labels": labels.cpu(),
        "image_id": image_id.cpu() if isinstance(image_id, torch.Tensor) else torch.tensor([int(image_id)], dtype=torch.int64),
    }


@torch.no_grad()
def infer_mmrotate_detector(model, images, image_ids=None):
    device = next(model.parameters()).device
    batch, img_metas = pack_mmdet_images(images, device=device)
    results = model(
        img=[batch],
        img_metas=[img_metas],
        return_loss=False,
        rescale=True,
    )
    if image_ids is None:
        image_ids = [torch.tensor([idx], dtype=torch.int64) for idx in range(len(results))]
    return [_oriented_result_to_prediction(result, image_id) for result, image_id in zip(results, image_ids)]


class MMByteTrackAdapter:
    def __init__(
        self,
        modality="both",
        high_threshold=0.5,
        low_threshold=0.1,
        match_iou_threshold=0.3,
        init_track_threshold=0.7,
        tentative_frames=3,
        device="cpu",
    ):
        require_mmdet_runtime(require_mmtrack=True)
        from mmtrack.models.motion import KalmanFilter
        from mmtrack.models.trackers import ByteTracker

        self.modality = modality
        self.device = torch.device(device)
        self.tracker = ByteTracker(
            obj_score_thrs={"high": high_threshold, "low": low_threshold},
            init_track_thr=init_track_threshold,
            match_iou_thrs={
                "high": match_iou_threshold,
                "low": match_iou_threshold,
                "tentative": match_iou_threshold,
            },
            num_tentatives=tentative_frames,
        )
        self.model = SimpleNamespace(motion=KalmanFilter(center_only=False))
        self.frame_idx = 0

    def reset(self):
        self.tracker.reset()
        self.frame_idx = 0

    def update_frame(self, frame, detections):
        from downstream.datasets import _merge_rgb_ir

        image = _merge_rgb_ir(frame["rgb_path"], frame["ir_path"], modality=self.modality)
        batch, img_metas = pack_mmdet_images([image], device=self.device)

        if detections:
            boxes = torch.tensor(
                [det["box"] + [det["score"]] for det in detections],
                dtype=torch.float32,
                device=self.device,
            )
            labels = torch.tensor(
                [max(0, int(det["label"]) - 1) for det in detections],
                dtype=torch.long,
                device=self.device,
            )
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32, device=self.device)
            labels = torch.zeros((0,), dtype=torch.long, device=self.device)

        tracked_boxes, tracked_labels, track_ids = self.tracker.track(
            batch,
            img_metas,
            self.model,
            boxes,
            labels,
            frame_id=self.frame_idx,
            rescale=False,
        )
        self.frame_idx += 1

        outputs = []
        for bbox, label, track_id in zip(tracked_boxes, tracked_labels, track_ids):
            outputs.append(
                {
                    "track_id": int(track_id.item()),
                    "box": bbox[:4].detach().cpu().tolist(),
                    "label": int(label.item()) + 1,
                    "score": float(bbox[4].item()),
                }
            )
        return outputs

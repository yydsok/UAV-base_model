# DINO-MM 下游任务

这个目录提供了面向 DINO-MM checkpoint 的统一下游微调与评测入口，目标是同时满足：

- 公平迁移主表（固定头，仅比较初始化）；
- 外部 benchmark（采用更强任务框架）；
- 向 RingMo-Aerial / SkySense 系列任务叙事对齐。

## 当前状态（2026-03-15）

本轮已完成从“轻量可用”到“强框架可落地”的关键升级：

- 分割：默认使用仓库内 `UPerNet` 强头（保留 `Lite-UPer` 旧基线）。
- 检测：`fasterrcnn`、`fcos`、`cascade_rcnn` 三种框架已打通，其中 `cascade_rcnn` 已接入 MMDetection。
- 旋转检测：`Oriented R-CNN (MMRotate)` 已接入，支持 DINO-MM backbone 训练/推理。
- 跟踪：`iou`、`bytetrack` 本地实现可用，新增 `mm_bytetrack`（MMTracking ByteTrack 适配）。
- 检索：在原有评测基础上新增 `train_retrieval.py`，支持跨模态/跨视角检索对比学习微调。
- 场景分类：`AID` / `RESISC-45` 线性探针已接入 `train_classification.py`。
- 变化检测：新增 `train_change_detection.py` / `eval_change_detection.py`，使用 DINO-MM + BIT-style 头。
- OpenMMLab：`skysence` 独立环境可导入并运行 `mmdet/mmtrack/mmseg/mmrotate`。

## 任务矩阵（对齐 RingMo/SkySense + 我们扩展）

### 已可直接运行

- 语义分割：`UAVid` / `UDD5` / `UDD6` / `AeroScapes`
  - 推荐框架：`UPerNet`（`--framework upernet`）
- 目标检测：`DroneVehicle` / `LLVIP` / `RGBT-Tiny` / `VT-Tiny-MOT` / `HIT-UAV` / `M3OT`
  - 推荐框架：`Cascade R-CNN`（`--framework cascade_rcnn`）
- 旋转目标检测：`DIOR-R` / `FAIR1M` / `custom`
  - 推荐框架：`Oriented R-CNN`（`train_oriented_detection.py`，基于 MMRotate）
- 多目标跟踪：`M3OT` / `VT-Tiny-MOT`
  - 推荐关联器：`MMTracking ByteTrack`（`--tracker mm_bytetrack`）
- 跨模态/跨视角检索：`DroneVehicle` / `LLVIP` / `SUES-200` / `CVOGL`
  - 训练：`train_retrieval.py`
  - 评测：`eval_retrieval.py`
- 场景分类：`AID` / `RESISC-45` / 通用 `imagefolder`
  - 训练：`train_classification.py`
- 变化检测：`LEVIR-CD` / `CDD` / `OSCD`（目录结构兼容 BIT_CD 风格）
  - 训练：`train_change_detection.py`
  - 评测：`eval_change_detection.py`

### 论文覆盖但当前仓库仍待补齐

- 多模态分割 / 多模态场景分类（当前本地缺少可用 RGB+IR 标注，暂不纳入）；
- Few-shot 任务（待定）。

### 已新增

- 三维重建：`LuoJia-MVS` / `WHU`（DINO-MM + Cas-MVSNet）
  - 训练：`train_3d_reconstruction.py`
  - 模型/数据集：`mvs_reconstruction.py`

## 框架说明

### 分割

- `--framework upernet`：默认强头，仓库内实现，无需编译版 `mmcv`。
- `--framework lite_uper`：旧轻量基线，仅用于对照。

### 检测

- `--framework fasterrcnn`：稳定主表基线。
- `--framework fcos`：纯 `torchvision` 的强 fallback。
- `--framework cascade_rcnn`：调用 MMDetection 2.x 的 Cascade R-CNN，已实现 DINO-MM backbone 适配。

### 旋转检测

- `train_oriented_detection.py`：调用 MMRotate 0.3.4 的 Oriented R-CNN，已实现 DINO-MM backbone 适配。
- 支持 `--checkpoint/--checkpoint_key/--init_mode`，可在训练入口显式加载预训练模型。

### 跟踪

- `--tracker iou`：轻量基线。
- `--tracker bytetrack`：仓库内 ByteTrack 风格两阶段关联。
- `--tracker mm_bytetrack`：调用 MMTracking 官方 ByteTracker 逻辑（更接近论文对比口径）。

### 检索

- `train_retrieval.py`：双塔对比学习（InfoNCE 对称损失）+ 线性投影头。
- `eval_retrieval.py`：输出 `rank_1/rank_5/rank_10/mAP/MRR`。

### 场景分类

- `train_classification.py --dataset aid`：AID 线性探针（默认 TR=20%，支持改 `--train_ratio`）。
- `train_classification.py --dataset resisc45`：RESISC-45 线性探针（默认 TR=10%）。

### 变化检测

- `train_change_detection.py`：DINO-MM + BIT-style 双时相变化检测训练。
- `eval_change_detection.py`：输出 `F1/IoU/Precision/Recall/OA`。

## OpenMMLab 独立环境

建议强框架全部放在独立环境执行：

```bash
conda activate skysence
export MPLCONFIGDIR=/tmp/mpl-skysence
mkdir -p "$MPLCONFIGDIR"
```

已验证版本组合：

- `python 3.10.13`
- `torch 1.13.1+cu117`
- `mmcv-full 1.7.1`
- `mmdet 2.28.2`
- `mmtrack 0.14.0`
- `mmseg 0.30.0`
- `mmrotate 0.3.4`

## 示例命令

### 分割（UPerNet）

```bash
python downstream/train_segmentation.py \
  --dataset uavid \
  --data_root /root/autodl-tmp/data/UAVid \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/uavid_seg \
  --framework upernet \
  --epochs 30 \
  --batch_size 4 \
  --image_size 512 \
  --amp
```

### 检测（Cascade R-CNN / MMDetection）

```bash
python downstream/train_detection.py \
  --dataset dronevehicle \
  --data_root /root/autodl-tmp/data/DroneVehicle \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/dronevehicle_det_cascade \
  --framework cascade_rcnn \
  --epochs 12 \
  --batch_size 2 \
  --modality both
```

### 旋转检测（Oriented R-CNN / MMRotate）

```bash
python downstream/train_oriented_detection.py \
  --dataset dior_r \
  --data_root /root/autodl-tmp/data/DIOR-R \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/dior_r_oriented \
  --angle_version le90 \
  --epochs 12 \
  --batch_size 2 \
  --modality rgb_only
```

### 跟踪（MMTracking ByteTrack）

```bash
python downstream/eval_tracking.py \
  --dataset m3ot \
  --data_root /root/autodl-tmp/data/M3OT \
  --detector_checkpoint /autodl-fs/data/DINO-MM/downstream_runs/m3ot_det/checkpoint_best.pth \
  --tracker mm_bytetrack
```

### 检索训练（跨模态/跨视角）

```bash
python downstream/train_retrieval.py \
  --dataset sues_200 \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/sues200_ret \
  --epochs 20 \
  --batch_size 64 \
  --trainable_blocks 12 \
  --projection_dim 256 \
  --amp
```

### 检索评测

```bash
python downstream/eval_retrieval.py \
  --dataset sues_200 \
  --checkpoint /autodl-fs/data/DINO-MM/downstream_runs/sues200_ret/checkpoint_best.pth \
  --query_view drone \
  --gallery_view satellite \
  --bidirectional
```

### 场景分类（AID 线性探针）

```bash
python downstream/train_classification.py \
  --dataset aid \
  --data_root /root/autodl-tmp/data/AID \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/aid_cls \
  --epochs 200 \
  --batch_size 64 \
  --train_ratio 0.2 \
  --amp
```

### 变化检测（LEVIR-CD，BIT-style）

```bash
python downstream/train_change_detection.py \
  --dataset levir_cd \
  --data_root /root/autodl-tmp/data/LEVIR-CD \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/levir_cd \
  --epochs 200 \
  --batch_size 8 \
  --image_size 256 \
  --amp
```

### 三维重建（LuoJia-MVS / WHU，Cas-MVSNet）

```bash
python downstream/train_3d_reconstruction.py \
  --data_root /root/autodl-tmp/data/WHU_MVS_dataset \
  --checkpoint /autodl-fs/data/DINO-MM/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth \
  --output_dir /autodl-fs/data/DINO-MM/downstream_runs/whu_mvs \
  --epochs 16 \
  --lr 5e-4 --weight_decay 0.01 \
  --num_views 5 --ndepths 48,32,8 \
  --amp --grad_clip 1.0 --warmup_epochs 2
```

## 下一步（你后面要做的 baseline 对齐）

为对齐 RingMo/SkySense 论文中每个任务的数据集，下一阶段建议并行推进：

1. 拉取并整理各任务官方/主流 baseline repo（检测、跟踪、变化检测、定向检测）。
2. 统一数据转换脚本和评测口径（同一指标实现、同一切分）。
3. 逐任务接入 DINO-MM 预训练权重，先跑单数据集再扩展多数据集主表。
4. 将公平主表与外部 benchmark 分离汇报，避免结论混淆。

RingMo-Aerial 任务优先的模型清单与仓库对齐见：
`downstream/RingMo-Aerial_任务与基线对齐清单.md`。

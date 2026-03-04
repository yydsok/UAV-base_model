# DINO-MM RGB-IR 项目统一说明（代码对齐版）

> 本文档为 RGB-IR 预训练主线说明文档。内容以当前代码实现为准，并整合历史文档中的有效信息。

## 1. 项目定位

DINO-MM 是面向无人机遥感场景的 RGB+IR 多模态自监督预训练框架。核心目标：

- 在无标注条件下学习稳定的多模态表征；
- 同时支持双模态、仅 RGB、仅 IR 样本；
- 通过多损失联合优化增强跨模态一致性与视角不变性。

当前主流程以以下代码为准：

- 训练入口：`main_dino_rgbir.py`
- 损失定义：`dino_loss_rgbir.py`
- 主干模型：`models/vision_transformer_rgbir.py`
- 数据增强：`models/transforms_rgbir.py`
- 数据集与拼接：`datasets/multimodal_drone.py`

---

## 2. 代码结构总览

- `main_dino_rgbir.py`：参数解析、模型构建、训练循环、EMA teacher 更新、日志与 checkpoint。
- `models/vision_transformer_rgbir.py`：ViT 主干与 4 通道输入改造，含 `DualModalPatchEmbed`。
- `models/dino_head.py`：`DINOHead` 与 `MultiCropWrapper`。
- `models/multi_granularity.py`：`SinkhornKnopp` 与 `MultiGranularityFeatures`。
- `models/transforms_rgbir.py`：multi-crop、模态随机丢弃、视角增强。
- `datasets/multimodal_drone.py`：`MultiModalDroneDataset` 与 `collate_multimodal`。
- `dino_loss_rgbir.py`：DINO 与多模态扩展损失（含 TCLLoss）。
- `models/temporal_module.py`：`TemporalAttention` 时序注意力模块（视频帧跨帧融合）。
- `scripts/run_pretrain_multi.sh`：当前推荐训练参数模板。

---

## 3. 数据与输入约定

### 3.1 输入张量

- 输入通道：`in_chans=4`
- 通道语义：前 3 通道为 RGB，第 4 通道为 IR
- 常见形状：`[B, 4, H, W]`

### 3.2 数据读取模式

`MultiModalDroneDataset(data_source, transform, mode, strict_loading, max_load_fail_ratio, dry_run_samples)` 支持：

1. `mode='file'`：JSON manifest（支持多种字段命名与相对路径）；
2. `mode='lmdb'`：LMDB 高吞吐读取。

`file` 模式中 `type` 支持：

- `paired`：RGB+IR；
- `rgb_only`：仅 RGB（IR 补零）；
- `ir_only`：仅 IR（RGB 补零）。

可配置参数：

- `strict_loading`：预检模式，任何加载失败立即报错；
- `max_load_fail_ratio`：失败率阈值（超过则中止训练，默认 0.01）；
- `dry_run_samples`：仅加载前 N 样本用于预检。

### 3.3 batch 输出

`collate_multimodal` 输出关键字段：

- `crops`：多 crop 张量列表；
- `view_crops`：视角增强分支或 `None`；
- `modality_masks`：`[B, 2]`，表示 `[rgb_available, ir_available]`。

---

## 4. 模型架构（当前实现）

### 4.1 ViT 与 4 通道 patch embedding

在 `models/vision_transformer_rgbir.py` 中：

- `in_chans == 4` 时使用 `DualModalPatchEmbed`；
- 支持 `fusion` 策略（当前脚本采用 `concat`）；
- 提供 `vit_tiny/small/base/large/huge/giant` 工厂函数。

### 4.2 DINO 头与多分辨率前向

在 `models/dino_head.py` 中：

- `DINOHead`：MLP + L2 normalize + weight norm 最后一层；
- `MultiCropWrapper`：按分辨率分组前向，减少重复计算；
- 可返回 global crop 对应的 patch tokens 供多粒度与对齐损失使用。

### 4.3 多粒度特征

在 `models/multi_granularity.py` 中：

- `SinkhornKnopp`：对 patch tokens 做软聚类；
- `MultiGranularityFeatures`：提取并投影三层特征：
  - token-level
  - object-level（聚类中心）
  - image-level（全局平均）

---

## 5. 增强策略与模态缺失鲁棒性

`DataAugmentationDINO_RGBIR` 包含：

- DINO multi-crop（2 global + 多个 local）；
- 翻转、旋转、裁剪、模糊、Solarize（按概率）；
- `RandomSensorDrop_RGBIR`：随机保留双模态 / 仅 RGB / 仅 IR；
- `AffineViewAugmentation`：构造视角变化样本。

`RandomSensorDrop_RGBIR` 当前逻辑：

- 50%：保留 RGB+IR
- 25%：仅 RGB
- 25%：仅 IR

---

## 6. 损失函数体系（以代码为准）

全部定义在 `dino_loss_rgbir.py`。

### 6.1 基础损失

- `DINOLoss`：student-teacher 自蒸馏损失。

### 6.2 扩展损失

- `MGCLLoss`：多粒度对比（token/object/image）；
- `CrossModalAlignLoss`：RGB-IR 跨模态对齐；
- `TCLLoss`：时序对比学习（InfoNCE），多帧场景下同序列帧对为正样本，不同序列为负样本；
- `ViewInvarianceLoss`：原图与视角增强图一致性；
- `ViewBridgeLoss`：跨视角原型分布桥接。

`ViewBridgeLoss` 构造签名：

- `temperature=1.0`
- `lambda_sharp=0.0`
- `lambda_balance=0.02`

### 6.3 总损失封装

`PretrainingLoss` 以加权形式汇总：

- `w_mgcl=1.0`
- `w_align=1.0`
- `w_view=0.5`
- `w_bridge=0.5`
- `w_latent=1.0`
- `w_rec=1.0`
- `w_tcl=0.1`（仅 `--use_temporal` 时生效）

---

## 7. 训练参数（当前脚本基线）

参考 `scripts/run_pretrain_multi.sh`：

### 7.1 模型与优化

- `ARCH=vit_small`
- `PATCH_SIZE=16`
- `IN_CHANS=4`
- `FUSION=concat`
- `OUT_DIM=65536`
- `EPOCHS=100`
- `BATCH_SIZE=64`（每卡）
- `LR=0.0005`
- `WARMUP_EPOCHS=10`
- `MIN_LR=1e-6`
- `WEIGHT_DECAY=0.04`
- `WEIGHT_DECAY_END=0.4`
- `CLIP_GRAD=3.0`
- `MOMENTUM_TEACHER=0.996`

### 7.2 温度与 multi-crop

- `WARMUP_TEACHER_TEMP=0.04`
- `TEACHER_TEMP=0.04`
- `GLOBAL_CROP_SIZE=224`
- `LOCAL_CROP_SIZE=96`
- `LOCAL_CROPS_NUMBER=6`

### 7.3 多粒度与 bridge

- `NUM_CLUSTERS=8`
- `PROJ_DIM=256`
- `BRIDGE_PROJ_DIM=256`
- `BRIDGE_NUM_PROTOTYPES=64`
- `BRIDGE_TEMP=1.0`
- `BRIDGE_LAMBDA_SHARP=0.0`
- `BRIDGE_LAMBDA_BALANCE=0.02`

> 脚本已注明：该 bridge 参数为修正后的稳定配置。

### 7.4 时序对比学习（TCL）

- `--use_temporal`：启用时序分支（默认 `False`，不影响静态图片训练）
- `--num_frames=1`：每序列帧数（视频数据就绪后设为 ≥2）
- `--temporal_layers=2`：TemporalAttention 层数
- `--w_tcl=0.1`：TCL 损失权重
- `--tcl_temperature=0.07`：InfoNCE 温度参数

> 当前阶段 TCL 默认关闭。即使启用 `--use_temporal`，因无视频数据加载器（`video_frames=None`），TCL 分支不执行、loss 恒为 0。

### 7.5 并行与精度

- `USE_FP16=True`
- 自动检测 GPU 数量，分别走单卡或 `torchrun` 多卡流程。

---

## 8. 预训练权重与恢复

训练脚本支持：

- `--pretrained`：加载预训练权重（用于 RGB 分支初始化）；
- `--resume`：从 checkpoint 继续训练。

若预训练权重路径不存在，脚本会提示并回退为从头训练。

---

## 9. 训练执行与产物

### 9.1 执行命令

```bash
# 推荐：先预检再训练
bash scripts/run_pretrain_multi.sh precheck   # 验证数据可读
bash scripts/run_pretrain_multi.sh train       # 正式训练
```

### 9.2 输出目录

当前脚本默认：`/root/autodl-tmp/train1/checkpoints_v3`

典型产物：

- `precheck_log.txt`（预检日志）
- `train_log.txt`（训练日志）
- 周期性 checkpoint（按 `save_every`）

---

## 10. 工程注意事项

1. 说明文档与历史参数若和代码冲突，以代码为准。
2. 异常样本在非严格模式下会 fallback 到全零图以防训练中断，但失败率超阈值时自动中止。建议正式训练前先用 `precheck` 模式验证数据。
3. `ViewBridgeLoss` 对温度和正则敏感，建议围绕当前脚本小范围搜索。
4. RGB-IR 配准质量会直接影响跨模态对齐收益。
5. IR 读取支持 8-bit/16-bit 自动识别，位深分布通过 `load_stats` 统计输出。

---

## 11. 最小复现清单

1. 准备 manifest（推荐使用路径已修复的 `merged_train_all_v3.json`）；
2. 校验 `PRETRAINED`、`OUTPUT_DIR`、GPU 环境；
3. 执行 `bash scripts/run_pretrain_multi.sh precheck` 验证数据可读性；
4. 执行 `bash scripts/run_pretrain_multi.sh train` 开始训练；
5. 在日志中确认 `dino/mgcl/align/view/bridge` 等分项稳定。

---

## 12. 维护说明

- 本文件为 RGB-IR 预训练主线说明文档（仓库内可能仍存在其他历史说明文件）；
- 后续代码更新时，请优先同步本文件，避免多文档漂移。

---

## 13. 全量数据集纳入预训练（v2 — 排除 WebUAV-3M）

### 13.1 背景与目标

原合并 manifest (`merged_train_all.json`) 仅覆盖 20 个数据集、约 47 万样本，且只使用了 train split。v2 版本的目标：

- **纳入全部可用数据集**（排除 WebUAV-3M）；
- **覆盖所有 split**（train / val / test 均作为预训练数据，预训练不区分 split）；
- **新增 15 个数据集** 的 manifest；
- **替换** 旧 LLVIP registered 版本为 unregistered 版本；
- **新增 BIRDSAI + LSOTB-TIR** 两个大规模红外数据集，使 RGB:IR 比例趋于 1:1。

### 13.2 新增脚本

#### `scripts/generate_new_manifests.py`

为 15 个缺少 manifest 的数据集生成 JSON manifest，同时将 LLVIP unregistered 的伪 RGB 红外图转换为单通道灰度图。

用法：

```bash
python scripts/generate_new_manifests.py \
    --data_root /root/autodl-tmp/data \
    --out_dir /root/autodl-tmp/train1/manifests \
    --ir_out_root /root/autodl-tmp/train1/converted_ir
```

#### `scripts/merge_manifests_v2.py`

将所有 manifest（旧 + 新，含 train/val/test/all split）合并为单一预训练 manifest。

合并逻辑：

1. 加载 `manifests/` 目录下所有 `.json` 文件；
2. **跳过**：`merged_*.json`（旧合并文件）、`UTUAV_*.json`（当前主训练流程未纳入）；
3. **替换**：旧 `LLVIP_train.json` / `LLVIP_test.json` → 新 `LLVIP_unreg_all.json`；
4. 确保每条样本都有 `type` 字段（`paired` / `rgb_only` / `ir_only`）；
5. 输出统计信息。

用法：

```bash
python scripts/merge_manifests_v2.py \
    --manifest_dir /root/autodl-tmp/train1/manifests \
    --output /root/autodl-tmp/train1/manifests/merged_train_all_v2.json
```

### 13.3 新增 15 个数据集详情

| # | 数据集 | 类型 | 样本数 | 说明 |
|---|--------|------|--------|------|
| 1 | LLVIP unregistered | paired | 13,975 | 替换旧 registered 版本；IR 为伪 RGB(1280×720)，已转 `convert('L')` 保存至 `converted_ir/LLVIP_unreg/` |
| 2 | BIRDSAI | ir_only | 61,994 | 热红外鸟类追踪，640×512，TrainReal + TestReal |
| 3 | LSOTB-TIR | ir_only | 606,829 | 大规模热红外单目标追踪，Training Dataset + Evaluation Dataset |
| 4 | SeaDronesSee MOT | rgb_only | 54,105 | 多目标追踪，使用 Compressed 版本（JPG，9.4GB vs 未压缩 118GB；resize 到 224 后质量差异可忽略） |
| 5 | CVOGL drone | rgb_only | 5,279 | 仅 DroneAerial/query，**排除** satellite/ 和 SVI |
| 6 | University-1652 drone | rgb_only | 127,076 | train/drone + test/gallery_drone + query_drone + 4K_drone，**排除** satellite/google/street |
| 7 | SUES-200 drone | rgb_only | 40,000 | drone_view_512 全部尺度，**排除** satellite-view/ |
| 8 | UAV-Human | rgb_only | 22,476 | 仅 PoseEstimation/frames/ |
| 9 | UAVScenes | rgb_only | 24,126 | interval=5，仅 interval5_CAM/（相机图像），排除 LIDAR |
| 10 | MDMT | rgb_only | 11,764 | train + val + test 全部 |
| 11 | Manipal-UAV | rgb_only | 4,349 | test + validation 的 images/，排除 labels/ |
| 12 | DTB70 | rgb_only | 15,777 | 70 个追踪序列的 img/ 目录 |
| 13 | UAVid | rgb_only | 420 | 仅 Images/，**排除** Labels/ |
| 14 | UDD | rgb_only | 589 | UDD5 + UDD6 的 src/ + m1-only-src/，**排除** gt/ |
| 15 | Stanford Drone | rgb_only | 60 | 仅各场景 reference.jpg 参考图 |

### 13.4 LLVIP unregistered 红外转换

LLVIP unregistered 的红外图像虽以 `.jpg` 存储，但实际为三通道伪 RGB（三个通道相同），分辨率 1280×720。

处理方式：

- 使用 `PIL.Image.convert('L')` 转为单通道灰度；
- 保存至 `/root/autodl-tmp/train1/converted_ir/LLVIP_unreg/{train,test}/`；
- manifest 中 `ir_path` 指向转换后路径，`ir_converted: false`（真实红外，仅格式转换）。

共转换 13,975 张（train 7,606 + test 6,369）。

### 13.5 合并结果统计

#### v2 合并 manifest（不含 UTUAV）

| 类型 | 样本数 |
|------|--------|
| 配对（RGB+IR） | 149,112 |
| 仅 RGB | 646,392 |
| 仅 IR | 724,498 |
| **合计** | **1,520,002** |

共加载 **53 个** manifest，覆盖 **34 个** 数据集（含新增 15 个）。

#### 各数据集样本数排名

| 数据集 | 样本数 |
|--------|--------|
| LSOTB-TIR | 606,829 |
| University-1652-drone | 127,076 |
| UAV123 | 113,476 |
| BIRDSAI | 61,994 |
| SeaDronesSee-MOT | 54,105 |
| MONET | 52,777 |
| VT-Tiny-MOT | 46,941 |
| RGBT-Tiny | 46,701 |
| UAVDT | 40,735 |
| SUES-200-drone | 40,000 |
| AnimalDrone | 35,294 |
| DroneCrowd | 33,960 |
| VisDrone-MOT | 33,682 |
| AU-AIR | 32,823 |
| SeaDronesSee | 28,454 |
| DroneVehicle | 28,439 |
| UAVScenes | 24,126 |
| UAV-Human | 22,476 |
| DTB70 | 15,777 |
| LLVIP-unreg | 13,975 |
| MDMT | 11,764 |
| EVD4UAV | 10,049 |
| VisDrone-DET | 8,629 |
| M3OT | 5,395 |
| CVOGL-drone | 5,279 |
| Manipal-UAV | 4,349 |
| TarDAL-M3FD | 4,200 |
| DroneRGBT | 3,424 |
| AeroScapes | 3,269 |
| HIT-UAV | 2,898 |
| UDD | 589 |
| UAVid | 420 |
| StanfordDrone | 60 |
| TarDAL-tno | 37 |

#### UTUAV 规模备注（当前主训练流程未接入）

| 部分 | 样本数 |
|------|--------|
| v2 合并 manifest（当前训练实际使用） | 1,520,002 |
| UTUAV（独立 manifest，当前未在 `main_dino_rgbir.py` 接入） | 1,664,549 |

> 说明：上述 UTUAV 数量仅作数据规模参考，不代表当前训练会自动纳入。

### 13.6 训练脚本更新

`scripts/run_pretrain_multi.sh` 中 `DATA_PATH` 历史变更：

```bash
# v1（旧）
DATA_PATH="/root/autodl-tmp/train1/manifest_merged.json"
# v2
DATA_PATH="/root/autodl-tmp/train1/manifests/merged_train_all_v2.json"
# v3（当前推荐，路径已修复）
DATA_PATH="/root/autodl-tmp/train1/manifests/merged_train_all_v3.json"
```

UTUAV 因体量过大（166 万样本）当前未纳入主训练流程，也不在合并 manifest 中。

### 13.7 排除策略

| 排除对象 | 原因 |
|----------|------|
| WebUAV-3M | 无 manifest，数据量极大且质量参差，暂不纳入 |
| 卫星遥感图像 | CVOGL satellite/、University-1652 satellite/google/street、SUES-200 satellite-view/ |
| 标注/掩码文件 | UAVid Labels/、UDD gt/、Manipal-UAV labels/ |
| LIDAR 数据 | UAVScenes interval5_LIDAR/ |

### 13.8 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/generate_new_manifests.py` | 新建 | 为 15 个数据集生成 manifest + LLVIP IR 转换 |
| `scripts/merge_manifests_v2.py` | 新建 | 合并所有 manifest 为 v2 |
| `scripts/fix_manifest_paths.py` | 新建（v3） | 修复 v2 manifest 中 414,985 条断裂路径 |
| `scripts/run_pretrain_multi.sh` | 修改 | DATA_PATH 指向 v3 manifest，支持 precheck/train 双模式 |
| `datasets/multimodal_drone.py` | 修改（v3） | 字段规范化、相对路径、IR 位深兼容、加载统计 |
| `main_dino_rgbir.py` | 修改（v3） | 新增 strict_loading / max_load_fail_ratio / dry_run_samples |
| `/root/autodl-tmp/train1/manifests/*.json` | 新增 15 个 | 各数据集的 manifest 文件 |
| `/root/autodl-tmp/train1/manifests/merged_train_all_v2.json` | 新增 | 合并后的总 manifest（290 MB） |
| `/root/autodl-tmp/train1/manifests/merged_train_all_v3.json` | 新增（v3） | 路径已修复的总 manifest |
| `/root/autodl-tmp/train1/converted_ir/LLVIP_unreg/` | 新增 | LLVIP unregistered 灰度 IR 图像 |

### 13.9 验证方式

```bash
# 1. 生成新 manifest
python scripts/generate_new_manifests.py

# 2. 合并所有 manifest
python scripts/merge_manifests_v2.py

# 3. 修复路径（v2 → v3）
python scripts/fix_manifest_paths.py

# 4. 验证合并结果
python -c "
import json, random, os
with open('/root/autodl-tmp/train1/manifests/merged_train_all_v3.json') as f:
    data = json.load(f)
print(f'Total: {len(data[\"samples\"]):,}')
print(f'Paired: {data[\"paired_count\"]:,}')
print(f'RGB-only: {data[\"rgb_only_count\"]:,}')
print(f'IR-only: {data[\"ir_only_count\"]:,}')

# 随机抽检路径可读性
random.seed(42)
for s in random.sample(data['samples'], 200):
    p = s['rgb_path'] or s['ir_path']
    assert os.path.isfile(p), f'Missing: {p}'
print('Spot-check passed.')
"

# 5. 验证数据集加载（推荐使用 precheck 模式）
bash scripts/run_pretrain_multi.sh precheck
```

---

## 14. v3 更新：数据加载鲁棒性增强与路径修复

### 14.1 背景

v2 合并 manifest (`merged_train_all_v2.json`) 中有 **414,985 条路径**（占 24.2%，涉及 12 个数据集）因压缩包解压嵌套结构不一致而断裂，导致训练时大量样本读取失败、静默回退为全零图。v3 的目标：

- 修复所有断裂路径，确保 manifest 100% 可读；
- 增强数据集加载层的字段兼容性、路径解析能力与 IR 位深兼容性；
- 增加可观测性（加载统计、限流日志）；
- 提供"先预检再训练"的工作流（`strict_loading` + `dry_run`）。

### 14.2 路径修复（`scripts/fix_manifest_paths.py`）

#### 14.2.1 断裂原因分析

12 个数据集的路径问题均源于压缩包解压时的目录嵌套差异：

| 数据集 | 断裂路径示例 | 修复方式 | 修复数 |
|--------|-------------|----------|--------|
| UAV123 | `.../UAV123/UAV123/data_seq/UAV123/bike1/` | 去重复目录 + 移除 `data_seq/UAV123/` | 113,476 |
| VT-Tiny-MOT | `.../VT-Tiny-MOT/VT-Tiny-MOT/test2017/` | 去重复目录 | 93,882 |
| UAVDT | `.../UAV-benchmark-M/UAV-benchmark-M/` | 去重复目录 | 40,735 |
| DroneCrowd | `.../test_data/test_data/images/` | 去重复目录 | 33,960 |
| VisDrone-MOT | `.../MOT-test-dev/MOT-test-dev/sequences/` | 去重复目录 | 33,682 |
| AU-AIR | `.../auair2019data/images/frame_...` | 移除多余 `images/` | 32,823 |
| DroneVehicle | `.../test/test/testimg/` | 去重复目录 | 28,439 |
| SeaDronesSee | `Uncompressed Version` → 实际为 `Compressed Version` | 替换目录名 | 14,227 |
| EVD4UAV | `.../EVD4UAV/EVD4UAV/images/` | 去重复目录 | 10,049 |
| VisDrone-DET | `.../DET-train/DET-train/images/` | 去重复目录 | 7,019 |
| DroneRGBT | `.../DroneRGBT/DroneRGBT/DroneRGBT/Test/` | 去重复目录 | 3,424 |
| AeroScapes | `.../AeroScapes/aeroscapes/JPEGImages/` | 移除多余 `aeroscapes/` | 3,269 |
| **合计** | | | **414,985** |

#### 14.2.2 修复策略

脚本采用组合策略：

1. **数据集特定修复**（优先）：AU-AIR 移除 `images/`、AeroScapes 移除 `aeroscapes/`、SeaDronesSee 替换 `Uncompressed→Compressed` 并修正 `.png→.jpg` 扩展名；
2. **通用去重复**：移除路径中连续重复的目录分量（如 `A/A/B` → `A/B`）；
3. **UAV123 特例**：去重后额外移除 `data_seq/UAV123/` 中多余的 `UAV123/`；
4. 每个修复后路径均通过 `os.path.isfile()` 验证。

#### 14.2.3 修复结果

- **修复成功**：414,985 / 414,985（100%）
- **仍然断裂**：0
- 输出文件：`/root/autodl-tmp/train1/manifests/merged_train_all_v3.json`

### 14.3 数据集加载增强（`datasets/multimodal_drone.py`）

#### 14.3.1 Manifest 字段规范化

在 `_init_file` 阶段增加 `_normalize_sample()` 步骤，自动映射常见字段名：

| 用途 | 接受的字段名 |
|------|------------|
| RGB 路径 | `rgb_path`, `rgb`, `visible_path`, `visible`, `img_path`, `image_path` |
| IR 路径 | `ir_path`, `ir`, `infrared_path`, `infrared`, `thermal_path`, `thermal` |
| 样本类型 | `type`, `modality_type`, `sample_type`, `modality` |
| 类型值 | `paired`/`pair`/`rgb_ir`, `rgb_only`/`rgb`/`visible`, `ir_only`/`ir`/`thermal` |

当 `type` 字段缺失时，自动根据 `rgb_path` 和 `ir_path` 是否存在推断类型。

#### 14.3.2 路径解析策略

- **绝对路径**：直接使用；
- **相对路径**：以 manifest 文件所在目录为基准解析（`os.path.join(manifest_dir, rel_path)`）。

#### 14.3.3 IR 读取位深兼容

新增 `_read_ir()` 函数替代原始 `cv2.imread(..., IMREAD_GRAYSCALE)`：

- 使用 `cv2.IMREAD_UNCHANGED` 保留原始位深信息；
- **8-bit**：直接使用；
- **16-bit**：min-max 归一化到 0-255 uint8；
- **浮点型**：clip 并转换；
- **多通道伪彩色 IR**：自动转为灰度；
- 所有 IR 统一输出 `[H, W, 1]` uint8 格式；
- 位深分布通过 `_LoadStats` 汇总统计。

#### 14.3.4 加载失败统计与限流日志

新增线程安全的 `_LoadStats` 类：

- 记录成功/失败次数、失败原因分布、IR 位深分布；
- 前 20 次失败打印详细日志（索引、原因、路径），之后仅累计计数；
- 提供 `get_summary()` 方法输出汇总报告。

#### 14.3.5 可配置的加载策略

`MultiModalDroneDataset` 新增三个参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strict_loading` | bool | `False` | 若为 True，任何样本加载失败立即抛出异常（预检模式） |
| `max_load_fail_ratio` | float | `0.01` | 当失败比例超过此阈值（且已处理 ≥100 样本）时中止训练 |
| `dry_run_samples` | int | `0` | 若 > 0，仅加载前 N 个样本（配合 precheck 使用） |

### 14.4 训练入口更新（`main_dino_rgbir.py`）

#### 14.4.1 新增参数

```
--strict_loading    True/False   预检模式：发现坏样本立即报错
--max_load_fail_ratio  0.001     失败率阈值（超过则中止），0 表示不限
--dry_run_samples   0            仅加载前 N 样本，完成后退出（0=全量）
```

#### 14.4.2 启动摘要

训练启动时输出数据配置摘要：

```
--- Data Config ---
  Manifest: /root/autodl-tmp/train1/manifests/merged_train_all_v3.json
  Mode: file
  Samples: 1520002
  Strict loading: False
  Max fail ratio: 0.001
-------------------
```

#### 14.4.3 Dry-run 模式

当 `--dry_run_samples > 0` 时：

1. 仅加载指定数量样本；
2. 通过 DataLoader 跑最多 10 个 batch 验证形状与解码；
3. 输出加载统计摘要；
4. 自动退出，不进入训练循环。

#### 14.4.4 每 epoch 加载统计

每个 epoch 结束时输出当前累计的加载统计信息（失败数、失败率、IR 位深分布）。

### 14.5 训练脚本更新（`scripts/run_pretrain_multi.sh`）

脚本升级为双模式运行：

```bash
# 预检模式：严格加载 500 样本，有问题立即报错
bash scripts/run_pretrain_multi.sh precheck

# 正式训练：容错模式，失败率超过 0.1% 自动中止
bash scripts/run_pretrain_multi.sh train
```

主要变更：

| 项目 | v2 | v3 |
|------|----|----|
| Manifest | `merged_train_all_v2.json` | `merged_train_all_v3.json` |
| 输出目录 | `checkpoints_v2` | `checkpoints_v3` |
| 运行模式 | 仅 train | `precheck` / `train` 双模式 |
| 数据加载策略 | 无控制 | precheck: `strict_loading=True, dry_run_samples=500`；train: `max_load_fail_ratio=0.001` |

### 14.6 推荐工作流

```bash
# 第一步：预检（不启动训练，快速验证数据可读性）
bash scripts/run_pretrain_multi.sh precheck
# 预期输出：500 样本全部加载成功，打印 batch 形状与统计

# 第二步：正式训练
bash scripts/run_pretrain_multi.sh train
```

### 14.7 验证结果

v3 修复与增强已通过以下测试：

| 测试 | 结果 |
|------|------|
| 随机抽样 200 条路径存在性检查 | 224/224 路径存在（100%） |
| 三种类型样本加载（paired/rgb_only/ir_only） | 全部成功，shape 正确 |
| DataLoader + transform 完整流水线（10 batch） | crops `[B,4,224,224]` × 2 + `[B,4,96,96]` × 6 + view_crop + mask |
| 混合类型 150 样本 DataLoader（18 batch，strict 模式） | 0 failures |
| IR 位深检测 | 8-bit 正确识别，16-bit 路径已就绪 |

### 14.8 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/fix_manifest_paths.py` | **新建** | 一次性修复 v2 manifest 的 414,985 条断裂路径 |
| `datasets/multimodal_drone.py` | **修改** | 字段规范化、相对路径解析、IR 位深兼容、加载统计、可配置加载策略 |
| `main_dino_rgbir.py` | **修改** | 新增 3 个数据加载参数、启动摘要、dry-run 模式、每 epoch 统计 |
| `scripts/run_pretrain_multi.sh` | **修改** | 指向 v3 manifest、双模式（precheck/train）、输出目录更新 |
| `merged_train_all_v3.json` | **新建** | 路径已修复的 1,520,002 样本总 manifest |

---

## 15. TCL (Time-Contrastive Learning) 时序对比学习

### 15.1 背景与目标

DINO-MM 当前以静态图片为训练数据。为未来接入视频序列做准备，实现了 TCL（Time-Contrastive Learning）模态无感知方案：

- 视频帧走 **同一个** DualModalPatchEmbed + ViT 骨干 → TemporalAttention 跨帧融合 → InfoNCE 多帧时序对比损失；
- **默认关闭**，不影响现有静态图片训练流程；
- 数据加载层（SequenceAwareDataset）待视频数据清洗和格式确定后再实现。

### 15.2 核心模块

#### 15.2.1 TemporalAttention（已有，`models/temporal_module.py`）

- 对多帧 patch token 做跨帧时序注意力融合；
- 输入 `[B*T, N, d]`，输出 `[B*T, N, d]`；
- `num_frames=1` 时自动 bypass，零开销；
- 包含可学习/正弦两种时序位置编码；
- 默认 2 层 TemporalAttentionBlock（self-attention + FFN）。

#### 15.2.2 TCLLoss（新增，`dino_loss_rgbir.py`）

InfoNCE 风格的多帧时序对比损失：

- 输入：`[B*T, d]` 帧级 pooled 特征 + `num_frames`（T）；
- 正样本：同序列内的帧对（block-diagonal 掩码，排除自身）；
- 负样本：不同序列的所有帧；
- 具体流程：
  1. L2 归一化 → 构造 `[BT, BT]` 全局余弦相似度矩阵（除以 temperature）；
  2. 正样本掩码：按 `seq_id` 分组，同组帧对为正（T×T 块内，除对角线）；
  3. 对每帧计算 `-(log_numer - log_denom)`，即 InfoNCE；
  4. 取所有帧的均值作为最终 loss；
- `num_frames <= 1` 时直接返回 `0.0`。

#### 15.2.3 PretrainingLoss 扩展

- `__init__` 新增 `tcl_loss=None`、`w_tcl=0.1`；
- `forward` 新增 `tcl_features=None`、`num_frames=1`；
- TCL loss 在 modality completion 之后计算，加入 total；
- `loss_dict['tcl']` 始终有值（无 TCL 时为 0）。

### 15.3 训练管线接入（`main_dino_rgbir.py`）

#### 15.3.1 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_temporal` | `False` | 启用时序分支 |
| `--num_frames` | `1` | 每序列帧数 |
| `--temporal_layers` | `2` | TemporalAttention 层数 |
| `--w_tcl` | `0.1` | TCL 损失权重 |
| `--tcl_temperature` | `0.07` | InfoNCE 温度 |

#### 15.3.2 build_loss

- `--use_temporal` 时创建 `TCLLoss(temperature=args.tcl_temperature)` 并传入 `PretrainingLoss`；
- 未启用时 `tcl_loss=None`，`PretrainingLoss` 不创建 TCL 子模块。

#### 15.3.3 train_one_epoch TCL 前向

在 view_bridge 之后、loss 调用之前插入 TCL 分支：

```python
tcl_features = None
tcl_num_frames = 1

if args.use_temporal and temporal_attn is not None:
    video_frames = None  # 等数据层就绪后由 DataLoader 提供
    if video_frames is not None:
        B_v, T, C, H, W = video_frames.shape
        flat = video_frames.reshape(B_v * T, C, H, W)
        frame_tok = student_backbone(flat, return_all_tokens=True)[:, 1:]
        frame_tok = temporal_attn(frame_tok, num_frames=T)
        tcl_features = frame_tok.mean(dim=1)  # [B_v*T, d]
        tcl_num_frames = T
```

当前 `video_frames` 始终为 `None`，TCL 分支完全不执行。

#### 15.3.4 梯度裁剪

两处 `clip_grad_norm_`（fp16 / 非 fp16）已补充 `temporal_attn.parameters()`（条件判断 not None）。

#### 15.3.5 Checkpoint 保存与恢复

- 保存：两处 `save_checkpoint` dict 增加 `'temporal_attn': temporal_attn.state_dict() if temporal_attn is not None else {}`；
- 恢复：加载时检查 `'temporal_attn' in checkpoint`，若存在则恢复。

#### 15.3.6 日志与参数打印

- `metric_logger` 增加 `'tcl'` 键；
- 日志行在 `--use_temporal` 时追加 `tcl: x.xxxx`；
- 启动摘要打印 TemporalAttention 参数量和 `w_tcl`。

### 15.4 三阶段验证保证

| 场景 | 行为 |
|------|------|
| 不加 `--use_temporal` | TCLLoss 不创建，`tcl_features=None`，`loss_dict['tcl']=0`，完全不影响现有训练 |
| 加 `--use_temporal` 但无视频数据 | `temporal_attn` 创建并保存至 checkpoint，TCL 分支因 `video_frames=None` 跳过，loss 恒为 0 |
| 视频数据就绪后 | 实现视频数据加载层，传入 `video_frames` 张量即可激活全部 TCL 流程 |

### 15.5 未来激活清单

视频数据就绪后，只需完成以下步骤即可激活 TCL：

1. 实现 `SequenceAwareDataset`（或扩展 `MultiModalDroneDataset`），输出 `video_frames: [B, T, C, H, W]`；
2. 在 `train_one_epoch` 中将 `video_frames = None` 替换为从 DataLoader 获取的实际数据；
3. 训练命令添加 `--use_temporal --num_frames T`。

### 15.6 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `dino_loss_rgbir.py` | **修改** | 新增 `TCLLoss` 类；`PretrainingLoss` 扩展 `tcl_loss`/`w_tcl` 参数和 TCL 前向逻辑 |
| `main_dino_rgbir.py` | **修改** | 新增 `--w_tcl`/`--tcl_temperature` 参数；`build_loss` 创建 TCLLoss；`train_one_epoch` 接入 TCL 前向；梯度裁剪补全；checkpoint 保存/恢复补全；日志扩展 |
| `models/temporal_module.py` | 无修改 | 已完整实现，无需改动 |
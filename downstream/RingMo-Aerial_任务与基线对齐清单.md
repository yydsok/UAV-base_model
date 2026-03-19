# RingMo-Aerial 任务与基线对齐清单

> 目标：以 RingMo-Aerial 论文为主，对齐“任务-数据集-对比模型”并落地到可运行 baseline 框架。
> 更新时间：2026-03-15

## 1) 论文任务与数据集（主优先级）

- 语义分割：UAVID、UDD6、FloodNet
- 目标检测：VisDrone-DET、UAVDT、ShipDataset、DroneVehicle（RGB+IR）、AerialSARData
- 场景分类：AID、RESISC-45
- 变化检测：LEVIR-CD、CDD
- 多目标跟踪：VisDrone-MOT、AIR-MOT、AIR-HSAO
- 三维重建：LuoJia-MVS、WHU

## 2) 论文对比模型清单（按任务）

### 2.1 语义分割（表1/2/3）

- DeepLabV3+, DANet, ACNet, OCRNet, SETR, SegFormer, CSWin
- UAVFormer, BiSeNetV2, Segmenter, CoaT, BANet, UNetFormer
- RingMo / RingMo-Aerial (U/A/M)

### 2.2 目标检测（表4/5/6/7/8）

- Light-RCNN, CornerNet, RetinaNet, YOLOv4, Cascade R-CNN
- CEASC, SOD-YOLOv7, MFFSODNet, GLSAN, QueryDet, VistrongerDet, DMNet, HRDNet, ClusDet, SDPDet, AMRNet, OGMN, DDQ-DETR
- UAV-YOLO, TPH-YOLOv5, YOLOv7, NanoDet-Plus, YOLOX-L/Tiny, Center-Net++, PP-PicoDet-L
- Faster R-CNN(OBB), RoITransformer, S2ANet, Oriented R-CNN
- UA-CMDet, Halfway Fusion, CIAN(OBB), Dual-YOLO

### 2.3 场景分类（表9）

- GASSL, SeCo, SatMAE, RVSA, DINO-MC, TOV, SSL4EO
- CMID, CACo, CROMA, SatLas, GFM, Scale-MAE, SkySense
- RingMo / RingMo-Aerial

### 2.4 变化检测（表10）

- FC-Siam-Di, FC-Siam-Conc, IFNet, STANet, BIT, SNUNet, ChangeFormer
- RingMo / RingMo-Aerial

### 2.5 多目标跟踪（表12/13/14）

- SiamMOT, ByteTrack, UAVMOT, OCSORT, MOTR, TrackFormer, FOLT, U2MOT
- CKDNet+SMTNet, StrongSORT, CenterTrack, TGram, RSMOT, CFTrack
- TraDeS, DSFNet, FairMOT, CTracker, MGTrack

### 2.6 三维重建（表11）

- PatchmatchNet, Fast-MVSNet, MVSNet, R-MVSNet, RED-Net, Cas-MVSNet

## 3) 我们当前 baseline 落地映射（可运行优先）

- 分割主线：`mmsegmentation`（UPerNet/Mask2Former/SegFormer/DeepLab/SETR/BiSeNetV2/OCRNet 等）
- 检测主线：`mmdetection_v2`（Cascade R-CNN/Faster R-CNN/RetinaNet 等）+ `mmdetection_v3`（DDQ/RTMDet/新配置参考）
- 旋转检测：`mmrotate`（Oriented R-CNN/RoITransformer/S2ANet）
- 跟踪主线：`mmtracking` + `ByteTrack`
- 变化检测：`BIT_CD` + `open-cd` + `changeformer`
- 场景分类：`mmpretrain` + `train_classification.py`（AID/RESISC-45 线性探针）
- 三维重建：`cas_mvsnet` + `patchmatchnet` + `fastmvsnet` + `mvsnet`

## 4) 新增补齐的关键仓库（RingMo-Aerial 优先）

位于 `/autodl-fs/data/baselines/`：

- `changeformer`
- `geoseg`
- `cas_mvsnet`
- `patchmatchnet`
- `fastmvsnet`
- `mvsnet`
- `motr`
- `trackformer`
- `uavmot`
- `u2mot`
- `oc_sort`
- `strongsort`
- `fairmot`
- `centertrack`
- `tgram`

## 5) 执行策略（强框架优先）

- 同任务优先选择论文中公认强框架进行主表对齐：
  - 分割：Mask2Former / UPerNet
  - 检测：DDQ-DETR / Cascade R-CNN
  - 旋转检测：Oriented R-CNN
  - 跟踪：ByteTrack 系列 + MOTR/TrackFormer
  - 变化检测：BIT + ChangeFormer
  - 三维重建：Cas-MVSNet
- 若论文方法源码不可复现或与现有环境冲突，优先使用同家族强 baseline（OpenMMLab 或官方 repo）并在主表标注“替代实现”。

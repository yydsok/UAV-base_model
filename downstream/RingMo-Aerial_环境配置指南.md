# RingMo-Aerial 基线环境配置指南

## 1. 主环境（检测/分割/跟踪/旋转检测/分类/变化检测）

推荐复用已验证环境 `skysence`：

```bash
conda activate skysence
python -c "import torch,mmcv,mmdet,mmseg,mmtrack,mmrotate; print('ok')"
```

建议版本（已验证）：

- python 3.10.13
- torch 1.13.1+cu117
- mmcv-full 1.7.1
- mmdet 2.28.2
- mmseg 0.30.0
- mmtrack 0.14.0
- mmrotate 0.3.4

## 2. 3D 重建环境（LuoJia-MVS / WHU）

```bash
conda create -n ringmo_mvs python=3.10 -y
conda activate ringmo_mvs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy matplotlib tqdm pyyaml tensorboard
```

按仓库分别补依赖：

```bash
pip install -r /autodl-fs/data/baselines/cas_mvsnet/requirements.txt || true
pip install -r /autodl-fs/data/baselines/patchmatchnet/requirements.txt || true
pip install -r /autodl-fs/data/baselines/fastmvsnet/requirements.txt || true
pip install -r /autodl-fs/data/baselines/mvsnet/requirements.txt || true
```

## 3. 扩展跟踪环境（MOTR/TrackFormer/U2MOT 等）

```bash
conda create -n ringmo_mot_ext python=3.10 -y
conda activate ringmo_mot_ext
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install cython opencv-python scipy matplotlib motmetrics lap
```

然后按需安装：

```bash
pip install -r /autodl-fs/data/baselines/motr/requirements.txt || true
pip install -r /autodl-fs/data/baselines/trackformer/requirements.txt || true
pip install -r /autodl-fs/data/baselines/u2mot/requirements.txt || true
pip install -r /autodl-fs/data/baselines/strongsort/requirements.txt || true
```

## 4. 一键仓库同步

```bash
bash /autodl-fs/data/DINO-MM/downstream/scripts/setup_ringmo_baselines.sh
```

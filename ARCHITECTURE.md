# DINO-MM 架构详细说明（代码阅读版）

> 本文档通过完整阅读代码生成，不依赖任何已有文档。

---

## 一、总体架构

DINO-MM 是一个 **多模态（RGB+IR）自监督预训练** 框架，基于 DINO 自蒸馏机制，
扩展了 6 个辅助损失来学习跨模态、多粒度、多视角的表征。

```
┌──────────────────────────────────────────────────────────────────┐
│                        输入: 4通道图像 [B, 4, H, W]                │
│                   (通道 0-2: RGB, 通道 3: IR)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │          数据增强 (DataAugmentationDINO_RGBIR)                │ │
│  │  ┌─────────────┬──────────────┬───────────────┐             │ │
│  │  │ 2个全局crop   │ 8个局部crop   │ 1个视角crop    │             │ │
│  │  │ 224×224      │ 98×98        │ 224×224(透视变换)│             │ │
│  │  └─────────────┴──────────────┴───────────────┘             │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              ViT Backbone (DualModal + GatedCrossAttention)  │ │
│  │                                                              │ │
│  │  Student: 处理所有 crops (2全局+8局部+视角+模态分支)             │ │
│  │  Teacher: 仅处理 2个全局 crops (EMA更新, 无梯度)                │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      7个损失函数                               │ │
│  │  1. DINO Loss (自蒸馏)          - 权重固定 1.0               │ │
│  │  2. MGCL Loss (多粒度对比)      - 权重自适应                 │ │
│  │  3. CrossModalAlign (跨模态对齐) - 权重自适应                 │ │
│  │  4. ViewInvariance (视角不变)    - 权重自适应                 │ │
│  │  5. ViewBridge (视角桥接)        - 权重自适应                 │ │
│  │  6. InfoMAE (跨模态重建)         - 权重自适应                 │ │
│  │  7. Commit Loss (VQ码本)        - 固定权重                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、数据流水线

### 2.1 数据集 (MultiModalDroneDataset)

**文件**: `datasets/multimodal_drone.py`

每个样本是一张 4 通道图像 `[H, W, 4]`，支持三种模态类型：

| 类型 | RGB (ch 0-2) | IR (ch 3) | modality_mask |
|------|-------------|-----------|---------------|
| paired | 真实图像 | 真实图像 | [1.0, 1.0] |
| rgb_only | 真实图像 | 全零 | [1.0, 0.0] |
| ir_only | 全零 | 真实图像 | [0.0, 1.0] |

IR 读取支持 uint8/uint16/float 多种位深，自动归一化到 0-255。
paired 样本中如果 RGB 和 IR 尺寸不一致，IR 会被 resize 到 RGB 尺寸。

错误处理：非 strict 模式下失败样本返回 512×640 黑色图像（mask=[0,0]），
超过 `max_load_fail_ratio` 阈值后中止训练。

采样器 `WeightedMultiDatasetSampler` 使用 `sqrt(dataset_size)` 加权，
防止大数据集主导训练。

### 2.2 数据增强 (DataAugmentationDINO_RGBIR)

**文件**: `models/transforms_rgbir.py`

增强流程产生 **2 全局 + 8 局部 + 1 视角** 共 11 个 crop：

| Crop | 尺寸 | Scale | 水平翻转 | 垂直翻转 | 90°旋转 | 高斯模糊 | Solarize | SensorDrop |
|------|------|-------|---------|---------|---------|---------|----------|------------|
| 全局1 | 224 | 0.4-1.0 | 50% | 50% | 75% | **100%** | 0% | 50% |
| 全局2 | 224 | 0.4-1.0 | 50% | 50% | 75% | 10% | 20% | 50% |
| 局部×8 | 98 | 0.05-0.4 | 50% | 50% | 75% | 50% | 0% | 50% |
| 视角 | 224 | 0.4-1.0 | 50% | 无 | 无 | 无 | 无 | **无** |

**RandomSensorDrop_RGBIR** (p_drop=0.5)：
- 50% 概率保留双模态
- 25% 概率只保留 RGB（IR 通道置零）
- 25% 概率只保留 IR（RGB 通道置零）
- **视角 crop 不做 SensorDrop**

**AffineViewAugmentation**：透视变换模拟无人机视角变化。
alpha_range=(0.05, 0.35)，5 种变形方向（上/下/左/右/混合）。

**MultiModalNormalize**：
- RGB: ImageNet 均值/标准差 ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])
- IR: 均值=0.5, 标准差=0.5

### 2.3 Collate

`collate_multimodal` 将 crops 按位置分组堆叠：
`[B个第1全局crop, B个第2全局crop, B个第1局部crop, ..., B个第8局部crop]`
modality_mask 堆叠为 `[B, 2]`。

---

## 三、Backbone 架构

### 3.1 ViT 配置

**文件**: `models/vision_transformer_rgbir.py`

| 变体 | embed_dim | depth | num_heads | patch_size | 参数量 |
|------|-----------|-------|-----------|------------|--------|
| vit_tiny | 192 | 12 | 3 | 16 | ~5.7M |
| **vit_small** | **384** | **12** | **6** | **14** | **~22M** |
| vit_base | 768 | 12 | 12 | 16 | ~86M |
| vit_large | 1024 | 24 | 16 | 14 | ~304M |
| vit_huge | 1280 | 32 | 16 | 14 | ~632M |

当前验证实验使用 **vit_small/14**，embed_dim=384, depth=12, num_heads=6。

### 3.2 DualModalPatchEmbed

当 `in_chans=4` 时使用双模态 patch embedding，将 RGB 和 IR 分别投影：

```
输入 x: [B, 4, H, W]
    ├─ x[:, :3] (RGB) ──→ rgb_proj: Conv2d(3, 384, k=14, s=14) ──→ rgb_tok [B, N, 384]
    └─ x[:, 3:] (IR)  ──→ ir_proj:  Conv2d(1, 384, k=14, s=14) ──→ ir_tok  [B, N, 384]

其中 N = (H/14)² = 256 (224px) 或 49 (98px)
```

然后通过 fusion 模式合并。支持 4 种模式：

| 模式 | 机制 | 额外参数 |
|------|------|---------|
| concat | cat([rgb, ir], dim=-1) → Linear(2d, d) + LN | fusion_proj |
| add | rgb + ir 逐元素相加 | 无 |
| cross_attn | 单向：RGB query 关注 IR K/V | MHA + LN |
| **gated_cross_attn** | **双向门控交叉注意力（当前使用）** | **2×MHA + 2×gate + fusion_proj** |

### 3.3 GatedCrossAttention（核心融合模块）

```
rgb_tok [B, N, 384]    ir_tok [B, N, 384]
    │                       │
    ├───Q──→ rgb_to_ir_attn(Q=rgb, K=ir, V=ir) ──→ ir_ctx [B, N, 384]
    │                       │
    │   ┌───Q──→ ir_to_rgb_attn(Q=ir, K=rgb, V=rgb) ──→ rgb_ctx [B, N, 384]
    │   │                   │
    ▼   ▼                   ▼
rgb_enhanced = rgb_tok + sigmoid(rgb_gate) × ir_ctx × ir_avail
ir_enhanced  = ir_tok  + sigmoid(ir_gate)  × rgb_ctx × rgb_avail
    │                       │
    └───────cat──────────────┘
              ↓
    fusion_proj: Linear(768, 384) + LayerNorm(384)
              ↓
        fused_tokens [B, N, 384]
```

- **rgb_to_ir_attn**: MultiheadAttention(384, 6, batch_first=True)
- **ir_to_rgb_attn**: MultiheadAttention(384, 6, batch_first=True)
- **rgb_gate**: 可学习参数 [1, 1, 384]，初始化为 0 → sigmoid(0)=0.5
- **ir_gate**: 同上
- **fusion_proj**: Linear(768, 384) + LayerNorm(384)

**模态缺失处理**：
当 `modality_masks` 指示某模态缺失时，对应的 `ir_avail` 或 `rgb_avail` 为 0，
cross-attention 仍然计算（避免 NaN），但输出被乘以 0 消除。
不使用 `key_padding_mask`（全 mask 时 softmax 产生 NaN）。

### 3.4 Transformer Blocks

标准 pre-norm 结构，每个 block：
```
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(MLP(LayerNorm(x)))

MLP: Linear(384, 1536) → GELU → Linear(1536, 384)
```

DropPath 从 0 线性增加到 `drop_path_rate`（默认 0.1）。
gradient checkpointing 开启时，每个 block 单独 checkpoint，
反向传播时重新计算前向激活值。

### 3.5 完整前向传播

```
输入 x: [B, 4, H, W]
    ↓
DualModalPatchEmbed + GatedCrossAttention → [B, N, 384]
    ↓
Prepend CLS token → [B, 1+N, 384]
    ↓
+ pos_embed (可学习, 支持双三次插值) → [B, 1+N, 384]
    ↓
pos_drop (Dropout)
    ↓
12 × Transformer Block (可选 gradient checkpointing)
    ↓
Final LayerNorm
    ↓
输出:
  return_all_tokens=True  → [B, 1+N, 384] (CLS + 所有 patch tokens)
  return_all_tokens=False → [B, 384] (仅 CLS token)
```

---

## 四、Teacher-Student 自蒸馏

### 4.1 DINOHead

**文件**: `models/dino_head.py`

将 backbone 的 CLS token 映射到高维输出空间：

```
Linear(384, 2048) → GELU
Linear(2048, 2048) → GELU
Linear(2048, 256)
      ↓
L2 Normalize
      ↓
WeightNorm(Linear(256, 65536, bias=False))
      ↓
输出: [B, 65536]
```

`norm_last_layer=True` 时冻结最后一层的 weight_norm gain（防止 collapse）。

### 4.2 MultiCropWrapper

将 backbone + head 组合，高效处理多分辨率 crops：

1. 按空间尺寸分组（全局 224px 一组，局部 98px 一组）
2. 同组内 concat 沿 batch 维度，一次前向
3. 分离 CLS token 和 patch tokens
4. CLS 过 DINOHead 得到投影特征
5. patch tokens 保留供辅助损失使用

返回：
- `student_out`: [B×10, 65536]（2全局+8局部的 CLS 投影）
- `student_tokens`: [B×2, 256, 384]（仅全局 crops 的 patch tokens）

### 4.3 EMA Teacher 更新

```python
# 每个 optimizer step 后执行
m = cosine_schedule(step)  # 从 0.996 → ~1.0
teacher_param = m × teacher_param + (1 - m) × student_param
```

- 应用于 backbone + DINOHead 和 MultiGranularity 模块
- Teacher 不包装 DDP，仅用于推理（`torch.no_grad()`）
- Teacher backbone 无 DropPath（`drop_path_rate=0.0`）

---

## 五、辅助模块

### 5.1 MultiGranularityFeatures (MGCL)

**文件**: `models/multi_granularity.py`

从 patch tokens 提取三个粒度的特征：

```
Patch Tokens [B, N, 384]
    │
    ├─ Token 级: token_proj (Linear(384,384) → GELU → Linear(384,256)) → [B, N, 256]
    │
    ├─ Object 级: SinkhornKnopp(K=8, iters=3, temp=0.1) → 软聚类 [B, N, 8]
    │             → 加权聚合 → [B, 8, 384] → object_proj → [B, 8, 256]
    │
    └─ Image 级: mean_pool → [B, 384] → image_proj → [B, 256]
```

Student 和 Teacher 各有独立的 MG 模块，Teacher 的 MG 通过 EMA 更新。

### 5.2 ModalityCompletion (InfoMAE + VQ)

**文件**: `models/modality_completion.py`

跨模态补全模块，处理缺失模态的重建：

```
特征输入: [rgb_features, ir_features]，各 [B, N, 384]
modality_mask: [B, 2]

每个模态分别:
  ├─ Encoder: Linear(384,384) → GELU → Linear(384,256) → [B, N, 256]
  ├─ VectorQuantizer: 8192 codes × 256 dim → z_q [B, N, 256] + commit_loss
  └─ Decoder:  Linear(256,384) → GELU → Linear(384,384) → [B, N, 384]

缺失模态的特征 = 用另一模态的 latent 解码重建
```

**VectorQuantizer**:
- 码本: nn.Embedding(8192, 256)
- Gumbel-Softmax 软分配（非 hard argmin）
- Straight-through estimator 传梯度
- `tokenize()` 方法: 硬最近邻查找，返回离散 token ID（用于 InfoMAE 目标）

### 5.3 ViewDomainBridge

**文件**: `models/view_bridge.py`

自组织原型学习，用于视角域桥接：

```
CLS Features [B, 384]
    ↓
Projector: Linear(384,384) → GELU → Linear(384,256) → L2 Normalize
    ↓
z [B, 256]
    ↓
Cosine Similarity → Prototypes [64, 256] → logits [B, 64]

特征队列: [4096, 256] 环形缓冲区
每 100 步: Spherical K-Means 重聚类 prototypes
```

- Prototypes 是 **非可学习 buffer**，通过球面 K-Means 在特征队列上聚类得到
- Teacher 的投影特征入队列（`enqueue()`，跨 rank all_gather）
- 多个视角的 logits 收集后传给 ViewBridgeLoss

---

## 六、损失函数

### 6.1 DINOLoss（自蒸馏损失）

```
权重: 固定 1.0（锚点损失，不参与自适应加权）
输入: student_out [B×10, 65536], teacher_out [B×2, 65536]

Teacher: softmax((out - center) / teacher_temp)  # teacher_temp: 0.04→0.07 warmup
Student: log_softmax(out / 0.1)

Loss = Σ_{q ∈ teacher_views} Σ_{v ∈ student_views, v≠q} -q · log(v) / n_pairs

Center: EMA 更新 (momentum=0.9)，跨 rank all_reduce
```

### 6.2 MGCLLoss（多粒度对比损失）

```
权重: w_mgcl (默认 1.0)
输入: student_mg {token, object, image}, teacher_mg {同}

三个粒度各自独立计算 DINO 风格交叉熵:
  Teacher: softmax((feat - center) / (0.1 × 0.4))  # 更锐利
  Student: log_softmax(feat / 0.1)

Loss = token_loss + object_loss + image_loss

每个粒度维护独立的 center buffer (EMA, momentum=0.9)
```

### 6.3 CrossModalAlignLoss（跨模态对齐损失）

```
权重: w_align (默认 1.0)
输入: feat_rgb [B, N, 384], feat_ir [B, N, 384], modality_mask [B, 2]

1. 仅使用 paired 样本（双模态均存在）
2. L2 归一化
3. 余弦相似度矩阵 S = rgb @ ir^T  → [B', N, N]
4. Log-domain Sinkhorn-Knopp (iters=3, temp=0.05) → 最优传输计划 T
5. 双向 soft-target InfoNCE:
   loss_rgb2ir = -Σ T[i,j] · log_softmax(S[i,j]/0.07)
   loss_ir2rgb = 同上但转置

Loss = (loss_rgb2ir + loss_ir2rgb) / 2
```

### 6.4 ViewInvarianceLoss（视角不变损失）

```
权重: w_view (默认 0.5)
输入: feat_original [B, 384], feat_view [B, 384]

1. L2 归一化
2. sim = original @ view^T / 0.07  → [B, B]
3. 对称 InfoNCE: CE(sim, identity) + CE(sim^T, identity)

Loss = (loss_1 + loss_2) / 2
```

`feat_original` = 全局 crop1 的 patch tokens mean-pool
`feat_view` = 视角 crop 的 patch tokens mean-pool

### 6.5 ViewBridgeLoss（视角桥接损失）

```
权重: w_bridge (默认 0.5)
输入: logits_list (2-3 个 [B, 64] 的 prototype logit 向量)

1. 每个 logits → softmax(logits / 1.0) → 概率分布
2. Consistency: 所有视角对之间的对称 KL 散度的均值
3. Sharpness: 平均每样本熵（lambda_sharp=0.1，鼓励锐利分配）
4. Balance: 负的 batch 均值概率分布的熵（lambda_balance=0.02，鼓励均匀使用原型）

Loss = consistency + 0.1 × sharpness - 0.02 × balance
```

### 6.6 InfoMAELoss（跨模态掩码重建损失）

```
权重: w_infomae (默认 1.0)
输入: rgb_tokens [B, N, 384], ir_tokens [B, N, 384], modality_mask

1. 仅使用 paired 样本
2. IR 编码: ir_encoder (冻结) → [B', N, 256] → ir_tokenizer (冻结) → 离散 ID [B', N]
3. 随机掩码: 选 75% 的 patch 位置
4. RGB 投影: rgb_proj (可训练) → [B', N, 256] → 仅取掩码位置
5. 预测: predictor (可训练) → [B', num_masked, 8192]
6. 交叉熵 vs IR token ID

Loss = CE(predicted_logits, ir_token_ids)
```

### 6.7 Commit Loss（VQ 承诺损失）

```
权重: 固定（不参与自适应加权）
来源: VectorQuantizer 内部计算

Loss = MSE(z_q.detach(), z) + MSE(z_q, z.detach())
```

---

## 七、训练循环（每步详细流程）

**文件**: `main_dino_rgbir.py`

```
每个 iteration:

┌─ 1. 更新 LR 和 weight_decay (cosine schedule)
│
├─ 2. 数据到 GPU: crops (列表), modality_mask [B,2], view_crop
│
├─ 3. Teacher 前向 (no_grad):
│     teacher_out = teacher(crops[:2])  # 仅 2 全局 crop
│     → [B×2, 65536]
│
├─ 4. Student 前向:
│     student_out, student_tokens = student(crops[:10], return_backbone_feat=True)
│     → student_out [B×10, 65536], student_tokens [B×2, N, 384]
│
├─ 5. 多粒度特征:
│     global_student_tokens = student_tokens[:B×2]  ← 从 step 4 获取
│     global_teacher_tokens = teacher_backbone(crops[:2], return_all_tokens=True)[:, 1:]
│     student_mg = mg_student(global_student_tokens)
│     teacher_mg = mg_teacher(global_teacher_tokens)  # no_grad
│
├─ 6. 模态分支前向 (用于 Align + InfoMAE):
│     gc1 = crops[0]  # 第一个全局 crop [B, 4, 224, 224]
│
│     rgb_only_input = gc1.clone(); rgb_only_input[:, 3:] = 0  # IR 通道置零
│     ir_only_input  = gc1.clone(); ir_only_input[:, :3] = 0   # RGB 通道置零
│
│     rgb_tokens = student_backbone(rgb_only_input, mask=[1,0])[:, 1:]  ← 额外前向!
│     ir_tokens  = student_backbone(ir_only_input, mask=[0,1])[:, 1:]   ← 额外前向!
│     ir_tokens_detached = ir_tokens.detach()  # stop-gradient
│
│     aug_modality_mask = dataset_mask AND 实际通道存在性(SensorDrop后)
│
├─ 7. 模态补全:
│     completed, completion_losses = mod_completion(
│         [rgb_tokens, ir_tokens],  # 注意: ir_tokens 这里有梯度
│         aug_modality_mask
│     )
│
├─ 8. 视角不变 + 桥接:
│     feat_original = mean_pool(student_tokens[:B, :, :])  # 全局 crop1 的 patch tokens
│     → student_backbone 的第 4 步已有结果
│
│     bridge_logits_1 = view_bridge(feat_original)
│     bridge_logits_2 = view_bridge(mean_pool(student_tokens[B:2B]))  # 全局 crop2
│
│     if view_crop exists:
│         view_tokens = student_backbone(view_crop)  ← 额外前向!
│         feat_view = mean_pool(view_tokens[:, 1:])
│         bridge_logits_3 = view_bridge(feat_view)
│
│     teacher_feat = teacher_backbone(crops[0])  # 入队列用
│     view_bridge.enqueue(teacher_projected_feat)
│
├─ 9. 计算总损失 (PretrainingLoss.forward):
│     dino_loss = DINOLoss(student_out, teacher_out, epoch)          ← 权重 1.0
│     mgcl_loss = MGCLLoss(student_mg, teacher_mg)                   ← 自适应
│     align_loss = CrossModalAlignLoss(rgb_tok, ir_tok_detached, mask) ← 自适应
│     view_loss = ViewInvarianceLoss(feat_original, feat_view)        ← 自适应
│     bridge_loss = ViewBridgeLoss(bridge_logits_list)                ← 自适应
│     infomae_loss = InfoMAELoss(rgb_tok, ir_tok, mask)               ← 自适应
│
│     if adaptive_weighting:
│         aux_total, ew = UncertaintyWeighting({mgcl, align, view, bridge, infomae})
│         total = dino_loss + aux_total + commit_loss
│     else:
│         total = dino + w_mgcl×mgcl + w_align×align + ... + commit
│
├─ 10. 反向传播:
│      fp16_scaler.scale(total / accum_steps).backward()
│
│      # 手动同步 uncertainty 参数梯度 (loss_fn 不在 DDP 中)
│      for p in loss_fn.uncertainty.parameters():
│          if p.grad is None: p.grad = zeros_like(p)
│          dist.all_reduce(p.grad)
│          p.grad /= world_size
│
├─ 11. 优化器更新 (仅在 accumulation 边界):
│      unscale → clip_grad_norm_(3.0) → cancel_gradients_last_layer → step
│
└─ 12. EMA Teacher 更新:
       m = cosine_schedule(step)  # 0.996 → ~1.0
       teacher = m × teacher + (1-m) × student
       mg_teacher = m × mg_teacher + (1-m) × mg_student
```

### 7.1 Student Backbone 前向调用次数统计

**每个训练 iteration 中，student backbone 被调用的次数：**

| 调用 | 输入 | 目的 |
|------|------|------|
| 1 | 全部 crops [B×10, 4, H, W] | DINO student_out + student_tokens |
| 2 | rgb_only_input [B, 4, 224, 224] | CrossModalAlign + InfoMAE 的 RGB 特征 |
| 3 | ir_only_input [B, 4, 224, 224] | CrossModalAlign + InfoMAE 的 IR 特征 |
| 4 | view_crop [B, 4, 224, 224]（如果存在） | ViewInvariance + ViewBridge |

**总计: 4 次 backbone 前向**（其中只有第 1 次包含局部 crops）。
Teacher backbone 也被调用 2 次（全局 crops 的 CLS + 全局 crops 的 patch tokens）。

---

## 八、DDP 分布式设置

```python
# 包装在 DDP 中的模块 (find_unused_parameters=True):
student              # ViT backbone + DINOHead
mg_student           # MultiGranularityFeatures
mod_completion       # ModalityCompletion
view_bridge          # ViewDomainBridge

# 不在 DDP 中的:
teacher              # 仅推理，no_grad
mg_teacher           # EMA 更新
loss_fn              # 包含 UncertaintyWeighting，梯度手动 all_reduce
```

`find_unused_parameters=True` 是必需的：当 batch 内无 paired 数据时，
GatedCrossAttention 的参数不参与计算图。

---

## 九、自适应加权 (Kendall Uncertainty Weighting)

```
DINO loss 固定权重 1.0（锚点）

对每个辅助损失 i, 维护可学习参数 s_i:
  初始值: s_i = -log(w_i)  (w_i 为静态权重)
  约束: clamp(-6, 6)  (FP16 安全，exp 范围 [0.0025, 403])

加权公式:
  total = L_dino + Σ(exp(-s_i) × L_i + 0.5 × s_i) + commit_loss

当某个 loss 在某 batch 中缺失时:
  weighted_sum += 0.0 × s_i  (确保参数在计算图中，防止 NCCL 死锁)

优化器:
  uncertainty 参数单独一组: weight_decay=0, lr_scale=0.01
```

---

## 十、已确认的架构问题（附训练实验证据）

> 以下问题基于代码分析 + vit_small/14 在 37 万样本子集上 9 epoch 训练的实际 loss 曲线确认。

### 问题 1 [严重]: backbone 调用 4 次 → 梯度冲突压制 DINO 主任务

**现象**: DINO loss 在 epoch 0 低 LR 时降到 2.42，LR 升高后反弹到 7.29，
warmup 后恢复到 6.57 但始终无法回到 epoch 0 水平。

**根因**: 同一个 student backbone 每 iteration 被调用 4 次：

```
调用1: 混合模态 crops  ──→ DINO + MGCL 梯度     ← 主任务
调用2: 仅 RGB 输入      ──→ Align + InfoMAE 梯度  ← 辅助
调用3: 仅 IR 输入       ──→ Align + InfoMAE 梯度  ← 辅助
调用4: 视角 crop        ──→ View + Bridge 梯度    ← 辅助
```

4 路梯度全部累加到同一组 backbone 参数。当 LR 较大时，辅助损失的梯度与
DINO 梯度方向冲突，把 backbone 拉向不利于自蒸馏的方向。

**实验证据**:
```
E0 结尾 (LR=0.00015):  dino = 2.42  ← LR 小，辅助梯度弱，DINO 自由下降
E4 结尾 (LR=0.00075):  dino = 7.29  ← LR 大，辅助梯度强，DINO 被拉高 3x
E8 结尾 (LR=0.00071):  dino = 6.57  ← 恢复缓慢，远不及 E0 水平
```

### 问题 2 [严重]: InfoMAE loss 上升 — 冻结随机 tokenizer 产生噪声目标

**现象**: InfoMAE loss 从 epoch 1 的 4.56 **上升**到 epoch 6 的 4.95，从未收敛。

**根因**: `ir_encoder` 和 `ir_tokenizer` 都是 **冻结的随机初始化**。

```python
# InfoMAE 的预测目标生成过程:
ir_encoder(ir_features)  # 冻结的随机 Linear，输出无语义含义
  → ir_tokenizer.tokenize()  # 冻结的随机码本，最近邻查找
  → 离散 token ID  # 完全随机的分类目标
```

网络被迫用 RGB 特征预测 IR 的随机 token ID（8192 类分类任务），
等价于拟合随机标签。这不仅浪费计算，还通过梯度干扰 backbone 学习。

**实验证据**:
```
Epoch 0 avg: infomae = 5.48
Epoch 1 avg: infomae = 4.56  ← 唯一一次下降（memorization）
Epoch 6 avg: infomae = 4.95  ← 反弹
Epoch 8 avg: infomae = 4.90  ← 永远在 4.5-5.0 震荡

随机 8192 分类的理论交叉熵下界: ln(8192) = 9.01
实际 loss 稳定在 ~4.9 ≈ 约 exp(4.9) ≈ 134 个有效类 → 过拟合到少数随机模式
```

**自适应权重验证**: ew_mae 从 1.000 降到 0.952，自动降权此噪声损失，
但降速太慢（lr_scale=0.01），损害已经造成。

### 问题 3 [中等]: align loss 下降极慢 — 合成输入 + stop-gradient

**现象**: align 从 5.25 → 3.39（9 个 epoch 降了 35%），是所有 loss 中最慢的。

**根因 A**: 合成输入通过 GatedCrossAttention 时产生无意义的 cross-attention。

```python
# 仅 RGB 输入通过 GCA 时:
rgb_tok = rgb_proj(real_rgb_data)          # 有意义的 token
ir_tok  = ir_proj(zeros)                   # Conv2d(zeros) ≠ zeros（有 bias!）
                                           # ir_tok 是纯 bias 值，无语义

ir_ctx = rgb_to_ir_attn(Q=rgb_tok, K=ir_tok, V=ir_tok)  # 注意力基于无意义 K/V
rgb_enhanced = rgb_tok + sigmoid(gate) × ir_ctx × 0       # × ir_avail=0 消除了

# 但问题是: rgb_enhanced 之后还要和 ir_enhanced 一起过 fusion_proj:
fused = fusion_proj(cat([rgb_enhanced, ir_enhanced], dim=-1))
# ir_enhanced 包含了基于 bias 的无意义特征!
```

即使 gate 乘以 0，`fusion_proj(cat([rgb, ir], dim=-1))` 仍然混入了无意义的 IR 分支特征。

**根因 B**: IR 侧 stop-gradient。

```python
ir_tokens_detached = ir_tokens.detach()  # align loss 不回传 IR 分支梯度
```

仅 RGB 分支学习对齐，IR 分支完全不参与 → 单侧优化效率低。

### 问题 4 [中等]: bridge loss 平坦震荡 — 原型重聚类导致跳变

**现象**: bridge loss 始终在 0.24-0.45 之间震荡，epoch 均值稳定在 0.28-0.29。

**根因**: ViewDomainBridge 的 prototypes 每 100 步通过 spherical K-Means 重聚类。
重聚类瞬间所有样本的 prototype logits 分布突变，导致 consistency loss（KL 散度）
在重聚类前后产生跳变。

此外，bridge loss 由三项组成，方向可能矛盾：
- consistency（降低多视角分配差异）
- sharpness（鼓励锐利分配，增加确定性）
- balance（鼓励均匀使用原型，增加随机性）

sharpness 和 balance 目标相互矛盾（一个要集中，一个要分散），
在 lambda_sharp=0.1 + lambda_balance=0.02 的配比下可能达到一个稳态。

### 问题 5 [低]: CrossModalAlign 中 IR stop-gradient 的连锁效应

IR 分支通过 `detach()` 被阻断了 align loss 的梯度，但在 InfoMAE 中 IR tokens
是有梯度的（`mod_completion` 接收未 detach 的 `ir_tokens`）。

这造成了不一致：
- Align 中 IR 是锚点（不学习），RGB 向 IR 对齐
- InfoMAE 中 IR 是可学习的目标

IR 分支从两个损失收到矛盾信号。

### 问题 6 [低]: 视角 crop 无 SensorDrop → 模态分布不匹配

视角 crop 始终保留完整双模态（无 SensorDrop），但原始全局 crop 有 50% 概率丢弃
一个模态。ViewInvariance loss 要求两者特征一致，但输入的模态组成不同。

---

## 十一、训练实验数据（vit_small/14, 37 万样本, 单卡 A800）

### 11.1 每 epoch 平均 loss

| Epoch | total | dino | mgcl | align | view | bridge | infomae | Peak LR |
|-------|-------|------|------|-------|------|--------|---------|---------|
| 0 | 21.44 | 6.35 | 1.94 | 5.25 | 3.05 | 0.38 | 5.48 | 0.00015 |
| 1 | 17.59 | 6.06 | 0.59 | 4.70 | 1.71 | 0.29 | 4.56 | 0.00030 |
| 2 | 17.95 | 6.86↑ | 0.61 | 4.31 | 1.44 | 0.29 | 4.66↑ | 0.00038 |
| 3 | 17.92 | 7.19↑ | 0.59 | 4.02 | 1.28 | 0.29 | 4.73↑ | 0.00053 |
| 4 | 17.80 | **7.29**↑ | 0.56 | 3.79 | 1.16 | 0.28 | 4.87↑ | 0.00068 |
| 5(warmup结束) | 17.39 | 7.11↓ | 0.51 | 3.65 | 1.08 | 0.28 | **4.93**↑ | **0.00075** |
| 6 | 16.88 | 6.84↓ | 0.45 | 3.54 | 1.00 | 0.29 | **4.95** | 0.00074 |
| 7 | 16.48 | 6.67↓ | 0.41 | 3.45 | 0.95 | 0.28 | 4.93 | 0.00073 |
| 8 | 16.18 | 6.57↓ | 0.38 | 3.39 | 0.90 | 0.28 | 4.90 | 0.00071 |

### 11.2 自适应权重变化趋势

| Epoch | ew_mgcl | ew_align | ew_view | ew_bridge | ew_infomae |
|-------|---------|----------|---------|-----------|------------|
| 0 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 |
| 4 | 0.996 | 0.985 | 0.496 | 0.508 | 0.984 |
| 8 | 1.017↑ | 0.955↓ | 0.499 | 0.523↑ | 0.952↓ |

解读：
- ew_mgcl ↑: MGCL 干净收敛，应被加权 → 正确
- ew_align ↓: align 收敛慢，应被降权 → 正确
- ew_infomae ↓: InfoMAE 不收敛，应被降权 → 正确，但降速太慢
- ew_bridge ↑: bridge 值很小，uncertainty 认为应该加权 → 可能不合理

### 11.3 Loss 分类评估

| Loss | 收敛状态 | 问题 | 建议 |
|------|---------|------|------|
| DINO | ⚠️ warmup 后恢复中但远不及 E0 | 被辅助梯度冲突压制 | 核心问题，需架构改进 |
| MGCL | ✅ 稳定下降 | 无 | 保留 |
| Align | ⚠️ 极慢下降 | 合成输入 + stop-grad | 需重新设计输入方式 |
| View | ✅ 稳定下降 | 轻微模态不匹配 | 可保留 |
| Bridge | ❌ 平坦震荡 | 原型跳变 + 目标矛盾 | 考虑移除或简化 |
| InfoMAE | ❌ 不收敛，反升 | 随机 tokenizer | 必须修复或移除 |
| Commit | ✅ 已收敛到 ~0 | 无 | 保留 |

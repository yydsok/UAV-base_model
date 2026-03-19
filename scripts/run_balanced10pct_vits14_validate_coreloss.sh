#!/bin/bash
# =============================================================================
# 当前架构/损失设计验证：vit_small + 10% balanced + 核心三损失
# 目标：
# 1. 验证当前实现没有明显代码错误
# 2. 验证 DINO / MGCL / View 三项 loss 能稳定下降
# 3. 在和 3_6 更可比的 teacher_temp=0.07 下做 sanity check
# =============================================================================

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/train1/manifests/subset_10pct_balanced.json}"
DATA_MODE="file"

ARCH="vit_small"
PATCH_SIZE=14
IN_CHANS=4
FUSION="gated_cross_attn"
OUT_DIM=65536
DROP_PATH_RATE=0.1

EPOCHS=10
BATCH_SIZE=640
LR=0.00020
WARMUP_EPOCHS=5
MIN_LR=1e-6
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
MOMENTUM_TEACHER=0.996
FREEZE_LAST_LAYER=1

WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.07
WARMUP_TEACHER_TEMP_EPOCHS=5

GLOBAL_CROP_SIZE=224
LOCAL_CROP_SIZE=98
LOCAL_CROPS_NUMBER=8

W_MGCL=0.2
W_VIEW=0.1
W_BRIDGE=0.0

ADAPTIVE_WEIGHTING=False
USE_PCGRAD=False
GRADIENT_CHECKPOINTING=True
ACCUMULATE_GRAD_BATCHES=1

USE_FP16=True
NUM_WORKERS=12
SAVE_EVERY=2
LOG_EVERY=20

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/root/autodl-tmp/train1/balanced10pct_vits14_validate_coreloss_${RUN_TAG}"

echo "============================================================"
echo "  当前架构验证：vit_small + 10% balanced + 核心三损失"
echo "============================================================"
echo "  Data: ${DATA_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Batch/GPU: ${BATCH_SIZE}"
echo "  LR arg: ${LR}"
echo "  Teacher temp: ${WARMUP_TEACHER_TEMP} -> ${TEACHER_TEMP}"
echo "  Losses: DINO + MGCL(${W_MGCL}) + View(${W_VIEW})"
echo "  Bridge: OFF"
echo "  Adaptive weighting: OFF"
echo "  PCGrad: OFF"
echo "============================================================"

[ ! -f "$DATA_PATH" ] && echo "ERROR: Manifest not found: $DATA_PATH" && exit 1
mkdir -p "$OUTPUT_DIR"

ARGS="
    --arch $ARCH
    --patch_size $PATCH_SIZE
    --in_chans $IN_CHANS
    --fusion $FUSION
    --out_dim $OUT_DIM
    --drop_path_rate $DROP_PATH_RATE
    --use_gradient_checkpointing $GRADIENT_CHECKPOINTING
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES
    --data_path $DATA_PATH
    --data_mode $DATA_MODE
    --output_dir $OUTPUT_DIR
    --epochs $EPOCHS
    --batch_size_per_gpu $BATCH_SIZE
    --lr $LR
    --warmup_epochs $WARMUP_EPOCHS
    --min_lr $MIN_LR
    --weight_decay $WEIGHT_DECAY
    --weight_decay_end $WEIGHT_DECAY_END
    --clip_grad $CLIP_GRAD
    --momentum_teacher $MOMENTUM_TEACHER
    --freeze_last_layer $FREEZE_LAST_LAYER
    --warmup_teacher_temp $WARMUP_TEACHER_TEMP
    --teacher_temp $TEACHER_TEMP
    --warmup_teacher_temp_epochs $WARMUP_TEACHER_TEMP_EPOCHS
    --global_crop_size $GLOBAL_CROP_SIZE
    --local_crop_size $LOCAL_CROP_SIZE
    --local_crops_number $LOCAL_CROPS_NUMBER
    --w_mgcl $W_MGCL
    --w_view $W_VIEW
    --w_bridge $W_BRIDGE
    --adaptive_weighting $ADAPTIVE_WEIGHTING
    --use_pcgrad $USE_PCGRAD
    --save_every $SAVE_EVERY
    --log_every $LOG_EVERY
    --use_fp16 $USE_FP16
    --num_workers $NUM_WORKERS
    --strict_loading False
    --max_load_fail_ratio 0.02
"

echo "Launching at $(date)..."
PYTHONUNBUFFERED=1 python main_dino_rgbir.py $ARGS 2>&1 | tee "${OUTPUT_DIR}/train.log"
echo "Finished at $(date). Output: ${OUTPUT_DIR}"

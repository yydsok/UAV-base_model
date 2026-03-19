#!/bin/bash
# =============================================================================
# DINO-MM 第二阶段：从第一阶段继续，小权重加回辅助损失
# 用法：
#   STAGE1_CKPT=/path/to/checkpoint_latest.pth bash scripts/run_stage2_multiloss_from_stage1.sh
# =============================================================================

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1

STAGE1_CKPT="${STAGE1_CKPT:-/root/autodl-tmp/train1/stage1_mainloss_10ep_20260310_150353/checkpoint_latest.pth}"
DATA_PATH="${DATA_PATH:-/tmp/subset_smoke_5permil.json}"
DATA_MODE="file"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage1 checkpoint not found: $STAGE1_CKPT"
    exit 1
fi

[ ! -f "$DATA_PATH" ] && echo "ERROR: Manifest not found: $DATA_PATH" && exit 1

ARCH="vit_small"
PATCH_SIZE=14
IN_CHANS=4
FUSION="gated_cross_attn"
OUT_DIM=65536
DROP_PATH_RATE=0.1

STAGE2_EPOCHS=10
BATCH_SIZE=512
LR=0.00020
WARMUP_EPOCHS=5
MIN_LR=1e-6
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
MOMENTUM_TEACHER=0.996
FREEZE_LAST_LAYER=1

WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=5

GLOBAL_CROP_SIZE=224
LOCAL_CROP_SIZE=98
LOCAL_CROPS_NUMBER=8

W_MGCL=0.1
W_VIEW=0.05
W_BRIDGE=0.02

ADAPTIVE_WEIGHTING=False
USE_PCGRAD=False
GRADIENT_CHECKPOINTING=True
ACCUMULATE_GRAD_BATCHES=1

USE_FP16=True
NUM_WORKERS=8
SAVE_EVERY=2
LOG_EVERY=10

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/root/autodl-tmp/train1/stage2_multiloss_from_stage1_${RUN_TAG}"

START_EPOCH=$(python - "$STAGE1_CKPT" <<'PY'
import torch
import sys
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", -1)) + 1)
PY
)
TOTAL_EPOCHS=$((START_EPOCH + STAGE2_EPOCHS))

echo "============================================================"
echo "  DINO-MM 第二阶段：小权重加回辅助损失"
echo "============================================================"
echo "  Stage1 CKPT: ${STAGE1_CKPT}"
echo "  Data: ${DATA_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Start epoch: ${START_EPOCH}"
echo "  Total epochs: ${TOTAL_EPOCHS}"
echo "  Stage2 extra epochs: ${STAGE2_EPOCHS}"
echo "  Losses: DINO + MGCL(${W_MGCL}) + View(${W_VIEW}) + Bridge(${W_BRIDGE})"
echo "============================================================"

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
    --epochs $TOTAL_EPOCHS
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
    --resume $STAGE1_CKPT
"

echo "Launching at $(date)..."
PYTHONUNBUFFERED=1 python main_dino_rgbir.py $ARGS 2>&1 | tee "${OUTPUT_DIR}/train.log"
echo "Finished at $(date). Output: ${OUTPUT_DIR}"

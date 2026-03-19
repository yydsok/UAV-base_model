#!/bin/bash
# =============================================================================
# DINO-MM ViT-Large/14 Production Pretraining — V2 Refactored Architecture
#
# Unified single forward pass, Sinkhorn-Knopp ViewBridge,
# DINO-anchored PCGrad. No CrossModalAlign, no VQ tokenizer.
#
# Architecture: ViT-Large/14 (337.5M) + DualModal GatedCrossAttention
# Dataset: merged_full_pretrain_v2.json (5.56M RGB+IR samples)
# Effective batch: 160 × 8 GPUs = 1280
# Multi-crop: 2×224 global + 8×98 local
#
# Usage:
#   bash scripts/run_pretrain_vithuge_production.sh
#   bash scripts/run_pretrain_vithuge_production.sh resume
# =============================================================================

set -e

MODE="${1:-start}"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# --- NCCL robustness ---
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1

# --- Data ---
DATA_PATH="/root/autodl-tmp/train1/manifests/merged_full_pretrain_v2_clean.json"
DATA_MODE="file"

# --- Model ---
ARCH="vit_large"
PATCH_SIZE=14
IN_CHANS=4
FUSION="gated_cross_attn"
OUT_DIM=65536
DROP_PATH_RATE=0.3

# --- Memory optimization ---
GRADIENT_CHECKPOINTING=True
ACCUMULATE_GRAD_BATCHES=1

# --- Training ---
EPOCHS=100
BATCH_SIZE=160                     # per GPU
LR=0.0005                         # base LR (auto-scaled)
WARMUP_EPOCHS=10
MIN_LR=1e-6
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
MOMENTUM_TEACHER=0.996
FREEZE_LAST_LAYER=3

# --- Teacher temperature ---
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.07
WARMUP_TEACHER_TEMP_EPOCHS=30

# --- Multi-crop ---
GLOBAL_CROP_SIZE=224
LOCAL_CROP_SIZE=98
LOCAL_CROPS_NUMBER=8

# --- Loss weights ---
W_MGCL=1.0
W_VIEW=0.5
W_BRIDGE=0.5

# --- Adaptive loss weighting ---
ADAPTIVE_WEIGHTING=True

# --- PCGrad (DINO-anchored gradient surgery) ---
USE_PCGRAD=True

# --- Output ---
OUTPUT_DIR="/root/autodl-tmp/train1/pretrain_vitl14_v2_production"
SAVE_EVERY=3
LOG_EVERY=50

# --- Other ---
USE_FP16=True
NUM_WORKERS=10

# --- Wandb ---
WANDB_KEY="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"
WANDB_PROJECT="DINO-MM"
WANDB_RUN_NAME="vitl14_v2_refactored_ep100"

# --- Detect GPUs ---
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE_GRAD_BATCHES))

echo "============================================================"
echo "  DINO-MM ViT-Large/14 V2 Production Pretraining"
echo "============================================================"
echo "  GPUs: ${NUM_GPUS}x A800-80GB"
echo "  Arch: ${ARCH}/${PATCH_SIZE}, Fusion: ${FUSION}"
echo "  Batch: ${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs = ${EFFECTIVE_BATCH}"
echo "  Epochs: ${EPOCHS}, LR: ${LR}"
echo "  PCGrad: ${USE_PCGRAD} (DINO-anchored)"
echo "  Adaptive weighting: ${ADAPTIVE_WEIGHTING}"
echo "  Unified forward: YES (1 student + 1 teacher pass)"
echo "  ViewBridge: Sinkhorn-Knopp OT (no K-Means)"
echo "  CrossModalAlign: REMOVED"
echo "  Data: $(basename ${DATA_PATH})"
echo "  Output: ${OUTPUT_DIR}"
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
    --wandb_key $WANDB_KEY
    --wandb_project $WANDB_PROJECT
    --wandb_run_name $WANDB_RUN_NAME
    --strict_loading False
    --max_load_fail_ratio 0.02
"

if [ "$MODE" = "resume" ]; then
    CKPT="${OUTPUT_DIR}/checkpoint_latest.pth"
    if [ -f "$CKPT" ]; then
        echo "Resuming from: $CKPT"
        ARGS="$ARGS --resume $CKPT"
    else
        echo "WARNING: No checkpoint at $CKPT - starting from scratch"
    fi
fi

echo ""
echo "Launching at $(date)..."
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=$NUM_GPUS main_dino_rgbir.py $ARGS 2>&1 | tee "${OUTPUT_DIR}/train_log.txt"

echo ""
echo "Training finished at $(date). Checkpoints: $OUTPUT_DIR"

#!/bin/bash
# =============================================================================
# DINO-MM RGB+IR Demo Training Script
# =============================================================================
#
# Usage:
#   # Step 1: Clean data and generate manifest
#   python scripts/clean_rgbt_tiny.py \
#       --data_root /root/autodl-tmp/data/RGBT-Tiny \
#       --output_dir /root/autodl-tmp/data/RGBT-Tiny \
#       --split train
#
#   # Step 2: Run training
#   bash scripts/run_demo_rgbir.sh
#
# =============================================================================

set -e

# --- Configuration ---
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Data
DATA_PATH="/root/autodl-tmp/data/RGBT-Tiny/manifest_train.json"
DATA_MODE="file"

# Model
ARCH="vit_small"
PATCH_SIZE=8
IN_CHANS=4
OUT_DIM=65536

# Training
EPOCHS=50
BATCH_SIZE=32
LR=0.0005
WARMUP_EPOCHS=10

# Multi-crop
GLOBAL_CROP_SIZE=224
LOCAL_CROP_SIZE=96
LOCAL_CROPS_NUMBER=6

# Output
OUTPUT_DIR="./checkpoints/demo_rgbir"
SAVE_EVERY=10
LOG_EVERY=50

# Mixed precision
USE_FP16=True

# Workers
NUM_WORKERS=8

# --- Detect GPU count ---
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "Detected ${NUM_GPUS} GPU(s)"

# --- Check manifest exists ---
if [ ! -f "$DATA_PATH" ]; then
    echo "Manifest not found: $DATA_PATH"
    echo "Run: python scripts/clean_rgbt_tiny.py --data_root /root/autodl-tmp/data/RGBT-Tiny --split train"
    exit 1
fi

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Run training ---
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU training with ${NUM_GPUS} GPUs"
    torchrun --nproc_per_node=$NUM_GPUS main_dino_rgbir.py \
        --arch $ARCH \
        --patch_size $PATCH_SIZE \
        --in_chans $IN_CHANS \
        --out_dim $OUT_DIM \
        --data_path $DATA_PATH \
        --data_mode $DATA_MODE \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size_per_gpu $BATCH_SIZE \
        --lr $LR \
        --warmup_epochs $WARMUP_EPOCHS \
        --global_crop_size $GLOBAL_CROP_SIZE \
        --local_crop_size $LOCAL_CROP_SIZE \
        --local_crops_number $LOCAL_CROPS_NUMBER \
        --save_every $SAVE_EVERY \
        --log_every $LOG_EVERY \
        --use_fp16 $USE_FP16 \
        --num_workers $NUM_WORKERS \
        2>&1 | tee "$OUTPUT_DIR/train_log.txt"
else
    echo "Single-GPU training"
    python main_dino_rgbir.py \
        --arch $ARCH \
        --patch_size $PATCH_SIZE \
        --in_chans $IN_CHANS \
        --out_dim $OUT_DIM \
        --data_path $DATA_PATH \
        --data_mode $DATA_MODE \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size_per_gpu $BATCH_SIZE \
        --lr $LR \
        --warmup_epochs $WARMUP_EPOCHS \
        --global_crop_size $GLOBAL_CROP_SIZE \
        --local_crop_size $LOCAL_CROP_SIZE \
        --local_crops_number $LOCAL_CROPS_NUMBER \
        --save_every $SAVE_EVERY \
        --log_every $LOG_EVERY \
        --use_fp16 $USE_FP16 \
        --num_workers $NUM_WORKERS \
        2>&1 | tee "$OUTPUT_DIR/train_log.txt"
fi

echo ""
echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"

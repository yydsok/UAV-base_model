#!/bin/bash
# =============================================================================
# DINO-MM RGB+IR Multi-Dataset Pretraining Script (clean manifest)
# 使用 merged_pretrain_clean.json（全量路径已验证）进行多数据集联合预训练
#
# Usage:
#   # Step 1: PIL 全量加载测试（34 数据集，每个 50 张）
#   bash scripts/run_pretrain_multi.sh loadtest
#
#   # Step 2: 训练管线 precheck（500样本 dry-run，不训练）
#   bash scripts/run_pretrain_multi.sh precheck
#
#   # Step 3: Full training (fault-tolerant mode)
#   bash scripts/run_pretrain_multi.sh train
#
#   # Default (no arg): runs training
#   bash scripts/run_pretrain_multi.sh
# =============================================================================

set -e

# --- Mode ---
MODE="${1:-train}"

# --- Configuration ---
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Data: merged manifest clean — all paths verified
DATA_PATH="/root/autodl-tmp/train1/manifests/merged_pretrain_clean.json"
DATA_MODE="file"

# Model
ARCH="vit_small"
PATCH_SIZE=16
IN_CHANS=4
FUSION="concat"
OUT_DIM=65536

# Pretrained: load DINO ViT-S/16 ImageNet weights into rgb_proj
PRETRAINED="/root/autodl-tmp/train1/dino_deitsmall16_pretrain.pth"
RESUME=""       # e.g. /path/to/checkpoint_latest.pth

# Training
EPOCHS=3
BATCH_SIZE=128
LR=0.0005
WARMUP_EPOCHS=1
MIN_LR=1e-6
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
MOMENTUM_TEACHER=0.996

# Temperature
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=0

# Multi-crop
GLOBAL_CROP_SIZE=224
LOCAL_CROP_SIZE=96
LOCAL_CROPS_NUMBER=6

# Multi-granularity
NUM_CLUSTERS=8
PROJ_DIM=256

# Loss weights
W_MGCL=1.0
W_ALIGN=1.0
W_VIEW=0.5
W_BRIDGE=0.5
# Note: w_latent and w_rec were removed in V3; current mainline no longer uses InfoMAE.
# W_INFOMAE, bridge_queue_size, bridge_update_interval, ot_temperature, sinkhorn_iters, use_pcgrad
# use argparse defaults unless explicitly overridden

# View-domain bridge (v2: fixed from broken v1 params)
BRIDGE_PROJ_DIM=256
BRIDGE_NUM_PROTOTYPES=64
BRIDGE_TEMP=0.07
BRIDGE_LAMBDA_SHARP=0.1
BRIDGE_LAMBDA_BALANCE=0.02

# Output
OUTPUT_DIR="/root/autodl-tmp/train1/checkpoints_test_clean"
SAVE_EVERY=1
LOG_EVERY=100

# Mixed precision
USE_FP16=True

# Workers
NUM_WORKERS=8

# --- Detect GPU count ---
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "============================================="
echo "DINO-MM RGB+IR Pretraining (clean manifest)"
echo "============================================="
echo "Mode: $MODE"
echo "Detected ${NUM_GPUS} GPU(s)"
echo "Data: $DATA_PATH"
echo "Arch: $ARCH, Patch: $PATCH_SIZE, Channels: $IN_CHANS, Fusion: $FUSION"
echo "Pretrained: $PRETRAINED"
echo "Epochs: $EPOCHS, Batch/GPU: $BATCH_SIZE, Total Batch: $((BATCH_SIZE * NUM_GPUS))"
echo "Bridge: temp=$BRIDGE_TEMP sharp=$BRIDGE_LAMBDA_SHARP balance=$BRIDGE_LAMBDA_BALANCE"
echo "Output: $OUTPUT_DIR"
echo "============================================="

# --- Check manifest exists ---
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Merged manifest not found: $DATA_PATH"
    echo "Run: python scripts/fix_manifest_paths.py"
    exit 1
fi

# --- Check pretrained weights ---
if [ -n "$PRETRAINED" ] && [ ! -f "$PRETRAINED" ]; then
    echo "WARNING: Pretrained weights not found: $PRETRAINED"
    echo "Training from scratch instead."
    PRETRAINED=""
fi

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Common args ---
COMMON_ARGS="
    --arch $ARCH
    --patch_size $PATCH_SIZE
    --in_chans $IN_CHANS
    --fusion $FUSION
    --out_dim $OUT_DIM
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
    --warmup_teacher_temp $WARMUP_TEACHER_TEMP
    --teacher_temp $TEACHER_TEMP
    --warmup_teacher_temp_epochs $WARMUP_TEACHER_TEMP_EPOCHS
    --global_crop_size $GLOBAL_CROP_SIZE
    --local_crop_size $LOCAL_CROP_SIZE
    --local_crops_number $LOCAL_CROPS_NUMBER
    --num_clusters $NUM_CLUSTERS
    --proj_dim $PROJ_DIM
    --w_mgcl $W_MGCL
    --w_align $W_ALIGN
    --w_view $W_VIEW
    --w_bridge $W_BRIDGE
    --bridge_proj_dim $BRIDGE_PROJ_DIM
    --bridge_num_prototypes $BRIDGE_NUM_PROTOTYPES
    --bridge_temp $BRIDGE_TEMP
    --bridge_lambda_sharp $BRIDGE_LAMBDA_SHARP
    --bridge_lambda_balance $BRIDGE_LAMBDA_BALANCE
    --save_every $SAVE_EVERY
    --log_every $LOG_EVERY
    --use_fp16 $USE_FP16
    --num_workers $NUM_WORKERS
"

# Optional args
if [ -n "$PRETRAINED" ]; then
    COMMON_ARGS="$COMMON_ARGS --pretrained $PRETRAINED"
fi
if [ -n "$RESUME" ]; then
    COMMON_ARGS="$COMMON_ARGS --resume $RESUME"
fi

# --- Mode-specific args ---
case "$MODE" in
    loadtest)
        echo ""
        echo ">>> LOAD TEST: PIL 全量图像加载测试（34 数据集 × 50 张）<<<"
        echo ""
        python3 /root/autodl-tmp/train1/scripts/test_data_loading.py \
            --n_per_ds 50 --workers 32 \
            2>&1 | tee "$OUTPUT_DIR/loadtest_log.txt"
        exit $?
        ;;
    precheck)
        echo ""
        echo ">>> PRECHECK: strict loading, 500 samples dry-run <<<"
        echo ""
        MODE_ARGS="--strict_loading True --dry_run_samples 500 --max_load_fail_ratio 0"
        ;;
    train)
        echo ""
        echo ">>> TRAINING: fault-tolerant mode (max fail ratio 0.1%) <<<"
        echo ""
        MODE_ARGS="--strict_loading False --dry_run_samples 0 --max_load_fail_ratio 0.001"
        ;;
    *)
        echo "Unknown mode: $MODE. Use 'precheck' or 'train'."
        exit 1
        ;;
esac

# --- Run ---
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU with ${NUM_GPUS} GPUs"
    torchrun --nproc_per_node=$NUM_GPUS main_dino_rgbir.py \
        $COMMON_ARGS $MODE_ARGS \
        2>&1 | tee "$OUTPUT_DIR/${MODE}_log.txt"
else
    echo "Single-GPU"
    python main_dino_rgbir.py \
        $COMMON_ARGS $MODE_ARGS \
        2>&1 | tee "$OUTPUT_DIR/${MODE}_log.txt"
fi

echo ""
if [ "$MODE" = "precheck" ]; then
    echo "Precheck complete! Review output above."
    echo "If all OK, run: bash scripts/run_pretrain_multi.sh train"
else
    echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
fi

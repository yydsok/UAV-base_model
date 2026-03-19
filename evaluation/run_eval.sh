#!/bin/bash
# =============================================================================
# DINO-MM Evaluation Suite — One-Click Runner
#
# Runs all evaluation scripts in priority order.
# Each script is self-contained and can also be run independently.
#
# Usage:
#   bash evaluation/run_eval.sh /path/to/checkpoint.pth [gpu_id]
#
# Examples:
#   bash evaluation/run_eval.sh checkpoints/checkpoint_ep50.pth
#   bash evaluation/run_eval.sh checkpoints/checkpoint_ep50.pth 0
#   SKIP_LINEAR=1 bash evaluation/run_eval.sh checkpoints/checkpoint.pth
#
# Environment variables to skip specific evaluations:
#   SKIP_RANKME=1       Skip RankMe evaluation
#   SKIP_ATTENTION=1    Skip attention map visualization
#   SKIP_KNN=1          Skip KNN classification
#   SKIP_RETRIEVAL=1    Skip cross-modal retrieval
#   SKIP_TSNE=1         Skip t-SNE visualization
#   SKIP_LINEAR=1       Skip linear probing (slowest)
#   SKIP_DET=1          Skip downstream detection
# =============================================================================

set -e

# --- Arguments ---
CKPT="${1:?Usage: bash run_eval.sh /path/to/checkpoint.pth [gpu_id]}"
GPU_ID="${2:-0}"

# --- Setup ---
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$EVAL_DIR/.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=============================================="
echo "  DINO-MM Evaluation Suite"
echo "=============================================="
echo "  Checkpoint: $CKPT"
echo "  GPU:        $GPU_ID"
echo "  Eval dir:   $EVAL_DIR"
echo "=============================================="

# Check checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# Track results
RESULTS_LOG="$EVAL_DIR/outputs/eval_results.log"
mkdir -p "$EVAL_DIR/outputs"
echo "DINO-MM Evaluation Results — $(date)" > "$RESULTS_LOG"
echo "Checkpoint: $CKPT" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"

run_eval() {
    local name="$1"
    local script="$2"
    shift 2
    local skip_var="SKIP_$(echo "$name" | tr '[:lower:]' '[:upper:]')"

    if [ "${!skip_var}" = "1" ]; then
        echo ""
        echo "[SKIP] $name (${skip_var}=1)"
        echo "[$name] SKIPPED" >> "$RESULTS_LOG"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "  Running: $name"
    echo "=============================================="

    local start_time=$(date +%s)

    if python "$script" "$@" 2>&1 | tee -a "$RESULTS_LOG"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        echo "  [$name] DONE in ${duration}s"
        echo "[$name] PASSED (${duration}s)" >> "$RESULTS_LOG"
    else
        echo ""
        echo "  [$name] FAILED"
        echo "[$name] FAILED" >> "$RESULTS_LOG"
    fi
    echo "" >> "$RESULTS_LOG"
}

# =============================================================================
# Priority 1: RankMe (fastest, no labels needed)
# =============================================================================
run_eval "rankme" "$EVAL_DIR/eval_rankme.py" \
    --checkpoint "$CKPT" \
    --num_samples 2000 \
    --random_baseline

# =============================================================================
# Priority 2: Attention Maps (no labels needed, visual)
# =============================================================================
run_eval "attention" "$EVAL_DIR/eval_attention_map.py" \
    --checkpoint "$CKPT" \
    --num_images 5

# =============================================================================
# Priority 3: KNN Classification
# =============================================================================
run_eval "knn" "$EVAL_DIR/eval_knn.py" \
    --checkpoint "$CKPT" \
    --k 10 20 \
    --random_baseline

# =============================================================================
# Priority 4: Cross-Modal Retrieval (DroneVehicle)
# =============================================================================
run_eval "retrieval" "$EVAL_DIR/eval_crossmodal_retrieval.py" \
    --checkpoint "$CKPT" \
    --dataset dronevehicle \
    --random_baseline

# =============================================================================
# Priority 5: t-SNE Visualization
# =============================================================================
run_eval "tsne" "$EVAL_DIR/eval_tsne.py" \
    --checkpoint "$CKPT" \
    --num_samples 1000

# =============================================================================
# Priority 6: Linear Probing (slowest)
# =============================================================================
run_eval "linear" "$EVAL_DIR/eval_linear_probe.py" \
    --checkpoint "$CKPT" \
    --epochs 100

# =============================================================================
# Priority 7: Downstream Detection (crop-based)
# =============================================================================
run_eval "det" "$EVAL_DIR/eval_downstream_det.py" \
    --checkpoint "$CKPT" \
    --random_baseline

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo "  All evaluations complete!"
echo "  Results log: $RESULTS_LOG"
echo "  Output dir:  $EVAL_DIR/outputs/"
echo "=============================================="

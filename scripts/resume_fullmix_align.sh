#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/autodl-fs/data/DINO-MM"
OUTPUT_DIR="/root/autodl-tmp/train1/fullmix_bs736_vb4_i1_20260313_083809"
MANIFEST="/root/autodl-tmp/train1/manifests/merged_full_pretrain_v2_plus_video_sequences_subset10pct.json"
CKPT="${OUTPUT_DIR}/checkpoint_latest.pth"
LOG="${OUTPUT_DIR}/console.log"

mkdir -p "${OUTPUT_DIR}"
exec >>"${LOG}" 2>&1

echo
echo "[resume-align] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[resume-align] cwd=${ROOT_DIR}"
echo "[resume-align] checkpoint=${CKPT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT_DIR}"

/root/miniconda3/bin/python -u main_dino_rgbir.py \
  --arch vit_small \
  --patch_size 16 \
  --in_chans 4 \
  --fusion gated_cross_attn \
  --data_path "${MANIFEST}" \
  --video_manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${CKPT}" \
  --epochs 40 \
  --warmup_epochs 0 \
  --lr 0.0002 \
  --batch_size_per_gpu 480 \
  --video_batch_size 4 \
  --num_frames 4 \
  --video_step_interval 1 \
  --num_workers 8 \
  --video_num_workers 4 \
  --prefetch_factor 1 \
  --video_prefetch_factor 1 \
  --persistent_workers true \
  --video_persistent_workers true \
  --use_gradient_checkpointing true \
  --use_pcgrad true \
  --adaptive_weighting false \
  --use_fp16 true \
  --use_view_aug true \
  --w_mgcl 0.05 \
  --w_view 0.08 \
  --w_bridge 0.08 \
  --w_dino_video 0.06 \
  --w_tcl 0.05 \
  --w_tcl_patch 0.08 \
  --w_align_rgbir 0.1 \
  --w_align_rgbir_patch 0.05 \
  --align_rgbir_queue_size 4096 \
  --align_hard_neg_topk 128 \
  --align_hard_neg_weight 2.0 \
  --align_temp_start 0.2 \
  --align_temp_end 0.05 \
  --align_temp_warmup_epochs 10 \
  --align_ramp_epochs 5 \
  --use_grayscale_bridge true \
  --w_gray_bridge 0.01 \
  --type_sampling_weights 0.6 0.35 0.05

status=$?
echo "[resume-align] python_exit=${status} at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
exit "${status}"

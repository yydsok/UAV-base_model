#!/bin/bash
# Resume training from epoch 37 with new balanced manifest + alignment offsets
# Using A800 80GB, single GPU

set -e

CKPT="/root/autodl-tmp/train1/fullmix_bs736_vb4_i1_20260313_083809/checkpoint_latest.pth"
MANIFEST="/root/autodl-tmp/train1/manifests/combined_pretrain_v3_10pct_balanced.json"
ALIGNMENT="/root/autodl-tmp/train1/manifests/alignment_offsets.json"
OUTPUT_DIR="/root/autodl-tmp/train1/fullmix_bs736_vb4_i1_20260313_083809"

cd /autodl-fs/data/DINO-MM

python -u main_dino_rgbir.py \
    --arch vit_small \
    --patch_size 16 \
    --in_chans 4 \
    --fusion gated_cross_attn \
    --data_path "${MANIFEST}" \
    --data_mode file \
    --video_manifest "${MANIFEST}" \
    --alignment_offsets "${ALIGNMENT}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume "${CKPT}" \
    --epochs 50 \
    --warmup_epochs 0 \
    --lr 0.0002 \
    --min_lr 1e-06 \
    --batch_size_per_gpu 480 \
    --video_batch_size 4 \
    --num_frames 4 \
    --video_step_interval 1 \
    --num_workers 12 \
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
    --save_every 5 \
    --log_every 50 \
    --w_mgcl 0.05 \
    --w_view 0.08 \
    --w_bridge 0.08 \
    --w_dino_video 0.06 \
    --w_tcl 0.05 \
    --w_tcl_patch 0.08 \
    --w_align_rgbir 0.03 \
    --w_align_rgbir_patch 0.01 \
    --type_sampling_weights 0.6 0.35 0.05 \
    2>&1 | tee -a "${OUTPUT_DIR}/console_resume_aligned.log"

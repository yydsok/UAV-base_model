#!/bin/bash
# ==============================================================================
# DINO-MM 下游任务一键训练脚本
# 对标 RingMo-Aerial 论文 baseline
# 使用 skysence 环境 (mmdet 2.28.2 / mmseg 0.30.0 / mmrotate 0.3.4 / mmtrack 0.14.0)
# ==============================================================================

set -e

# ---- 路径配置 ----
DINO_MM_ROOT="/autodl-fs/data/DINO-MM"
DOWNSTREAM="${DINO_MM_ROOT}/downstream"
CKPT="${DINO_MM_ROOT}/runs/fullmix_resume_bs640_20260314/checkpoint_latest.pth"
DATA_ROOT="/root/autodl-tmp/data"
RESULTS="${DOWNSTREAM}/results"

# ---- 通用参数 ----
DEVICE="cuda"
SEED=42
NUM_WORKERS=4

mkdir -p "${RESULTS}"

echo "============================================"
echo "DINO-MM Downstream Baseline Training"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RESULTS}"
echo "============================================"

# ==============================================================================
# 1. 语义分割 - UAVid (RingMo-Aerial Tab.1, SOTA mIoU=78.7)
# ==============================================================================
run_segmentation_uavid() {
    echo "[1/10] Segmentation: UAVid + UPerNet"
    python "${DOWNSTREAM}/train_segmentation.py" \
        --dataset uavid \
        --data_root "${DATA_ROOT}/UAVid" \
        --checkpoint "${CKPT}" \
        --framework upernet \
        --epochs 80 \
        --batch_size 4 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --image_size 512 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/seg_uavid_upernet" \
        2>&1 | tee "${RESULTS}/seg_uavid_upernet/train.log"
}

# ==============================================================================
# 2. 语义分割 - UDD6 (RingMo-Aerial Tab.2, SOTA mIoU=78.9)
# ==============================================================================
run_segmentation_udd6() {
    echo "[2/10] Segmentation: UDD6 + UPerNet"
    python "${DOWNSTREAM}/train_segmentation.py" \
        --dataset udd6 \
        --data_root "${DATA_ROOT}/UDD/UDD" \
        --checkpoint "${CKPT}" \
        --framework upernet \
        --epochs 80 \
        --batch_size 4 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --image_size 512 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/seg_udd6_upernet" \
        2>&1 | tee "${RESULTS}/seg_udd6_upernet/train.log"
}

# ==============================================================================
# 3. 检测 - VisDrone-DET + Cascade R-CNN (RingMo-Aerial Tab.4, SOTA mAP=38.6)
# ==============================================================================
run_detection_visdrone() {
    echo "[3/10] Detection: VisDrone-DET + Cascade R-CNN"
    python "${DOWNSTREAM}/train_detection.py" \
        --dataset visdrone_det \
        --data_root "${DATA_ROOT}/VisDrone/Object Detection in Images" \
        --checkpoint "${CKPT}" \
        --framework cascade_rcnn \
        --modality rgb_only \
        --epochs 12 \
        --batch_size 2 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --trainable_blocks 4 \
        --feature_dim 256 \
        --min_size 640 \
        --max_size 1333 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/det_visdrone_cascade" \
        2>&1 | tee "${RESULTS}/det_visdrone_cascade/train.log"
}

# ==============================================================================
# 4. 检测 - UAVDT + Cascade R-CNN (RingMo-Aerial Tab.5, SOTA mAP=25.2)
# ==============================================================================
run_detection_uavdt() {
    echo "[4/10] Detection: UAVDT + Cascade R-CNN"
    python "${DOWNSTREAM}/train_detection.py" \
        --dataset uavdt \
        --data_root "${DATA_ROOT}/UAVDT" \
        --checkpoint "${CKPT}" \
        --framework cascade_rcnn \
        --modality rgb_only \
        --epochs 12 \
        --batch_size 2 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --trainable_blocks 4 \
        --feature_dim 256 \
        --min_size 640 \
        --max_size 1333 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/det_uavdt_cascade" \
        2>&1 | tee "${RESULTS}/det_uavdt_cascade/train.log"
}

# ==============================================================================
# 5. 检测 - DroneVehicle RGB+IR + Cascade R-CNN (RingMo-Aerial Tab.7, SOTA mAP50=71.6)
# ==============================================================================
run_detection_dronevehicle() {
    echo "[5/10] Detection: DroneVehicle RGB+IR + Cascade R-CNN"
    python "${DOWNSTREAM}/train_detection.py" \
        --dataset dronevehicle \
        --data_root "${DATA_ROOT}/DroneVehicle" \
        --checkpoint "${CKPT}" \
        --framework cascade_rcnn \
        --modality both \
        --epochs 12 \
        --batch_size 2 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --trainable_blocks 4 \
        --feature_dim 256 \
        --min_size 640 \
        --max_size 1333 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --annotation_source rgb \
        --output_dir "${RESULTS}/det_dronevehicle_cascade_rgbir" \
        2>&1 | tee "${RESULTS}/det_dronevehicle_cascade_rgbir/train.log"
}

# ==============================================================================
# 6. 场景分类 - AID (RingMo-Aerial Tab.9, SOTA OA=95.81/96.46)
# ==============================================================================
run_classification_aid() {
    if [ ! -d "${DATA_ROOT}/AID" ]; then
        echo "[SKIP] AID dataset not found at ${DATA_ROOT}/AID"
        return
    fi
    echo "[6/10] Classification: AID + Linear Probe"
    python "${DOWNSTREAM}/train_classification.py" \
        --dataset aid \
        --data_root "${DATA_ROOT}/AID" \
        --checkpoint "${CKPT}" \
        --epochs 200 \
        --batch_size 64 \
        --lr 6e-5 \
        --train_ratio 0.2 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/cls_aid_tr20" \
        2>&1 | tee "${RESULTS}/cls_aid_tr20/train.log"
}

# ==============================================================================
# 7. 变化检测 - LEVIR-CD (RingMo-Aerial Tab.10, SOTA F1=92.71)
# ==============================================================================
run_change_detection_levir() {
    if [ ! -d "${DATA_ROOT}/LEVIR-CD" ]; then
        echo "[SKIP] LEVIR-CD dataset not found at ${DATA_ROOT}/LEVIR-CD"
        return
    fi
    echo "[7/10] Change Detection: LEVIR-CD + BIT-style"
    python "${DOWNSTREAM}/train_change_detection.py" \
        --dataset levir_cd \
        --data_root "${DATA_ROOT}/LEVIR-CD" \
        --checkpoint "${CKPT}" \
        --epochs 200 \
        --batch_size 8 \
        --lr 6e-5 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/cd_levir_bit" \
        2>&1 | tee "${RESULTS}/cd_levir_bit/train.log"
}

# ==============================================================================
# 8. 旋转检测 - DIOR-R + Oriented R-CNN
# ==============================================================================
run_oriented_detection_dior() {
    if [ ! -d "${DATA_ROOT}/DIOR-R" ]; then
        echo "[SKIP] DIOR-R dataset not found at ${DATA_ROOT}/DIOR-R"
        return
    fi
    echo "[8/10] Oriented Detection: DIOR-R + Oriented R-CNN"
    python "${DOWNSTREAM}/train_oriented_detection.py" \
        --dataset dior_r \
        --data_root "${DATA_ROOT}/DIOR-R" \
        --checkpoint "${CKPT}" \
        --angle_version le90 \
        --epochs 12 \
        --batch_size 2 \
        --lr 8e-5 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/odet_dior_r_oriented" \
        2>&1 | tee "${RESULTS}/odet_dior_r_oriented/train.log"
}

# ==============================================================================
# 9. 检索 - DroneVehicle 跨模态检索
# ==============================================================================
run_retrieval_dronevehicle() {
    echo "[9/10] Retrieval: DroneVehicle cross-modal"
    python "${DOWNSTREAM}/train_retrieval.py" \
        --dataset dronevehicle \
        --data_root "${DATA_ROOT}/DroneVehicle" \
        --checkpoint "${CKPT}" \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --output_dir "${RESULTS}/ret_dronevehicle" \
        2>&1 | tee "${RESULTS}/ret_dronevehicle/train.log"
}

# ==============================================================================
# 10. 跟踪 - VisDrone-MOT (RingMo-Aerial Tab.12, SOTA MOTA=52.6)
# 注意: 跟踪需要先训练检测器，然后用检测结果做 MOT
# ==============================================================================
run_tracking_visdrone() {
    local DET_CKPT="${RESULTS}/det_visdrone_cascade/checkpoint_best.pth"
    if [ ! -f "${DET_CKPT}" ]; then
        echo "[SKIP] VisDrone-MOT tracking requires VisDrone-DET detector first"
        return
    fi
    echo "[10/10] Tracking: VisDrone-MOT + ByteTrack"
    python "${DOWNSTREAM}/eval_tracking.py" \
        --dataset visdrone_mot \
        --data_root "${DATA_ROOT}/VisDrone/Multi-Object Tracking" \
        --detector_checkpoint "${DET_CKPT}" \
        --tracker mm_bytetrack \
        --device ${DEVICE} \
        --score_thresh 0.3 \
        2>&1 | tee "${RESULTS}/track_visdrone_mot/eval.log"
}

# ==============================================================================
# 执行入口
# ==============================================================================
echo ""
echo "Available tasks:"
echo "  all       - Run all tasks sequentially"
echo "  seg       - Segmentation (UAVid + UDD6)"
echo "  det       - Detection (VisDrone + UAVDT + DroneVehicle)"
echo "  cls       - Classification (AID)"
echo "  cd        - Change Detection (LEVIR-CD)"
echo "  odet      - Oriented Detection (DIOR-R)"
echo "  ret       - Retrieval (DroneVehicle)"
echo "  track     - Tracking (VisDrone-MOT)"
echo ""

TASK="${1:-all}"

case "$TASK" in
    seg)
        run_segmentation_uavid
        run_segmentation_udd6
        ;;
    det)
        run_detection_visdrone
        run_detection_uavdt
        run_detection_dronevehicle
        ;;
    cls)
        run_classification_aid
        ;;
    cd)
        run_change_detection_levir
        ;;
    odet)
        run_oriented_detection_dior
        ;;
    ret)
        run_retrieval_dronevehicle
        ;;
    track)
        run_tracking_visdrone
        ;;
    all)
        run_segmentation_uavid
        run_segmentation_udd6
        run_detection_visdrone
        run_detection_uavdt
        run_detection_dronevehicle
        run_classification_aid
        run_change_detection_levir
        run_oriented_detection_dior
        run_retrieval_dronevehicle
        run_tracking_visdrone
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Usage: bash run_all_baselines.sh [all|seg|det|cls|cd|odet|ret|track]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "All requested tasks completed!"
echo "Results saved to: ${RESULTS}"
echo "============================================"

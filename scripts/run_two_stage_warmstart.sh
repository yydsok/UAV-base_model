#!/bin/bash
# =============================================================================
# DINO-MM 两阶段训练总入口
# 阶段一：纯 DINO 预热
# 阶段二：从阶段一 checkpoint 继续，小权重加回辅助损失
# =============================================================================

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo "  DINO-MM 两阶段训练"
echo "============================================================"
echo "  阶段一：纯 DINO 预热"
echo "  阶段二：小权重加回 MGCL / View / Bridge"
echo "============================================================"

bash scripts/run_stage1_dino_only_10ep.sh

STAGE1_DIR=$(ls -td /root/autodl-tmp/train1/stage1_dino_only_10ep_* 2>/dev/null | head -n 1)
if [ -z "$STAGE1_DIR" ]; then
    echo "ERROR: 未找到第一阶段输出目录"
    exit 1
fi

STAGE1_CKPT="${STAGE1_DIR}/checkpoint_latest.pth"
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: 未找到第一阶段 checkpoint: $STAGE1_CKPT"
    exit 1
fi

echo "阶段一完成，进入阶段二"
echo "使用 checkpoint: $STAGE1_CKPT"

STAGE1_CKPT="$STAGE1_CKPT" bash scripts/run_stage2_multiloss_from_stage1.sh

#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/autodl-fs/data/baselines"
mkdir -p "$BASE_DIR"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo >/dev/null 2>&1 || true
fi

clone_if_missing() {
  local name="$1"
  local url="$2"
  if [[ -d "$BASE_DIR/$name/.git" ]]; then
    echo "[skip] $name"
  else
    echo "[clone] $name <- $url"
    git clone "$url" "$BASE_DIR/$name"
  fi
}

# RingMo-Aerial 主任务相关 baseline（任务优先，而非穷举全部论文方法）
clone_if_missing mmdetection_v2 https://github.com/open-mmlab/mmdetection.git
clone_if_missing mmsegmentation https://github.com/open-mmlab/mmsegmentation.git
clone_if_missing mmrotate https://github.com/open-mmlab/mmrotate.git
clone_if_missing mmtracking https://github.com/open-mmlab/mmtracking.git
clone_if_missing mmpretrain https://github.com/open-mmlab/mmpretrain.git
clone_if_missing ByteTrack https://github.com/ifzhang/ByteTrack.git
clone_if_missing BIT_CD https://github.com/justchenhao/BIT_CD.git
clone_if_missing open-cd https://github.com/likyoo/open-cd.git
clone_if_missing changeformer https://github.com/wgcban/ChangeFormer.git
clone_if_missing geoseg https://github.com/WangLibo1995/GeoSeg.git
clone_if_missing cas_mvsnet https://github.com/alibaba/cascade-stereo.git
clone_if_missing patchmatchnet https://github.com/FangjinhuaWang/PatchmatchNet.git
clone_if_missing fastmvsnet https://github.com/svip-lab/FastMVSNet.git
clone_if_missing mvsnet https://github.com/YoYo000/MVSNet.git
clone_if_missing motr https://github.com/megvii-research/MOTR.git
clone_if_missing trackformer https://github.com/timmeinhardt/trackformer.git
clone_if_missing uavmot https://github.com/LiuShuaiyr/UAVMOT.git
clone_if_missing u2mot https://github.com/alibaba/u2mot.git
clone_if_missing oc_sort https://github.com/noahcao/OC_SORT.git
clone_if_missing strongsort https://github.com/dyhBUPT/StrongSORT.git
clone_if_missing fairmot https://github.com/ifzhang/FairMOT.git
clone_if_missing centertrack https://github.com/xingyizhou/CenterTrack.git
clone_if_missing tgram https://github.com/HeQibin/TGraM.git

cat <<'MSG'

[done] RingMo-Aerial 任务基线仓库同步完成。
建议后续环境：
  1) skysence: mmdet/mmseg/mmtrack/mmrotate + downstream 主链路
  2) ringmo_mvs: cas_mvsnet/patchmatchnet/fastmvsnet/mvsnet
  3) ringmo_mot_ext: motr/trackformer/u2mot/strongsort 等
MSG

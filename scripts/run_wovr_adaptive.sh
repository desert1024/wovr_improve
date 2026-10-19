#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /data/wovr/code/WoVR

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "1p")
GPU_COUNT=$(nvidia-smi -L | wc -l)
RAM_GB=$(free -g | awk "/^Mem:/{print \$2}")
echo "[adaptive] gpu=$GPU_NAME count=$GPU_COUNT ram=${RAM_GB}G"

if echo "$GPU_NAME" | grep -qiE "L20|A800|A100"; then
    # Keep identical optimization settings across GPU types so that
    # preemption/resume does not change the effective training config.
    OVERRIDES="actor.enable_offload=True rollout.enable_offload=True env.train.enable_offload=True"
    OVERRIDES+=" actor.micro_batch_size=16 env.train.total_num_envs=64"
    OVERRIDES+=" env.train.rollout_epoch=16 env.eval.total_num_envs=32"
else
    echo "[adaptive] unsupported GPU: $GPU_NAME" >&2
    exit 2
fi

echo "[adaptive] overrides: $OVERRIDES"
export HYDRA_OVERRIDES="$OVERRIDES"
exec bash "$SCRIPT_DIR/run_wovr.sh"

#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /data/wovr/code/WoVR

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "1p")
GPU_COUNT=$(nvidia-smi -L | wc -l)
RAM_GB=$(free -g | awk "/^Mem:/{print \$2}")
echo "[adaptive] gpu=$GPU_NAME count=$GPU_COUNT ram=${RAM_GB}G"

if echo "$GPU_NAME" | grep -qi "L20"; then
    OFFLOAD="actor.enable_offload=True rollout.enable_offload=True env.train.enable_offload=True"
    MB=16
    if [ "$RAM_GB" -lt 600 ]; then RE=8; else RE=16; fi
elif echo "$GPU_NAME" | grep -qiE "A800|A100"; then
    # 80GB cards: keep actor/rollout resident, only offload the world model
    OFFLOAD="actor.enable_offload=False rollout.enable_offload=False env.train.enable_offload=True"
    MB=16
    if [ "$RAM_GB" -lt 600 ]; then RE=8; else RE=16; fi
else
    echo "[adaptive] unsupported GPU: $GPU_NAME" >&2
    exit 2
fi

OVERRIDES="$OFFLOAD actor.micro_batch_size=$MB env.train.total_num_envs=64 env.train.rollout_epoch=$RE env.eval.total_num_envs=32"

echo "[adaptive] overrides: $OVERRIDES"
export HYDRA_OVERRIDES="$OVERRIDES"
exec bash "$SCRIPT_DIR/run_wovr.sh"

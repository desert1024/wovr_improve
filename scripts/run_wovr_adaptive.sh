#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /data/wovr/code/WoVR

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "1p")
GPU_COUNT=$(nvidia-smi -L | wc -l)
RAM_GB=$(free -g | awk "/^Mem:/{print \$2}")
echo "[adaptive] gpu=$GPU_NAME count=$GPU_COUNT ram=${RAM_GB}G"

if echo "$GPU_NAME" | grep -qi "L20"; then
    # 48GB cards: offload everything, smaller batch
    RE=16
    MB=16
    OVERRIDES="actor.enable_offload=True rollout.enable_offload=True env.train.enable_offload=True"
    OVERRIDES+=" actor.micro_batch_size=${MB} env.train.total_num_envs=64"
    OVERRIDES+=" env.train.rollout_epoch=${RE} env.eval.total_num_envs=32"
elif echo "$GPU_NAME" | grep -qiE "A800|A100"; then
    # 80GB cards: actor/rollout resident, env offloaded
    OVERRIDES="actor.enable_offload=False rollout.enable_offload=False env.train.enable_offload=True"
    OVERRIDES+=" actor.micro_batch_size=32 env.train.total_num_envs=64"
    OVERRIDES+=" env.train.rollout_epoch=16 env.eval.total_num_envs=32"
else
    echo "[adaptive] unsupported GPU: $GPU_NAME" >&2
    exit 2
fi

echo "[adaptive] overrides: $OVERRIDES"
export HYDRA_OVERRIDES="$OVERRIDES"
exec bash "$SCRIPT_DIR/run_wovr.sh"

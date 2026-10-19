#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail
cd /data/wovr/code/WoVR

export PATH=/data/wovr/venv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export EMBODIED_PATH=/data/wovr/code/WoVR/examples/embodiment
export PYTHONPATH=/data/wovr/code/WoVR:${PYTHONPATH:-}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
rm -rf "$HOME/.cache/huggingface/modules/transformers_modules"

CKPT=${CKPT:-/data/wovr/code/WoVR/logs/20261018-21:40:25-wan_libero_spatial_grpo_openvlaoft/wan_libero_spatial_grpo_openvlaoft/checkpoints/global_step_20}
HF=${HF:-/data/wovr/checkpoints/WoVR-step20-hf}
OUT=${OUT:-/data/wovr/eval_results/step20}
ENVS=${ENVS:-32}

if [ ! -f "$HF/model.safetensors" ]; then
    python $SCRIPT_DIR/convert_ckpt_to_hf.py --ckpt "$CKPT" --out "$HF"
fi

python evaluations/eval_embodied_agent.py \
    --config-name libero_spatial_openvlaoft_eval \
    rollout.model.model_path="$HF" \
    env.eval.total_num_envs="$ENVS" \
    runner.logger.log_path="$OUT"

echo "[eval_video] videos are under $OUT/video/eval"

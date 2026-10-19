#!/bin/bash
set -euo pipefail
cd /data/wovr/code/WoVR
rm -rf "$HOME/.cache/huggingface/modules/transformers_modules"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

if [ -f /data/wovr/venv/bin/activate ]; then
    source /data/wovr/venv/bin/activate
elif command -v switch_env >/dev/null 2>&1; then
    source switch_env openvla-oft
fi

echo "[run_wovr] host=$(hostname) python=$(command -v python)"
nvidia-smi -L || true

bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft

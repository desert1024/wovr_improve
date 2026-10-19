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

export HF_HOME=/data/wovr/hf_cache
mkdir -p "$HF_HOME"
rm -rf "$HF_HOME/modules/transformers_modules"
python - <<"PYWARM"
from transformers import AutoProcessor
p = "/data/wovr/code/WoVR/checkpoints/Openvla-oft-SFT-libero-spatial-traj1"
AutoProcessor.from_pretrained(p, trust_remote_code=True)
print("[run_wovr] processor cache warmed")
PYWARM

echo "[run_wovr] host=$(hostname) python=$(command -v python)"
nvidia-smi -L || true

bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft

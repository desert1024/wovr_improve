#!/bin/bash
set -euo pipefail
cd /data/wovr/code/WoVR
export PATH="/home/wuqianqian/.local/bin:$PATH"
bash requirements/install.sh embodied --model openvla-oft --env wan \
  --venv /data/wovr/venv --use-mirror

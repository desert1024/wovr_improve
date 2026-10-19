#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec > /data/wovr/logs/eval_video.log 2>&1
export ENVS=32
bash $SCRIPT_DIR/eval_video.sh

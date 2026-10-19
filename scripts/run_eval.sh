#!/bin/bash
exec > /data/wovr/logs/eval_video.log 2>&1
export ENVS=32
bash /data/wovr/eval_video.sh

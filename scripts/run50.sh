#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec > /data/wovr/logs/run50.log 2>&1
export STEPS=50
bash "$SCRIPT_DIR/run_wovr.sh"

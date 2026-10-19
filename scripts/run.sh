#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STEPS=${STEPS:-20}
bash $SCRIPT_DIR/run_wovr.sh

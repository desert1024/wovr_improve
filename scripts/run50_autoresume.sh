#!/bin/bash
set -uo pipefail
cd /data/wovr/code/WoVR
TOTAL=${TOTAL:-50}
LOGDIR=/data/wovr/logs
CMD=$LOGDIR/run50_autoresume_cmd.sh
mkdir -p "$LOGDIR"

latest_ckpt() {
    find logs -maxdepth 5 -type d -path "*/checkpoints/global_step_*" 2>/dev/null \
        | awk -F"global_step_" "{print \$2 \"\t\" \$0}" | sort -n | tail -1 | cut -f2-
}

while true; do
    CKPT=$(latest_ckpt)
    STEP=0
    if [ -n "$CKPT" ]; then
        CKPT=$(realpath "$CKPT")
        STEP=$(basename "$CKPT" | sed "s/global_step_//")
    fi
    if [ "$STEP" -ge "$TOTAL" ]; then
        echo "[auto] reached step $STEP, done"
        break
    fi
    echo "[auto] $(date) starting from step $STEP"
    {
        echo "#!/bin/bash"
        echo "export STEPS=$TOTAL"
        [ -n "$CKPT" ] && echo "export RESUME=$CKPT"
        echo "exec bash /data/wovr/code/WoVR/scripts/run_wovr.sh"
    } > "$CMD"
    chmod +x "$CMD"

    OLD=$(find logs -maxdepth 5 -type d -path "*/checkpoints/global_step_*" 2>/dev/null \
          | awk -F"global_step_" "{print \$2 \"\t\" \$0}" | sort -n | head -n -2 | cut -f2-)
    if [ -n "$OLD" ]; then
        echo "[auto] removing old checkpoints:"
        echo "$OLD"
        echo "$OLD" | xargs -r rm -rf
    fi

    bash scripts/start.sh bash "$CMD"
    echo "[auto] $(date) worker exited, relaunching in 10s"
    sleep 10
done

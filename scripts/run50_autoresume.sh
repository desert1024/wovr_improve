#!/bin/bash
set -uo pipefail
cd /data/wovr/code/WoVR

TOTAL=${TOTAL:-50}
HEARTBEAT_TIMEOUT=${HEARTBEAT_TIMEOUT:-1200}   # 20 minutes without log growth
LOGDIR=/data/wovr/logs
CMD=$LOGDIR/run50_autoresume_cmd.sh
mkdir -p "$LOGDIR"

newest_log() {
    find logs -name run_embodiment.log -printf "%T@ %p\n" 2>/dev/null \
        | sort -n | tail -1 | cut -d" " -f2-
}

complete_ckpts_desc() {
    find logs -maxdepth 5 -type d -path "*/checkpoints/global_step_*" 2>/dev/null \
        | awk -F"global_step_" "{print \$2 \"\t\" \$0}" | sort -rn | cut -f2-
}

latest_complete_ckpt() {
    local d
    while read -r d; do
        [ -z "$d" ] && continue
        if [ -f "$d/actor/dcp_checkpoint/.metadata" ] && ls "$d/actor/dcp_checkpoint/"*.distcp >/dev/null 2>&1; then
            echo "$d"
            return 0
        fi
    done < <(complete_ckpts_desc)
}

kill_tree() {
    local pid=$1
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
}

while true; do
    CKPT=$(latest_complete_ckpt)
    STEP=0
    if [ -n "$CKPT" ]; then
        CKPT=$(realpath "$CKPT")
        STEP=$(basename "$CKPT" | sed "s/global_step_//")
    fi
    if [ "$STEP" -ge "$TOTAL" ]; then
        echo "[auto] $(date) reached step $STEP, done"
        break
    fi

    echo "[auto] $(date) starting from step $STEP (ckpt=${CKPT:-none})"
    {
        echo "#!/bin/bash"
        echo "export STEPS=$TOTAL"
        [ -n "$CKPT" ] && echo "export RESUME=$CKPT"
        echo "exec bash /data/wovr/code/WoVR/scripts/run_wovr.sh"
    } > "$CMD"
    chmod +x "$CMD"

    # keep only the two newest complete checkpoints
    complete_ckpts_desc | tail -n +3 | while read -r old; do
        [ -n "$old" ] && echo "[auto] removing old checkpoint $old" && rm -rf "$old"
    done

    LOG=$(newest_log)
    LAST_MTIME=$( [ -n "$LOG" ] && stat -c %Y "$LOG" 2>/dev/null || echo 0 )
    STALL=0

    bash scripts/start.sh bash "$CMD" &
    RLPID=$!
    echo "[auto] $(date) launched rlaunch pid=$RLPID"

    while kill -0 "$RLPID" 2>/dev/null; do
        sleep 60
        LOG=$(newest_log)
        M=0
        [ -n "$LOG" ] && M=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
        if [ "$M" -gt "$LAST_MTIME" ]; then
            LAST_MTIME=$M
            STALL=0
        else
            STALL=$((STALL + 60))
            echo "[auto] $(date) no log growth for ${STALL}s"
        fi
        if [ "$STALL" -ge "$HEARTBEAT_TIMEOUT" ]; then
            echo "[auto] $(date) heartbeat timeout, killing rlaunch pid=$RLPID"
            kill_tree "$RLPID"
            break
        fi
    done

    echo "[auto] $(date) worker/launcher exited; relaunching in 10s"
    sleep 10
done

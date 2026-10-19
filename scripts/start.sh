#!/bin/bash
set -euo pipefail

# Usage:
#   bash /data/wovr/start.sh                       # request a GPU worker (workspace rootfs, no image)
#   bash /data/wovr/start.sh -d                    # detached worker
#   bash /data/wovr/start.sh bash /path/to/script  # custom command
#
# Env overrides:
#   GPU=8 CPU=100 MEM_GB=1000 PREEMPTIBLE=yes
#   NODE=<node fqdn>   optional: pin to a node
#   IMAGE=<ref>        optional: switch back to image mode

cd /data/wovr/code/WoVR

DETACH_FLAG=()
if [ "${1:-}" = "-d" ]; then
    DETACH_FLAG=(-d)
    shift
fi
if [ "$#" -eq 0 ]; then
    set -- bash
fi

NODE=${NODE:-}
GPU_TAG=${GPU_TAG-a800}
TAG_ARGS=()
[ -n "${GPU_TAG:-}" ] && TAG_ARGS=(--positive-tags="${GPU_TAG}")
POSITIVE_TAGS=()
if [ -n "${NODE}" ]; then
    POSITIVE_TAGS=(--positive-tags="node/${NODE}")
fi

IMAGE_ARGS=()
if [ -n "${IMAGE:-}" ]; then
    IMAGE_ARGS=(--image="$IMAGE" --image-check=false --image-pull-policy=IfNotPresent)
fi

echo "[start] mode: ${IMAGE:-workspace rootfs (no image)}"
echo "[start] command: $*"

exec rlaunch "${DETACH_FLAG[@]}" \
    --gpu="${GPU:-8}" \
    --cpu="${CPU:-60}" \
    --memory=$((1024 * ${MEM_GB:-400})) \
    --preemptible "${PREEMPTIBLE:-yes}" \
    ${TAG_ARGS[@]+"${TAG_ARGS[@]}"} \
    ${POSITIVE_TAGS[@]+"${POSITIVE_TAGS[@]}"} \
    ${IMAGE_ARGS[@]+"${IMAGE_ARGS[@]}"} \
    --volume /data:/data \
    --mount=juicefs+s3://oss.i.machdrive.cn/zhengyulong-recon:/mnt/zhengyulong_recon \
    --mount=gpfs://gpfs1/acceldata/dynamic:/mnt/acceldata/dynamic \
    --mount=gpfs://gpfs2/dynamic2:/mnt/dynamic2 \
    --mount=gpfs://gpfs2/perceptron-cache:/mnt/cache \
    --mount=gpfs://gpfs3/dynamic3:/mnt/dynamic3 \
    --mount=gpfs://gpfs1/acceldata:/mnt/acceldata \
    --mount=gpfs://gpfs3/mdriver-cache:/mnt/mdriver-cache \
    --mount=juicefs+s3://oss.i.machdrive.cn/mdriver-juicefs:/mnt/mdriver-juicefs \
    --mount=juicefs+s3://oss.i.machdrive.cn/mdriver-v2-juicefs:/mnt/mdriver-v2-juicefs \
    --mount=juicefs+s3://oss.i.machdrive.cn/mdriver-v5-juicefs:/mnt/mdriver-v5-juicefs \
    -- "$@"

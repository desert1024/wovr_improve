#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rlaunch \
  --cpu=30 \
  --gpu=4 \
  --memory=$((1024*240)) \
  --preemptible yes \
  -n mach-generator \
  --image=rlinf/rlinf:agentic-rlinf0.4-wan \
  --image-check=false \
  --image-pull-policy=IfNotPresent \
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
  -- bash $SCRIPT_DIR/imgtest.sh

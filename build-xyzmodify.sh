#!/usr/bin/env bash
set -o pipefail
cd "$(dirname "$0")"

VER=0.3
JOBS=16
REPO=https://github.com/xyznq1/xyz-llama
REF=f095f4dd559cdb742c9b3ed71b7131855179f935
MMPROJ=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
IMG=ghcr.io/wabinab/bonsai

build () {   # $1 = APPLY_B, $2 = tag
  docker buildx build --progress=plain -f docker/Dockerfile.xyzmodify \
    --build-arg REPO=$REPO --build-arg REF=$REF \
    --build-arg CUDA_VERSION=13.5.0 --build-arg UBUNTU=26.04 \
    --build-arg JOBS=$JOBS --build-arg APPLY_B=$1 \
    --build-arg MMPROJ_FILE=$MMPROJ \
    --load -t $IMG:$2 .
}

: > xyz-build.log
{
  echo "=== VERSION: $VER ===" &&
  echo "=== BUILD A (APPLY_B=0) ==="  && build 0 xyz-mm-a-$VER  &&
  # echo "=== BUILD A+B (APPLY_B=1) ===" && build 1 xyz-mm-ab-$VER &&
  echo "=== PUSH ===" &&
  docker push $IMG:xyz-mm-a-$VER &&
  # docker push $IMG:xyz-mm-ab-$VER &&
  echo "=== ALL DONE ==="
} 2>&1 | tee -a xyz-build.log
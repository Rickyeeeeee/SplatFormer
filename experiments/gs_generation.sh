#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

# One GPU with eight scenes per microbatch and 64 scenes per optimizer update.
# GPU_IDS=${GPU_ID:-3} \
# NGPUS=1 \
# BATCH_SIZE=64 \
# GRAD_ACCUM_STEPS=1 \
# EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/$(date +%m%d)/noise_to_gs_hr32_n1000_batch64} \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# GPU_IDS=${GPU_ID:-2} \
# NGPUS=1 \
# MASTER_PORT=29522 \
# BATCH_SIZE=64 \
# GRAD_ACCUM_STEPS=1 \
# RANDOM_ROTATE=True \
# EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/$(date +%m%d)/noise_to_gs_hr32_n1000_batch64_rotate} \
# bash scripts/train-noise-to-gs-on-objaverse.sh

GPU_IDS=${GPU_ID:-1} \
NGPUS=1 \
MASTER_PORT=29523 \
BATCH_SIZE=64 \
COORDINATE_MODE=current \
GRAD_ACCUM_STEPS=1 \
RANDOM_ROTATE=True \
EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/$(date +%m%d)/noise_to_gs_hr32_n1000_batch64_no_morton} \
bash scripts/train-noise-to-gs-on-objaverse.sh

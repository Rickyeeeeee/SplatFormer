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

# GPU_IDS=${GPU_ID:-1} \
# NGPUS=1 \
# MASTER_PORT=29523 \
# BATCH_SIZE=64 \
# COORDINATE_MODE=current \
# GRAD_ACCUM_STEPS=1 \
# RANDOM_ROTATE=True \
# EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/$(date +%m%d)/noise_to_gs_hr32_n1000_batch64_no_morton} \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# Example experiment: explicitly set coupling, Fourier features, and warm-up.
# RANDOM_ROTATE=False LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
# GS_FOURIER_INPUT_FEATURES="['means', 'scales']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" \
# GPU_IDS=0 NGPUS=1 BATCH_SIZE=64 COORDINATE_MODE=morton_rank FLOW_COUPLING=morton \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=3e288ee8aced4a0797e66d53536112b1 \
# OUTPUT_DIR=/project2/ricky/outputs/noise_to_gs_morton_fourier_warmup \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# Morton coupling only: sort both noise and target Gaussians by Morton order.
# GPU_IDS=1 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 \
# COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False LR_WARMUP_STEPS=0 \
# GS_FOURIER_INPUT_FEATURES='[]' GS_FOURIER_NUM_FREQUENCIES='{}' GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# Warm-up only: compare against Morton-rank random coupling without rotation.
# GPU_IDS=2 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29523 \
# COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
# LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
# GS_FOURIER_INPUT_FEATURES='[]' GS_FOURIER_NUM_FREQUENCIES='{}' GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_warmup_2000 \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# Fourier features only: encode means with six frequencies.
# GPU_IDS=1 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 \
# COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
# LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
# GS_FOURIER_INPUT_FEATURES="['means']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6}" \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_fourier_means6 \
# bash scripts/train-noise-to-gs-on-objaverse.sh

# GPU_IDS=1 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29524 \
# COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=True \
# LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
# GS_FOURIER_INPUT_FEATURES="['means']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6}" \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_fourier_means6_rotate \
# bash scripts/train-noise-to-gs-on-objaverse.sh
# GS_FOURIER_INPUT_FEATURES="['means']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6}" \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \

# Predict clean Gaussians and train with velocity loss.
GPU_IDS=6 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29524 \
COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
MODEL_OUTPUT_TYPE=x PREDICTION_TYPE=x \
LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_x_xloss \
bash scripts/train-noise-to-gs-on-objaverse.sh

# Predict clean Gaussians and train with velocity loss.
GPU_IDS=6 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29524 \
DIT_SIZE=B \
COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
MODEL_OUTPUT_TYPE=x PREDICTION_TYPE=x \
LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_x_xloss \
bash scripts/train-noise-to-gs-on-objaverse.sh

GPU_IDS=6 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29524 \
COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
MODEL_OUTPUT_TYPE=x PREDICTION_TYPE=v \
LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_x_vloss \
bash scripts/train-noise-to-gs-on-objaverse.sh

GPU_IDS=6 NGPUS=1 BATCH_SIZE=64 GRAD_ACCUM_STEPS=1 MASTER_PORT=29524 \
COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False \
MODEL_OUTPUT_TYPE=v PREDICTION_TYPE=v \
LR_WARMUP_STEPS=2000 LR_WARMUP_START_FACTOR=0.03333333333333333 \
GS_FEATURE_LOSS_WEIGHTS='{}' EVAL_SCENE_NAME=${EVAL_SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/outputs/$(date +%m%d)}/noise_to_gs_morton_v_vloss \
bash scripts/train-noise-to-gs-on-objaverse.sh

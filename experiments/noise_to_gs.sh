#!/bin/bash

# Example: FLOW_COUPLING=morton GS_COORDINATE_MODE=morton_rank \
# GS_FOURIER_INPUT_FEATURES="['means', 'scales']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" \
# bash scripts/overfit-noise-to-gs.sh
set -euo pipefail
export GPU_ID=3

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

run_date=$(date +%m%d)

# Use scaled images with MCMC statistics and 32-resolution GS checkpoints.
data_root=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}


# GPU_ID=0 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# GPU_ID=4 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# RANDOM_ROTATE=True \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# GPU_ID=3 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# GPU_ID=3 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=current \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# RANDOM_ROTATE=True \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_COUNT=4000 \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_COUNT=4000 \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=current \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-8000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-8000} \
# GS_COUNT=8000 \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-8000/test-set/objaverse/512/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# INPUT_RESOLUTION=512 \
# TARGET_RESOLUTION=512 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# 1027
# Resolution-32 Morton-rank DiT patch overfits; the earlier 32 run is the patch-size-1 baseline.
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# DIT_PATCH_SIZE=4 \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr32_n1000_patch4 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# DIT_PATCH_SIZE=8 \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr32_n1000_patch8 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# DIT_PATCH_SIZE=16 \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr32_n1000_patch16 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# GS_COUNT=4000 \
# DIT_PATCH_SIZE=4 \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr128_n4000_patch4 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# GS_COUNT=4000 \
# DIT_PATCH_SIZE=8 \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr128_n4000_patch8 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# GS_COUNT=4000 \
# DIT_PATCH_SIZE=16 \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr128_n4000_patch16 \
# bash scripts/overfit-noise-to-gs.sh

# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-4000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-4000/test-set/objaverse/128/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=dit \
# GS_COORDINATE_MODE=morton_rank \
# GS_COUNT=4000 \
# DIT_PATCH_SIZE=1 \
# INPUT_RESOLUTION=128 \
# TARGET_RESOLUTION=128 \
# GS_SH_DEGREE=0 \
# OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}_dit_morton_rank_hr128_n4000_patch16 \
# bash scripts/overfit-noise-to-gs.sh

export GPU_ID=3
export DATASET_ROOT=${data_root}
export SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000}
export TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000}
export GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json}
export SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
export PREDICTOR=${PREDICTOR:-dit}
export GS_COORDINATE_MODE=morton_rank
export SCENE_COUNT=18
export INPUT_RESOLUTION=32
export TARGET_RESOLUTION=32
export GS_SH_DEGREE=0
export OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/}

# Example: FLOW_COUPLING=morton GS_COORDINATE_MODE=morton_rank \
SCENE_MODE=one \
FLOW_COUPLING=random \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/dit_hr32_n1000_random_coupling \
bash scripts/overfit-noise-to-gs.sh

SCENE_MODE=one \
FLOW_COUPLING=morton \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/dit_hr32_n1000_morton_coupling \
bash scripts/overfit-noise-to-gs.sh

SCENE_MODE=one \
FLOW_COUPLING=morton \
GS_FOURIER_INPUT_FEATURES="['means', 'scales', 'quats']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 6, 'quats': 6}" \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/dit_hr32_n1000_morton_coupling_fourier \
bash scripts/overfit-noise-to-gs.sh

SCENE_MODE=many \
FLOW_COUPLING=random \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/mnay_dit_hr32_n1000_random_coupling \
bash scripts/overfit-noise-to-gs.sh

SCENE_MODE=many \
FLOW_COUPLING=morton \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/mnay_dit_hr32_n1000_morton_coupling \
bash scripts/overfit-noise-to-gs.sh

SCENE_MODE=many \
FLOW_COUPLING=morton \
GS_FOURIER_INPUT_FEATURES="['means', 'scales', 'quats']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 6, 'quats': 6}" \
OUTPUT_DIR=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}}/many_dit_hr32_n1000_morton_coupling_fourier \
bash scripts/overfit-noise-to-gs.sh

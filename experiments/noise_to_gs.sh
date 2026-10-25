#!/bin/bash
set -euo pipefail

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
# PREDICTOR=${PREDICTOR:-ptv3} \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

# GPU_ID=0 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dipt} \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh

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

GPU_ID=0 \
DATASET_ROOT=${data_root} \
SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
PREDICTOR=${PREDICTOR:-dit} \
GS_COORDINATE_MODE=morton_rank \
RANDOM_ROTATE=True \
INPUT_RESOLUTION=32 \
TARGET_RESOLUTION=32 \
GS_SH_DEGREE=0 \
OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
bash scripts/overfit-noise-to-gs.sh

# GPU_ID=0 \
# DATASET_ROOT=${data_root} \
# SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000} \
# GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-/project/ricky/splatformer-sr-data-mcmc-1000/test-set/objaverse/32/gs_statistics.json} \
# SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1} \
# PREDICTOR=${PREDICTOR:-dit} \
# GS_COORDINATE_MODE=morton_rank \
# SCENE_MODE=many \
# SCENE_COUNT=20 \
# INPUT_RESOLUTION=32 \
# TARGET_RESOLUTION=32 \
# GS_SH_DEGREE=0 \
# OUTPUT_ROOT=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/} \
# bash scripts/overfit-noise-to-gs.sh


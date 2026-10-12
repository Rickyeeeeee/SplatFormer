#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-3}

export INTERPOLANT_TYPE=linear
export GS_SHIFT_NEGATIVE_GRID_COORDS=True
export QUATERNION_REPRESENTATION=unit_unstandardized
export ROTATION_MODE=${ROTATION_MODE:-gravity_consistent}
# The launcher interpolates this optional output-name suffix under set -u.
export quaternion_suffix=

export RANDOM_JITTER=False
export RANDOM_ROTATE=False
# JITTER_MAX_LEVELS="{
# 'means': 0.01,
# 'scales': 0.05,
# 'opacities': 0.1,
# 'quats': 0.05,
# 'features_dc': 0.05,
# 'features_rest': 0.1
# }"

run_date=$(date +%m%d)

# for predictor in gdpt dipt; do
    # PREDICTOR="${predictor}" \
    # GDPT_CHANNELS=384 \
    # GDPT_NUM_HEAD=6 \
    # GDPT_USE_SPARSE_CONV=False \
    # SCENE_MODE=one \
    # LR_WARMUP_STEPS=500 \
    # BATCH_SIZE=16 \
    # OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/}${run_date}/${predictor}_one" \
    # bash scripts/overfit-sr-interpolants.sh 4000 4000 500 500 "$@"

    # PREDICTOR="${predictor}" \
    # SCENE_MODE=many \
    # SCENE_COUNT=18 \
    # LR_WARMUP_STEPS=1000 \
    # BATCH_SIZE=8 \
    # OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/1003}/${predictor}_many" \
    # bash scripts/overfit-sr-interpolants.sh 10000 10000 1000 1000 "$@"
# done

# PREDICTOR=gdpt \
# GDPT_CHANNELS=384 \
# GDPT_NUM_HEAD=6 \
# GDPT_USE_SPARSE_CONV=True \
# SCENE_MODE=one \
# LR_WARMUP_STEPS=500 \
# BATCH_SIZE=16 \
# OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/}${run_date}/gdpt_sparse_conv_one" \
# bash scripts/overfit-sr-interpolants.sh 4000 4000 500 500 "$@"

PREDICTOR=gdpt \
GDPT_CHANNELS=768 \
GDPT_NUM_HEAD=12 \
GDPT_USE_SPARSE_CONV=False \
SCENE_MODE=one \
LR_WARMUP_STEPS=500 \
BATCH_SIZE=8 \
OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/}${run_date}/gdpt_wide_one" \
bash scripts/overfit-sr-interpolants.sh 4000 4000 500 500 "$@"

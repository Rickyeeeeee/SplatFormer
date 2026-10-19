#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-1}

export INTERPOLANT_TYPE=linear
export QUATERNION_REPRESENTATION=unit_unstandardized
export ROTATION_MODE=${ROTATION_MODE:-gravity_consistent}
export RANDOM_JITTER=False
export RANDOM_ROTATE=False
export FLOW_STEPS=${FLOW_STEPS:-50}

run_date=$(date +%m%d)

# Compare attention modes with the same EDiPT width and GDPT training settings.
# EDIPT_ATTENTION_MODE=serialized \
# EDIPT_CHANNELS=${EDIPT_CHANNELS:-384} \
# EDIPT_NUM_HEAD=${EDIPT_NUM_HEAD:-6} \
# SCENE_MODE=one \
# LR_WARMUP_STEPS=500 \
# BATCH_SIZE=1 \
# OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/}${run_date}/edipt_serialized_one" \
# bash scripts/overfit-sr-interpolants-edipt.sh 4000 4000 500 500 "$@"

# Query tiling keeps global attention feasible for full scenes.
EDIPT_ATTENTION_MODE=global \
EDIPT_CHANNELS=${EDIPT_CHANNELS:-384} \
EDIPT_NUM_HEAD=${EDIPT_NUM_HEAD:-6} \
SCENE_MODE=one \
LR_WARMUP_STEPS=500 \
BATCH_SIZE=1 \
EDIPT_GLOBAL_QUERY_CHUNK_SIZE=8000 \
OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/}${run_date}/edipt_global_one" \
bash scripts/overfit-sr-interpolants-edipt.sh 4000 4000 500 500 "$@"

#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-2}
partition_mode=${DIPT_PARTITION_MODE:-single}
partition_suffix=
if [[ "${partition_mode}" != "single" ]]; then
    partition_suffix=_partition_${partition_mode}
fi

export PREDICTOR=dipt 
export DIPT_PARTITION_MODE="${partition_mode}" 
export INTERPOLANT_TYPE=linear 
export SCENE_MODE=many 
export SCENE_COUNT=18 
export LR_WARMUP_STEPS=1000 
export BATCH_SIZE=8 
export RANDOM_JITTER=True 
export GS_SHIFT_NEGATIVE_GRID_COORDS=True
export JITTER_MAX_LEVELS="{
'means': 0.01,
'scales': 0.05,
'opacities': 0.1,
'quats': 0.05,
'features_dc': 0.05,
'features_rest': 0.1
}"
export QUATERNION_REPRESENTATION=unit_unstandardized 

RANDOM_ROTATE=False \
OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/1002}/dipt_noise${partition_suffix}" \
bash scripts/overfit-sr-interpolants.sh 10000 10000 1000 1000"$@"

RANDOM_ROTATE=True \
ROTATION_MODE=${ROTATION_MODE:-gravity_consistent} \
ROTATION_MAX_DEGREES=180 \
OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/1002}/dipt_noise_rotate${partition_suffix}" \
bash scripts/overfit-sr-interpolants.sh 10000 10000 1000 1000"$@"
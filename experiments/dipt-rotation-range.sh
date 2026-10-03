#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-2}
partition_mode=${DIPT_PARTITION_MODE:-single}
partition_suffix=
if [[ "${partition_mode}" != "single" ]]; then
    partition_suffix=_partition_${partition_mode}
fi

# RANDOM_JITTER=True \
# JITTER_MAX_LEVELS="{
# 'means': 0.01,
# 'scales': 0.05,
# 'opacities': 0.1,
# 'quats': 0.05,
# 'features_dc': 0.05,
# 'features_rest': 0.1
# }" \
# Keep jitter off to isolate the effect of rotation range; 0 degrees is identity.
# for angle in ${ROTATION_RANGES:-90 180}; do
#     INTERPOLANT_TYPE=linear \
#     RANDOM_JITTER=False \
#     RANDOM_ROTATE=True \
#     ROTATION_MODE=${ROTATION_MODE:-gravity_consistent} \
#     ROTATION_MAX_DEGREES="$angle" \
#     GS_SHIFT_NEGATIVE_GRID_COORDS=True \
#     QUATERNION_REPRESENTATION=unit_unstandardized \
#     OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/0929/rotation_range}/${ROTATION_MODE:-gravity_consistent}_${angle}deg" \
#     bash scripts/overfit-sr-interpolants.sh "$@"
# done

for angle in ${ROTATION_RANGES:-0 45 90 180}; do
    PREDICTOR=dipt \
    DIPT_PARTITION_MODE="${partition_mode}" \
    INTERPOLANT_TYPE=linear \
    SCENE_MODE=many \
    SCENE_COUNT=18 \
    BATCH_SIZE=8 \
    RANDOM_JITTER=True \
    JITTER_MAX_LEVELS="{
    'means': 0.01,
    'scales': 0.05,
    'opacities': 0.1,
    'quats': 0.05,
    'features_dc': 0.05,
    'features_rest': 0.1
    }" \
    RANDOM_ROTATE=True \
    ROTATION_MODE=${ROTATION_MODE:-gravity_consistent} \
    ROTATION_MAX_DEGREES="$angle" \
    GS_SHIFT_NEGATIVE_GRID_COORDS=True \
    QUATERNION_REPRESENTATION=unit_unstandardized \
    OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/1001/rotation_range_512}/${ROTATION_MODE:-gravity_consistent}_${angle}deg${partition_suffix}" \
    bash scripts/overfit-sr-interpolants.sh 10000 10000 1000 1000"$@"
done

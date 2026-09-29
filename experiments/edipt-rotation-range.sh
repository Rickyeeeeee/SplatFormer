#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-3}

# Keep jitter off to isolate the effect of rotation range; 0 degrees is identity.
for angle in ${ROTATION_RANGES:-0 1 10 45}; do
    INTERPOLANT_TYPE=linear \
    RANDOM_JITTER=False \
    RANDOM_ROTATE=True \
    ROTATION_MODE=${ROTATION_MODE:-gravity_consistent} \
    ROTATION_MAX_DEGREES="$angle" \
    QUATERNION_REPRESENTATION=unit_unstandardized \
    OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/rotation_range_edipt}/${ROTATION_MODE:-gravity_consistent}_${angle}deg" \
    bash scripts/overfit-sr-interpolants-edipt.sh "$@"
done

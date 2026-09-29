#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-3}
    # SERIALIZATION_REFERENCE=unaugmented \
    # GS_USE_FEATURES_REST=False \

# Keep jitter off to isolate the effect of rotation range; 0 degrees is identity.
for angle in ${ROTATION_RANGES:-1 10 45}; do
    INTERPOLANT_TYPE=linear \
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
    QUATERNION_REPRESENTATION=unit_unstandardized \
    CUSTOM_POSFIX=fix_references \
    OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/rotation_range_edipt}/${ROTATION_MODE:-gravity_consistent}_${angle}deg" \
    bash scripts/overfit-sr-interpolants-edipt.sh "$@"
done

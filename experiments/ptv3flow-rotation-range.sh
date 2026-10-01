#!/bin/bash
set -euo pipefail
export GPU_ID=${GPU_ID:-3}

# Sweep rotation ranges with and without jitter, keeping outputs separate.
for random_jitter in False True; do
    jitter_suffix=
    if [[ "${random_jitter}" == "True" ]]; then
        jitter_suffix=_jitter
    fi
    for angle in ${ROTATION_RANGES:-0 45 90 180}; do
        PREDICTOR=ptv3 \
        BATCH_SIZE=16 \
        INTERPOLANT_TYPE=linear \
        RANDOM_JITTER="${random_jitter}" \
        RANDOM_ROTATE=True \
        ROTATION_MODE=${ROTATION_MODE:-gravity_consistent} \
        ROTATION_MAX_DEGREES="$angle" \
        QUATERNION_REPRESENTATION=unit_unstandardized \
        OUTPUT_DIR="${OUTPUT_ROOT:-/project2/ricky/experiments/rotation_range_ptv3flow}/${ROTATION_MODE:-gravity_consistent}_${angle}deg${jitter_suffix}" \
        bash scripts/overfit-sr-interpolants.sh "$@"
    done
done

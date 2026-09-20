#!/bin/bash

set -euo pipefail

run_date=$(date +%m%d)
# Fourier input ablation: retain raw inputs and use 6/4/4 bands for means/scales/quaternions.
# Override GPU_ID, FOURIER_OUTPUT_ROOT, or launcher positional arguments when needed.
GPU_ID=${GPU_ID:-6}
FOURIER_OUTPUT_ROOT=${FOURIER_OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}_fourier_ablation_no_raw}
GS_FOURIER_INCLUDE_RAW=False

# FeaturePredictor: means only.
# GPU_ID=${GPU_ID} \
# OUTPUT_DIR=${FOURIER_OUTPUT_ROOT}/default \
# bash scripts/overfit-sr-on-objaverse.sh

# FeaturePredictor: means only.
# GPU_ID=${GPU_ID} \
# OUTPUT_DIR=${FOURIER_OUTPUT_ROOT}/feature_means \
# FEATURE_FOURIER_INPUT_FEATURES="['means']" \
# FEATURE_FOURIER_NUM_FREQUENCIES="{'means': 6}" \
# bash scripts/overfit-sr-on-objaverse.sh

# FeaturePredictor: means and scales.
# GPU_ID=${GPU_ID} \
# OUTPUT_DIR=${FOURIER_OUTPUT_ROOT}/feature_means_scales \
# FEATURE_FOURIER_INPUT_FEATURES="['means', 'scales']" \
# FEATURE_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" \
# bash scripts/overfit-sr-on-objaverse.sh

# FeaturePredictor: means, scales, and quaternions.
# GPU_ID=${GPU_ID} \
# OUTPUT_DIR=${FOURIER_OUTPUT_ROOT}/feature_means_scales_quats \
# FEATURE_FOURIER_INPUT_FEATURES="['means', 'scales', 'quats']" \
# FEATURE_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4, 'quats': 4}" \
# bash scripts/overfit-sr-on-objaverse.sh

# GSFlowPredictor: 
GPU_ID=${GPU_ID} \
CUSTOM_POSFIX=fourier_no_384 \
bash scripts/overfit-sr-gsfm-on-objaverse.sh

# GSFlowPredictor: means only.
# GPU_ID=${GPU_ID} \
# CUSTOM_POSFIX=fourier_means_no_raw \
# GS_FOURIER_INPUT_FEATURES="['means']" \
# GS_FOURIER_NUM_FREQUENCIES="{'means': 6}" \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh

# GSFlowPredictor: means and scales.
# GPU_ID=${GPU_ID} \
# CUSTOM_POSFIX=fourier_means_scales_no_raw \
# GS_FOURIER_INPUT_FEATURES="['means', 'scales']" \
# GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh


# GSFlowPredictor: means, scales, and quaternions.
# GPU_ID=${GPU_ID} \
# CUSTOM_POSFIX=fourier_means_scales_quats_no_raw \
# GS_FOURIER_INPUT_FEATURES="['means', 'scales', 'quats']" \
# GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4, 'quats': 4}" \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh

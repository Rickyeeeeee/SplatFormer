#!/bin/bash
set -euo pipefail

# Usage: bash scripts/overfit-sr-flow-on-objaverse.sh [total_steps] [save_interval] [eval_interval] [log_image_interval]
#
# Examples:
#   # Point-supervised x1 prediction with a freshly optimized target each training step.
#   GPU_ID=5 X1_LOSS_MODE=point_mse X1_PREDICTION_TYPE=residual X1_OPTIMIZATION_STEPS=10000 \
#     bash scripts/overfit-sr-flow-on-objaverse.sh 1000 200 200 200
#
#   # Image-supervised x1 prediction without target-GS optimization.
#   GPU_ID=5 X1_LOSS_MODE=render_l1 X1_PREDICTION_TYPE=velocity_extrapolation \
#     bash scripts/overfit-sr-flow-on-objaverse.sh 1000 200 200 200

GPU_ID=${GPU_ID:-5}

SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
TOTAL_STEPS=${1:-${TOTAL_STEPS:-1000}}
SAVE_INTERVAL=${2:-${SAVE_INTERVAL:-200}}
EVAL_INTERVAL=${3:-${EVAL_INTERVAL:-200}}
LOG_IMAGE_INTERVAL=${4:-${LOG_IMAGE_INTERVAL:-200}}

X1_LOSS_MODE=${X1_LOSS_MODE:-point_mse}
X1_PREDICTION_TYPE=${X1_PREDICTION_TYPE:-residual}
X1_OPTIMIZATION_STEPS=${X1_OPTIMIZATION_STEPS:-10000}

case "${X1_LOSS_MODE}" in
    point_mse|render_l1)
        ;;
    *)
        echo "[ERROR] X1_LOSS_MODE must be point_mse or render_l1."
        exit 1
        ;;
esac

case "${X1_PREDICTION_TYPE}" in
    residual|velocity_extrapolation)
        ;;
    *)
        echo "[ERROR] X1_PREDICTION_TYPE must be residual or velocity_extrapolation."
        exit 1
        ;;
esac

if ! [[ "${X1_OPTIMIZATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] X1_OPTIMIZATION_STEPS must be a positive integer."
    exit 1
fi

NS_ROOT=${NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
COLMAP_ROOT=${COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/ricky/outputs/objaverse_overfit_sr_512_flow_x1}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${SCENE_NAME}}

echo "Using GPU: ${GPU_ID}"
echo "Scene: ${SCENE_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "X1 loss mode: ${X1_LOSS_MODE}"
echo "X1 prediction type: ${X1_PREDICTION_TYPE}"
echo "X1 optimization steps: ${X1_OPTIMIZATION_STEPS}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-flow.py \
    --output_dir="${OUTPUT_DIR}" \
    --scene_name="${SCENE_NAME}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr-x1.gin \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="training.x1_loss_mode='${X1_LOSS_MODE}'" \
    --gin_param="training.x1_prediction_type='${X1_PREDICTION_TYPE}'" \
    --gin_param="training.x1_optimization_steps=${X1_OPTIMIZATION_STEPS}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${COLMAP_ROOT}'"

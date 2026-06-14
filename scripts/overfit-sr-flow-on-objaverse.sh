#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-5o}

SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
TOTAL_STEPS=${1:-${TOTAL_STEPS:-1000}}
SAVE_INTERVAL=${2:-${SAVE_INTERVAL:-200}}
EVAL_INTERVAL=${3:-${EVAL_INTERVAL:-200}}
LOG_IMAGE_INTERVAL=${4:-${LOG_IMAGE_INTERVAL:-200}}

FLOW_TRAJECTORY_MODE=${FLOW_TRAJECTORY_MODE:-optimized_target_discrete}
FLOW_NUM_TIMESTEPS=${FLOW_NUM_TIMESTEPS:-10}
FLOW_SEGMENT_STEPS=${FLOW_SEGMENT_STEPS:-10}
FLOW_EVAL_STEPS=${FLOW_EVAL_STEPS:-${FLOW_NUM_TIMESTEPS}}
FLOW_LOSS_WEIGHT=${FLOW_LOSS_WEIGHT:-1.0}
FLOW_LOSS_TYPE=${FLOW_LOSS_TYPE:-mse}
FLOW_INTEGRATION_TIMESTEP_MODE=${FLOW_INTEGRATION_TIMESTEP_MODE:-left}

case "${FLOW_TRAJECTORY_MODE}" in
    optimized_target_discrete|progressive_segment)
        ;;
    *)
        echo "[ERROR] FLOW_TRAJECTORY_MODE must be optimized_target_discrete or progressive_segment."
        exit 1
        ;;
esac

NS_ROOT=${NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
COLMAP_ROOT=${COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/ricky/outputs/objaverse_splatformer_overfit_sr_flow512}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${SCENE_NAME}}

echo "Using GPU: ${GPU_ID}"
echo "Scene: ${SCENE_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Flow mode: ${FLOW_TRAJECTORY_MODE}"
echo "Flow timesteps: ${FLOW_NUM_TIMESTEPS}"
echo "GS segment steps: ${FLOW_SEGMENT_STEPS}"
echo "Eval integration steps: ${FLOW_EVAL_STEPS}"
echo "Flow loss: ${FLOW_LOSS_TYPE} x ${FLOW_LOSS_WEIGHT}"
echo "Integration timestep mode: ${FLOW_INTEGRATION_TIMESTEP_MODE}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-flow.py \
    --output_dir="${OUTPUT_DIR}" \
    --scene_name="${SCENE_NAME}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr-flow.gin \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="training.flow_trajectory_mode='${FLOW_TRAJECTORY_MODE}'" \
    --gin_param="training.flow_num_timesteps=${FLOW_NUM_TIMESTEPS}" \
    --gin_param="training.flow_segment_steps=${FLOW_SEGMENT_STEPS}" \
    --gin_param="training.flow_eval_steps=${FLOW_EVAL_STEPS}" \
    --gin_param="training.flow_loss_weight=${FLOW_LOSS_WEIGHT}" \
    --gin_param="training.flow_loss_type='${FLOW_LOSS_TYPE}'" \
    --gin_param="training.flow_integration_timestep_mode='${FLOW_INTEGRATION_TIMESTEP_MODE}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${COLMAP_ROOT}'"

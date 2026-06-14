#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

GPU_ID=5

PYTHON=${PYTHON:-python}
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

FLOW_TRAJECTORY_MODE=${FLOW_TRAJECTORY_MODE:-optimized_target_discrete}
FLOW_NUM_TIMESTEPS=${FLOW_NUM_TIMESTEPS:-10}
FLOW_SEGMENT_STEPS=${FLOW_SEGMENT_STEPS:-10}
FLOW_EVAL_STEPS=${FLOW_EVAL_STEPS:-${FLOW_NUM_TIMESTEPS}}
FLOW_LOSS_WEIGHT=${FLOW_LOSS_WEIGHT:-1.0}
FLOW_LOSS_TYPE=${FLOW_LOSS_TYPE:-mse}

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
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/test_sr_flow}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${SCENE_NAME}}
DIAGNOSTIC_SUBDIR=${DIAGNOSTIC_SUBDIR:-test_flow}
DIAGNOSTIC_TIMESTEPS=${DIAGNOSTIC_TIMESTEPS:--1}
CKPT=${CKPT:-}

echo "Using GPU: ${GPU_ID}"
echo "Scene: ${SCENE_NAME}"
echo "Output: ${OUTPUT_DIR}/${DIAGNOSTIC_SUBDIR}"
echo "Checkpoint: ${CKPT:-<none; teacher/oracle only>}"
echo "Flow mode: ${FLOW_TRAJECTORY_MODE}"
echo "Flow timesteps: ${FLOW_NUM_TIMESTEPS}"
echo "GS segment steps: ${FLOW_SEGMENT_STEPS}"
echo "Diagnostic timesteps: ${DIAGNOSTIC_TIMESTEPS}"

ARGS=(
    test-sr-flow.py
    --output_dir="${OUTPUT_DIR}"
    --diagnostic_subdir="${DIAGNOSTIC_SUBDIR}"
    --scene_name="${SCENE_NAME}"
    --diagnostic_timesteps="${DIAGNOSTIC_TIMESTEPS}"
    --gin_file=configs/model/ptv3.gin
    --gin_file=configs/overfit/sr-flow.gin
    --gin_param="training.flow_trajectory_mode='${FLOW_TRAJECTORY_MODE}'"
    --gin_param="training.flow_num_timesteps=${FLOW_NUM_TIMESTEPS}"
    --gin_param="training.flow_segment_steps=${FLOW_SEGMENT_STEPS}"
    --gin_param="training.flow_eval_steps=${FLOW_EVAL_STEPS}"
    --gin_param="training.flow_loss_weight=${FLOW_LOSS_WEIGHT}"
    --gin_param="training.flow_loss_type='${FLOW_LOSS_TYPE}'"
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${NS_ROOT}'"
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${COLMAP_ROOT}'"
)

if [ -n "${CKPT}" ]; then
    ARGS+=(--ckpt="${CKPT}")
fi

mkdir -p "${OUTPUT_DIR}/${DIAGNOSTIC_SUBDIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${ARGS[@]}"

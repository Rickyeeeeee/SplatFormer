#!/bin/bash
set -euo pipefail

# Usage: bash scripts/overfit-sr-ar-on-objaverse.sh [stages] [rollouts] [save_interval] [eval_interval] [log_image_interval]

GPU_ID=${GPU_ID:-4}
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

AR_NUM_STAGES=${1:-${AR_NUM_STAGES:-10}}
AR_NUM_ROLLOUTS=${2:-${AR_NUM_ROLLOUTS:-${AR_STEPS_PER_STAGE:-100}}}
SAVE_INTERVAL=${3:-${SAVE_INTERVAL:-200}}
EVAL_INTERVAL=${4:-${EVAL_INTERVAL:-200}}
LOG_IMAGE_INTERVAL=${5:-${LOG_IMAGE_INTERVAL:-200}}

IMAGE_L1_LOSS_WEIGHT=${IMAGE_L1_LOSS_WEIGHT:-1.0}
LPIPS_LOSS_WEIGHT=${LPIPS_LOSS_WEIGHT:-1.0}

NS_ROOT=${NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
COLMAP_ROOT=${COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/ricky/outputs/objaverse_splatformer_overfit_sr_ar_roll512}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${SCENE_NAME}}

if (( AR_NUM_STAGES <= 0 || AR_NUM_ROLLOUTS <= 0 )); then
    echo "[ERROR] AR_NUM_STAGES and AR_NUM_ROLLOUTS must be positive."
    exit 1
fi

TOTAL_STEPS=$((AR_NUM_STAGES * AR_NUM_ROLLOUTS))

echo "Using GPU: ${GPU_ID}"
echo "Scene: ${SCENE_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "AR schedule: ${AR_NUM_ROLLOUTS} rollouts x ${AR_NUM_STAGES} stages = ${TOTAL_STEPS} steps"
echo "Image loss: L1 x ${IMAGE_L1_LOSS_WEIGHT}, LPIPS x ${LPIPS_LOSS_WEIGHT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-ar.py \
    --output_dir="${OUTPUT_DIR}" \
    --scene_name="${SCENE_NAME}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr-ar.gin \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.ar_num_stages=${AR_NUM_STAGES}" \
    --gin_param="training.ar_num_rollouts=${AR_NUM_ROLLOUTS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="training.image_l1_loss_weight=${IMAGE_L1_LOSS_WEIGHT}" \
    --gin_param="training.lpips_loss_weight=${LPIPS_LOSS_WEIGHT}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${COLMAP_ROOT}'"

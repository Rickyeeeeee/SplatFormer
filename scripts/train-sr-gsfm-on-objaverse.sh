#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-8}
TOTAL_STEPS=${1:-1000000}
SAVE_INTERVAL=${2:-2000}
EVAL_INTERVAL=${3:-2000}
LOG_IMAGE_INTERVAL=${4:-1000}
ALIGNMENT=${5:-${ALIGNMENT:-fit_lr_to_hr}}
ATTRIBUTE_INIT=${6:-${ATTRIBUTE_INIT:-aligned}}
INPUT_RESOLUTION=${7:-${INPUT_RESOLUTION:-128}}
TARGET_RESOLUTION=${8:-${TARGET_RESOLUTION:-512}}
MIX_SCHEDULE=${9:-${MIX_SCHEDULE:-fm-only}}
FLOW_LOSS_TYPE=${10:-${FLOW_LOSS_TYPE:-velocity}}
FLOW_STEPS=${11:-${FLOW_STEPS:-5}}
FLOW_NOISE_STD=${12:-${FLOW_NOISE_STD:-0.0}}
IMAGE_L1_LOSS_WEIGHT=${13:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
LPIPS_LOSS_WEIGHT=${14:-${LPIPS_LOSS_WEIGHT:-1.0}}

DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/objaverse_splatformer_sr_gsfm_${INPUT_RESOLUTION}to${TARGET_RESOLUTION}_${ALIGNMENT}_${MIX_SCHEDULE}}

case "${MIX_SCHEDULE}" in
    linear|free-range-gs|fm-only) ;;
    *) echo "Unsupported MIX_SCHEDULE=${MIX_SCHEDULE}" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=${GPU_ID} python train-sr-gsfm.py \
    --output_dir="${OUTPUT_DIR}" \
    --alignment="${ALIGNMENT}" \
    --attribute_init="${ATTRIBUTE_INIT}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/dataset/objaverse-sr-dev.gin \
    --gin_file=configs/train/sr_gsfm.gin \
    --gin_param="GSFlowPredictor.grid_resolution=1024" \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="SplatFactoSRDevDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDevDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    --gin_param="flow_matching.gs_statistics_path='${GS_STATISTICS_PATH}'" \
    --gin_param="flow_matching.flow_steps=${FLOW_STEPS}" \
    --gin_param="flow_matching.flow_noise_std=${FLOW_NOISE_STD}" \
    --gin_param="flow_matching.loss_type='${FLOW_LOSS_TYPE}'" \
    --gin_param="loss_mixing.schedule='${MIX_SCHEDULE}'" \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="training.image_l1_loss_weight=${IMAGE_L1_LOSS_WEIGHT}" \
    --gin_param="training.lpips_loss_weight=${LPIPS_LOSS_WEIGHT}" \
    --gin_param="train2D/build_scheduler.total_step=${TOTAL_STEPS}"

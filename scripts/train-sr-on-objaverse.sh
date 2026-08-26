#!/bin/bash
set -euo pipefail

NGPUS=${NGPUS:-1}
GPU_IDS=${GPU_IDS:-}
MASTER_PORT=${MASTER_PORT:-29518}
TOTAL_STEPS=${1:-20000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
INPUT_RESOLUTION=${5:-${INPUT_RESOLUTION:-128}}
TARGET_RESOLUTION=${6:-${TARGET_RESOLUTION:-512}}
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/objaverse_splatformer_sr_${INPUT_RESOLUTION}to${TARGET_RESOLUTION}}

DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_valid_scenes.csv}

if [[ -n "${GPU_IDS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --rdzv-endpoint="localhost:${MASTER_PORT}" \
    train-sr.py \
    --output_dir="${OUTPUT_DIR}" \
    --input_resolution="${INPUT_RESOLUTION}" \
    --target_resolution="${TARGET_RESOLUTION}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/train/sr.gin \
    --gin_param="dataset_root=\"${DATASET_ROOT}\"" \
    --gin_param="train_scene_list=\"${TRAIN_SCENE_LIST}\"" \
    --gin_param="test_scene_list=\"${TEST_SCENE_LIST}\"" \
    --gin_param="SplatFactoSRDataset.resolutions=[${INPUT_RESOLUTION}, ${TARGET_RESOLUTION}]" \
    --gin_param="SplatFactoSRDataset.fit_source_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.fit_target_resolution=${TARGET_RESOLUTION}" \
    --gin_param="total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}"

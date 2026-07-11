#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
TOTAL_STEPS=${1:-20000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
ALIGNMENT=${5:-${ALIGNMENT:-emd}}
ATTRIBUTE_INIT=${6:-${ATTRIBUTE_INIT:-aligned}}
INPUT_FACTOR=${7:-${INPUT_FACTOR:-4}}
TARGET_FACTOR=${8:-${TARGET_FACTOR:-2}}
MEANS_SOURCE=${9:-${MEANS_SOURCE:-gt}}
case "${MEANS_SOURCE}" in
    gt|predicted|splatformer) ;;
    *)
        echo "Unsupported MEANS_SOURCE='${MEANS_SOURCE}'. Expected gt, predicted, or splatformer." >&2
        exit 1
        ;;
esac

TRAIN_NS_ROOT=${TRAIN_NS_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/nerfstudio}
TRAIN_COLMAP_ROOT=${TRAIN_COLMAP_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/colmap}
TEST_NS_ROOT=${TEST_NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
TEST_COLMAP_ROOT=${TEST_COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/objaverse_splatformer_train_sr_2stage_${INPUT_FACTOR}to${TARGET_FACTOR}_${MEANS_SOURCE}}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train-sr-2stage.py \
    --output_dir="${OUTPUT_DIR}" \
    --input_factor="${INPUT_FACTOR}" \
    --target_factor="${TARGET_FACTOR}" \
    --means_source="${MEANS_SOURCE}" \
    --alignment="${ALIGNMENT}" \
    --attribute_init="${ATTRIBUTE_INIT}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/sr_2stage.gin \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TRAIN_NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TRAIN_COLMAP_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TEST_NS_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TEST_COLMAP_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.factors=[${TARGET_FACTOR}, ${INPUT_FACTOR}]" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.factors=[${TARGET_FACTOR}, ${INPUT_FACTOR}]"

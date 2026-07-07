#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-8}
TOTAL_STEPS=${1:-20000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
ALIGNMENT=${5:-${ALIGNMENT:-emd}}
ATTRIBUTE_INIT=${6:-${ATTRIBUTE_INIT:-aligned}}
INPUT_FACTOR=${7:-${INPUT_FACTOR:-4}}
TARGET_FACTOR=${8:-${TARGET_FACTOR:-2}}
LOSS_FEATURES=${9:-${LOSS_FEATURES:-}}
POST_ACTIVATE_LOSS=${10:-${POST_ACTIVATE_LOSS:-false}}
DIRECT_PREDICTION=${11:-${DIRECT_PREDICTION:-false}}
MEANS_ORIGIN_SCALE=${12:-${MEANS_ORIGIN_SCALE:-1.0}}
OUTPUT_DIR=${OUTPUT_DIR:-/project/ricky/outputs/objaverse_splatformer_sr_mse_${INPUT_FACTOR}to${TARGET_FACTOR}}
MIN_TRAIN_SPLATS=${MIN_TRAIN_SPLATS:-20000}

TRAIN_NS_ROOT=${TRAIN_NS_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/nerfstudio}
TRAIN_COLMAP_ROOT=${TRAIN_COLMAP_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/colmap}
TEST_NS_ROOT=${TEST_NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
TEST_COLMAP_ROOT=${TEST_COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}

to_bit() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES) echo 1 ;;
        *) echo 0 ;;
    esac
}

if [ "$(to_bit "${DIRECT_PREDICTION}")" = "1" ]; then
    OUTPUT_FEATURES_TYPE=dc
    MAX_SCALE_NORMALIZED=${MAX_SCALE_NORMALIZED:--1}
else
    OUTPUT_FEATURES_TYPE=res
    MAX_SCALE_NORMALIZED=${MAX_SCALE_NORMALIZED:-1e-2}
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python train-sr-mse.py \
    --output_dir="${OUTPUT_DIR}" \
    --min_train_splats_per_factor="${MIN_TRAIN_SPLATS}" \
    --input_factor="${INPUT_FACTOR}" \
    --target_factor="${TARGET_FACTOR}" \
    --alignment="${ALIGNMENT}" \
    --attribute_init="${ATTRIBUTE_INIT}" \
    --loss_features="${LOSS_FEATURES}" \
    --post_activate_loss="${POST_ACTIVATE_LOSS}" \
    --means_origin_scale="${MEANS_ORIGIN_SCALE}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/sr_mse.gin \
    --gin_param="FeaturePredictor.output_features_type='${OUTPUT_FEATURES_TYPE}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${MAX_SCALE_NORMALIZED}" \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.factors=[${TARGET_FACTOR}, ${INPUT_FACTOR}]" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.factors=[${TARGET_FACTOR}, ${INPUT_FACTOR}]" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TRAIN_NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TRAIN_COLMAP_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TEST_NS_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TEST_COLMAP_ROOT}'"

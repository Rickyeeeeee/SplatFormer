#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-8}
TOTAL_STEPS=${1:-20000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-/project/ricky/outputs/objaverse_splatformer_sr_4to1_no20k}
MIN_TRAIN_SPLATS=${MIN_TRAIN_SPLATS:-20000}

TRAIN_NS_ROOT=${TRAIN_NS_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/nerfstudio}
TRAIN_COLMAP_ROOT=${TRAIN_COLMAP_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/colmap}
TEST_NS_ROOT=${TEST_NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
TEST_COLMAP_ROOT=${TEST_COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train-sr.py \
    --output_dir="${OUTPUT_DIR}" \
    --min_train_splats_per_factor="${MIN_TRAIN_SPLATS}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/sr.gin \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TRAIN_NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TRAIN_COLMAP_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TEST_NS_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TEST_COLMAP_ROOT}'"

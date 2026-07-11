#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-8}
total_steps=${1:-20000}
save_interval=${2:-1000}
eval_interval=${3:-1000}
log_image_interval=${4:-1000}
input_factor=${5:-${INPUT_FACTOR:-4}}
target_factor=${6:-${TARGET_FACTOR:-2}}
OUTPUT_DIR=${OUTPUT_DIR:-/project/ricky/outputs/objaverse_splatformer_sr_${input_factor}to${target_factor}_no20k_anchor}
MIN_TRAIN_SPLATS=${MIN_TRAIN_SPLATS:-20000}

TRAIN_NS_ROOT=${TRAIN_NS_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/nerfstudio}
TRAIN_COLMAP_ROOT=${TRAIN_COLMAP_ROOT:-/project2/ricky/splatformer-data/train-set-512/objaverse/colmap}
TEST_NS_ROOT=${TEST_NS_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
TEST_COLMAP_ROOT=${TEST_COLMAP_ROOT:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train-sr.py \
    --output_dir="${OUTPUT_DIR}" \
    --min_train_splats_per_factor="${MIN_TRAIN_SPLATS}" \
    --input_factor="${input_factor}" \
    --target_factor="${target_factor}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/sr.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TRAIN_NS_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TRAIN_COLMAP_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${TEST_NS_ROOT}'" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.colmap_folder='${TEST_COLMAP_ROOT}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.factors=[${target_factor}, ${input_factor}]" \
    --gin_param="test_dataset/SplatFactoMultiLevelDataset.factors=[${target_factor}, ${input_factor}]"

#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-8}
TOTAL_STEPS=${1:-20000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
ALIGNMENT=${5:-${ALIGNMENT:-fit_lr_to_hr}}
ATTRIBUTE_INIT=${6:-${ATTRIBUTE_INIT:-aligned}}
INPUT_RESOLUTION=${7:-${INPUT_RESOLUTION:-128}}
TARGET_RESOLUTION=${8:-${TARGET_RESOLUTION:-512}}
POST_ACTIVATE_LOSS=${9:-${POST_ACTIVATE_LOSS:-true}}
DIRECT_PREDICTION=${10:-${DIRECT_PREDICTION:-false}}
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/objaverse_splatformer_sr_mse_${INPUT_RESOLUTION}to${TARGET_RESOLUTION}_${ALIGNMENT}}

DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}

case "${ALIGNMENT}" in
    emd|random|fit_lr_to_hr|fit_hr_to_lr) ;;
    *)
        echo "Unsupported ALIGNMENT=${ALIGNMENT}. Use emd, random, fit_lr_to_hr, or fit_hr_to_lr." >&2
        exit 1
        ;;
esac

case "${ATTRIBUTE_INIT}" in
    aligned|3dgs) ;;
    *)
        echo "Unsupported ATTRIBUTE_INIT=${ATTRIBUTE_INIT}. Use aligned or 3dgs." >&2
        exit 1
        ;;
esac

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
    --input_resolution="${INPUT_RESOLUTION}" \
    --target_resolution="${TARGET_RESOLUTION}" \
    --alignment="${ALIGNMENT}" \
    --attribute_init="${ATTRIBUTE_INIT}" \
    --post_activate_loss="${POST_ACTIVATE_LOSS}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr-dev.gin \
    --gin_file=configs/train/sr_mse.gin \
    --gin_param="dataset_root=\"${DATASET_ROOT}\"" \
    --gin_param="train_scene_list=\"${TRAIN_SCENE_LIST}\"" \
    --gin_param="test_scene_list=\"${TEST_SCENE_LIST}\"" \
    --gin_param="SplatFactoSRDevDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDevDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    --gin_param="FeaturePredictor.output_features_type=\"${OUTPUT_FEATURES_TYPE}\"" \
    --gin_param="FeaturePredictor.max_scale_normalized=${MAX_SCALE_NORMALIZED}" \
    --gin_param="FeaturePredictor.grid_resolution=1024" \
    --gin_param="total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="train_dataset/SplatFactoSRDevDataset.load_src_gs=True" \
    --gin_param="train_dataset/SplatFactoSRDevDataset.load_tgt_gs=True" \
    --gin_param="train_dataset/SplatFactoSRDevDataset.load_src_images=False" \
    --gin_param="train_dataset/SplatFactoSRDevDataset.load_tgt_images=False"

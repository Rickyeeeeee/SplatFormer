#!/bin/bash

set -euo pipefail
GPU_ID=${GPU_ID:-5}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
FIT_LR_TO_HR_ROOT=${FIT_LR_TO_HR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
FIT_HR_TO_LR_ROOT=${FIT_HR_TO_LR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
echo "Using GPU: $GPU_ID"

total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
input_resolution=${5:-${INPUT_RESOLUTION:-32}}
target_resolution=${6:-${TARGET_RESOLUTION:-128}}
scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

out_name=${scene_name}_ir${input_resolution}_tr${target_resolution}
output_dir=${OUTPUT_DIR:-/project/ricky/outputs/objaverse_splatformer_overfit_sr/${out_name}}

# Optional Fourier input overrides use Gin literals and are ignored when unset.
# Example: FEATURE_FOURIER_INPUT_FEATURES="['means', 'scales']" FEATURE_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" bash scripts/overfit-sr-on-objaverse.sh
# Optional: FEATURE_FOURIER_INCLUDE_RAW=False FEATURE_FOURIER_LOG_SAMPLING=False FEATURE_FOURIER_MAX_FREQUENCY_LOG2="{'means': 5}"
fourier_gin_args=()
for parameter in fourier_input_features fourier_num_frequencies fourier_include_raw fourier_log_sampling fourier_max_frequency_log2; do
    env_name=FEATURE_${parameter^^}
    value=${!env_name:-}
    if [[ -n "${value}" ]]; then
        fourier_gin_args+=("--gin_param=FeaturePredictor.${parameter}=${value}")
    fi
done

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/overfit/sr.gin \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="test_fit_lr_to_hr_root='${FIT_LR_TO_HR_ROOT}'" \
    --gin_param="test_fit_hr_to_lr_root='${FIT_HR_TO_LR_ROOT}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${input_resolution}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${target_resolution}" \
    --gin_param="SplatFactoSRDataset.load_src_gs=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_gs=True" \
    --gin_param="SplatFactoSRDataset.load_src_images=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_images=True" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    "${fourier_gin_args[@]}"
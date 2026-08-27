#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-4}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_valid_scenes.csv}
FIT_LR_TO_HR_ROOT=${FIT_LR_TO_HR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
FIT_HR_TO_LR_ROOT=${FIT_HR_TO_LR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
alignment=${5:-emd}
attribute_init=${6:-3dgs}
input_resolution=${7:-128}
target_resolution=${8:-512}
post_activate_loss=${9:-${POST_ACTIVATE_LOSS:-true}}
direct_prediction=${10:-${DIRECT_PREDICTION:-false}}
gs_statistics_path=${11:-${GS_STATISTICS_PATH:-}}

grid_resolution=${GRID_RESOLUTION:-384}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0827/input_frame_v1}
to_bit() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES) echo 1 ;;
        *) echo 0 ;;
    esac
}

case "${alignment}" in
    emd|random|fit_lr_to_hr|fit_hr_to_lr) ;;
    *)
        echo "Unsupported alignment '${alignment}'. Use emd, random, fit_lr_to_hr, or fit_hr_to_lr." >&2
        exit 1
        ;;
esac

case "${attribute_init}" in
    aligned|3dgs) ;;
    *)
        echo "Unsupported attribute_init '${attribute_init}'. Use aligned or 3dgs." >&2
        exit 1
        ;;
esac

post_activate_bit=$(to_bit "${post_activate_loss}")
direct_prediction_bit=$(to_bit "${direct_prediction}")
if [ -n "${gs_statistics_path}" ] && [ "${post_activate_bit}" = "1" ]; then
    echo "GS_STATISTICS_PATH requires POST_ACTIVATE_LOSS=false" >&2
    exit 1
fi

if [ "${direct_prediction_bit}" = "1" ]; then
    output_features_type=dc
    max_scale_normalized=${MAX_SCALE_NORMALIZED:--1}
else
    output_features_type=res
    max_scale_normalized=${MAX_SCALE_NORMALIZED:-1e-2}
fi

stats_suffix=""
gs_statistics_args=()
if [ -n "${gs_statistics_path}" ]; then
    stats_suffix="_gsnorm"
    gs_statistics_args=(--gs_statistics_path="${gs_statistics_path}")
fi

out_name=${scene_name}_${alignment}_${attribute_init}_ir${input_resolution}_tr${target_resolution}_grid${grid_resolution}_input_frame_v1${stats_suffix}
output_dir=${output_root}/${out_name}

echo "Using GPU: ${GPU_ID}"
echo "Alignment: ${alignment}"
echo "Attributes: all"
echo "Output: ${output_dir}"

TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" python overfit-sr-mse.py \
    --output_dir="${output_dir}" \
    --scene_name="${scene_name}" \
    --alignment="${alignment}" \
    --attribute_init="${attribute_init}" \
    --post_activate_loss="${post_activate_loss}" \
    "${gs_statistics_args[@]}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr-dev.gin \
    --gin_file=configs/overfit/sr_mse.gin \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="fit_lr_to_hr_root='${FIT_LR_TO_HR_ROOT}'" \
    --gin_param="fit_hr_to_lr_root='${FIT_HR_TO_LR_ROOT}'" \
    --gin_param="SplatFactoSRDevDataset.src_resolution=${input_resolution}" \
    --gin_param="SplatFactoSRDevDataset.tgt_resolution=${target_resolution}" \
    --gin_param="SplatFactoSRDevDataset.load_gs=True" \
    --gin_param="SplatFactoSRDevDataset.load_images=True" \
    --gin_param="FeaturePredictor.output_features_type='${output_features_type}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${max_scale_normalized}" \
    --gin_param="FeaturePredictor.grid_resolution=${grid_resolution}" \
    --gin_param="total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}"

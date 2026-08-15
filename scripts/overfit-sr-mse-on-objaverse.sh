#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-4}
scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
alignment=${5:-emd}
attribute_init=${6:-3dgs}
input_factor=${7:-4}
target_factor=${8:-1}
post_activate_loss=${9:-${POST_ACTIVATE_LOSS:-true}}
direct_prediction=${10:-${DIRECT_PREDICTION:-false}}
gs_statistics_path=${11:-${GS_STATISTICS_PATH:-}}

matching_steps=${MATCHING_STEPS:-2000}
matching_image_per_step=${MATCHING_IMAGE_PER_STEP:-32}
matching_l1_weight=${MATCHING_L1_LOSS_WEIGHT:-1.0}
matching_lpips_weight=${MATCHING_LPIPS_LOSS_WEIGHT:-1.0}
alignment_cache_root=${ALIGNMENT_CACHE_ROOT:-/project2/ricky/splatformer-data-to-4x}
force_alignment_fit=${FORCE_ALIGNMENT_FIT:-false}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0814-unify-tuned-wo-mean/overfit_sr_mse_512}

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

out_name=${scene_name}_${alignment}_${attribute_init}_if${input_factor}_tf${target_factor}_${output_features_type}_pa${post_activate_bit}${stats_suffix}
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
    --input_factor="${input_factor}" \
    --target_factor="${target_factor}" \
    --post_activate_loss="${post_activate_loss}" \
    --alignment_cache_root="${alignment_cache_root}" \
    --force_alignment_fit="${force_alignment_fit}" \
    "${gs_statistics_args[@]}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr_mse.gin \
    --gin_param="FeaturePredictor.output_features_type='${output_features_type}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${max_scale_normalized}" \
    --gin_param="total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="matching_total_steps=${matching_steps}" \
    --gin_param="matching_fit.image_per_step=${matching_image_per_step}" \
    --gin_param="matching_fit.image_l1_loss_weight=${matching_l1_weight}" \
    --gin_param="matching_fit.lpips_loss_weight=${matching_lpips_weight}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project2/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project2/ricky/splatformer-data/test-set-512/objaverse/colmap'"

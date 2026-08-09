#!/bin/bash

GPU_ID=${GPU_ID:-4}
echo "Using GPU: $GPU_ID"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
total_steps=${1:-1000}
save_interval=${2:-1000}
eval_interval=${3:-200}
log_image_interval=${4:-200}
input_factor=${5:-4}
target_factor=${6:-2}
loss_features=${7:-${LOSS_FEATURES:-means}}
post_activate_loss=${8:-${POST_ACTIVATE_LOSS:-true}}
direct_prediction=${9:-${DIRECT_PREDICTION:-false}}
means_origin_scale=${10:-${MEANS_ORIGIN_SCALE:-1.01}}
model_features_from_loss=${11:-${MODEL_FEATURE_FROM_LOSS:-true}}
gs_statistics_path=${12:-${GS_STATISTICS_PATH:-}}

to_bit() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES) echo 1 ;;
        *) echo 0 ;;
    esac
}

sanitize_name() {
    echo "$1" | tr ',' '-'
}

loss_name="$(sanitize_name "${loss_features}")"
post_activate_bit="$(to_bit "${post_activate_loss}")"
direct_prediction_bit="$(to_bit "${direct_prediction}")"
model_features_from_loss_bit="$(to_bit "${model_features_from_loss}")"
if [ -n "${gs_statistics_path}" ] && [ "${post_activate_bit}" = "1" ]; then
    echo "GS_STATISTICS_PATH requires POST_ACTIVATE_LOSS=false" >&2
    exit 1
fi
if [ "${model_features_from_loss_bit}" = "1" ]; then
    model_output_suffix=lossout
else
    model_output_suffix=fullout
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

matching_steps=${MATCHING_STEPS:-3000}
matching_image_per_step=${MATCHING_IMAGE_PER_STEP:-16}
matching_l1_weight=${MATCHING_L1_LOSS_WEIGHT:-1.0}
matching_lpips_weight=${MATCHING_LPIPS_LOSS_WEIGHT:-1.0}
out_name=${scene_name}_if${input_factor}_tf${target_factor}_noemd_mse_${loss_name}_${model_output_suffix}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0809/overfit_sr_mse_noemd_512}
output_dir=${output_root}/${out_name}

TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-mse-noemd.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --loss_features=${loss_features} \
    --model_features_from_loss=${model_features_from_loss} \
    --post_activate_loss=${post_activate_loss} \
    --means_origin_scale=${means_origin_scale} \
    "${gs_statistics_args[@]}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr_mse_noemd.gin \
    --gin_param="FeaturePredictor.output_features_type='${output_features_type}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${max_scale_normalized}" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="matching_fit.total_steps=${matching_steps}" \
    --gin_param="matching_fit.image_per_step=${matching_image_per_step}" \
    --gin_param="matching_fit.image_l1_loss_weight=${matching_l1_weight}" \
    --gin_param="matching_fit.lpips_loss_weight=${matching_lpips_weight}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"

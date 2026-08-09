#!/bin/bash

GPU_ID=${GPU_ID:-5}
echo "Using GPU: $GPU_ID"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
total_steps=${1:-4000}
save_interval=${2:-4000}
eval_interval=${3:-400}
log_image_interval=${4:-400}
input_factor=${5:-4}
target_factor=${6:-2}
loss_features=${7:-${LOSS_FEATURES:-means}}
flow_steps=${8:-${FLOW_STEPS:-1}}
flow_noise_std=${9:-${FLOW_NOISE_STD:-0.0}}
flow_t_eps=${FLOW_T_EPS:-1e-4}
means_origin_scale=${MEANS_ORIGIN_SCALE:-1.05}
model_features_from_loss=${MODEL_FEATURES_FROM_LOSS:-false}
matching_steps=${MATCHING_STEPS:-2000}
matching_image_per_step=${MATCHING_IMAGE_PER_STEP:-16}
matching_l1_weight=${MATCHING_L1_LOSS_WEIGHT:-1.0}
matching_lpips_weight=${MATCHING_LPIPS_LOSS_WEIGHT:-1.0}
pre_matching_root=${PRE_MATCHING_ROOT:-/project2/ricky/splatformer-data-to-4x}
force_pre_matching=${FORCE_PRE_MATCHING:-false}

sanitize_name() {
    echo "$1" | tr ',' '-'
}

loss_name="$(sanitize_name "${loss_features}")"
out_name=${scene_name}_if${input_factor}_tf${target_factor}_gsfm_noemd_${loss_name}}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0810/overfit_sr_gsfm_noemd_512}
output_dir=${output_root}/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-gsfm-noemd.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --pre_matching_root="${pre_matching_root}" \
    --force_pre_matching=${force_pre_matching} \
    --loss_features=${loss_features} \
    --model_features_from_loss=${model_features_from_loss} \
    --means_origin_scale=${means_origin_scale} \
    --flow_steps=${flow_steps} \
    --flow_noise_std=${flow_noise_std} \
    --gin_param="flow_matching.flow_t_eps=${flow_t_eps}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/overfit/sr_gsfm_noemd.gin \
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

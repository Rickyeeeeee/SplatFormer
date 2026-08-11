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
image_l1_loss_weight=${10:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
lpips_loss_weight=${11:-${LPIPS_LOSS_WEIGHT:-1.0}}
mix_schedule=${12:-${MIX_SCHEDULE:-free-range-gs}}
flow_t_eps=${FLOW_T_EPS:-1e-4}
means_origin_scale=${MEANS_ORIGIN_SCALE:-1.01}
model_features_from_loss=${MODEL_FEATURES_FROM_LOSS:-false}
matching_steps=${MATCHING_STEPS:-3000}
matching_image_per_step=${MATCHING_IMAGE_PER_STEP:-16}
matching_l1_weight=${MATCHING_L1_LOSS_WEIGHT:-1.0}
matching_lpips_weight=${MATCHING_LPIPS_LOSS_WEIGHT:-1.0}
pre_matching_root=${PRE_MATCHING_ROOT:-/project2/ricky/splatformer-data-to-4x}
force_pre_matching=${FORCE_PRE_MATCHING:-false}
ptv3_drop_path=${PTV3_DROP_PATH:-0.3}
ptv3_shuffle_orders=${PTV3_SHUFFLE_ORDERS:-True}
ptv3_shuffle_orders_eval=${PTV3_SHUFFLE_ORDERS_EVAL:-True}
ptv3_turn_off_bn=${PTV3_TURN_OFF_BN:-False}

sanitize_name() {
    echo "$1" | tr ',' '-'
}

loss_name="$(sanitize_name "${loss_features}")"
out_name=${scene_name}_if${input_factor}_tf${target_factor}_gsfm_noemd_${loss_name}_${mix_schedule}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0811-test-fixes/overfit_sr_gsfm_noemd_512}
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
    --gin_param="PointTransformerV3FlowModel.drop_path=${ptv3_drop_path}" \
    --gin_param="PointTransformerV3FlowModel.shuffle_orders=${ptv3_shuffle_orders}" \
    --gin_param="PointTransformerV3FlowModel.shuffle_orders_eval=${ptv3_shuffle_orders_eval}" \
    --gin_param="PointTransformerV3FlowModel.turn_off_bn=${ptv3_turn_off_bn}" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="train2D/build_scheduler.total_step=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="training.image_l1_loss_weight=${image_l1_loss_weight}" \
    --gin_param="training.lpips_loss_weight=${lpips_loss_weight}" \
    --gin_param="loss_mixing.schedule='${mix_schedule}'" \
    --gin_param="matching_fit.total_steps=${matching_steps}" \
    --gin_param="matching_fit.image_per_step=${matching_image_per_step}" \
    --gin_param="matching_fit.image_l1_loss_weight=${matching_l1_weight}" \
    --gin_param="matching_fit.lpips_loss_weight=${matching_lpips_weight}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project2/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project2/ricky/splatformer-data/test-set-512/objaverse/colmap'"

#!/bin/bash

GPU_ID=${GPU_ID:-5}
echo "Using GPU: ${GPU_ID}"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
total_steps=${1:-4000}
save_interval=${2:-4000}
eval_interval=${3:-400}
log_image_interval=${4:-400}
alignment=${5:-emd}
attribute_init=${6:-aligned}
input_resolution=${7:-128}
target_resolution=${8:-512}
mix_schedule=${9:-${MIX_SCHEDULE:-fm-only}}
flow_loss_type=${10:-${FLOW_LOSS_TYPE:-velocity}}
flow_steps=${11:-${FLOW_STEPS:-1}}
flow_noise_std=${12:-${FLOW_NOISE_STD:-0.0}}
image_l1_loss_weight=${13:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
lpips_loss_weight=${14:-${LPIPS_LOSS_WEIGHT:-1.0}}
flow_t_eps=${FLOW_T_EPS:-1e-4}
matching_steps=${MATCHING_STEPS:-3000}
matching_image_per_step=${MATCHING_IMAGE_PER_STEP:-16}
matching_l1_weight=${MATCHING_L1_LOSS_WEIGHT:-1.0}
matching_lpips_weight=${MATCHING_LPIPS_LOSS_WEIGHT:-1.0}
matching_cache_root=${MATCHING_CACHE_ROOT:-/project2/ricky/splatformer-data-to-4x}
force_matching_fit=${FORCE_MATCHING_FIT:-true}
ptv3_drop_path=${PTV3_DROP_PATH:-0.0}
ptv3_shuffle_orders=${PTV3_SHUFFLE_ORDERS:-True}
ptv3_shuffle_orders_eval=${PTV3_SHUFFLE_ORDERS_EVAL:-False}
ptv3_turn_off_bn=${PTV3_TURN_OFF_BN:-True}

out_name=${scene_name}_${alignment}_${attribute_init}_ir${input_resolution}_tr${target_resolution}_gsfm_all_${mix_schedule}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/0820/overfit_sr_gsfm_512}
output_dir=${OUTPUT_DIR:-${output_root}/${out_name}}

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-gsfm.py \
    --output_dir="${output_dir}" \
    --scene_name="${scene_name}" \
    --alignment="${alignment}" \
    --attribute_init="${attribute_init}" \
    --input_resolution="${input_resolution}" \
    --target_resolution="${target_resolution}" \
    --matching_cache_root="${matching_cache_root}" \
    --force_matching_fit="${force_matching_fit}" \
    --gin_param="flow_matching.flow_steps=${flow_steps}" \
    --gin_param="flow_matching.flow_noise_std=${flow_noise_std}" \
    --gin_param="flow_matching.loss_type='${flow_loss_type}'" \
    --gin_param="flow_matching.flow_t_eps=${flow_t_eps}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/overfit/sr_gsfm.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
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
    --gin_param="matching_fit.lpips_loss_weight=${matching_lpips_weight}"

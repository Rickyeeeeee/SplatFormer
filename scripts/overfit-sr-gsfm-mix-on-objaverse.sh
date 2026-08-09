#!/bin/bash

GPU_ID=${GPU_ID:-5}
echo "Using GPU: $GPU_ID"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

total_steps=${1:-4000}
save_interval=${2:-4000}
eval_interval=${3:-400}
log_image_interval=${4:-400}
alignment=${5:-emd}
attribute_init=${6:-aligned}
input_factor=${7:-4}
target_factor=${8:-2}
flow_steps=${9:-${FLOW_STEPS:-1}}
flow_noise_std=${10:-${FLOW_NOISE_STD:-0.0}}
image_l1_loss_weight=${11:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
lpips_loss_weight=${12:-${LPIPS_LOSS_WEIGHT:-1.0}}
mix_schedule=${13:-${MIX_SCHEDULE:-free-range-gs}}
flow_t_eps=${FLOW_T_EPS:-1e-4}

out_name=${scene_name}_${alignment}_${attribute_init}_if${input_factor}_tf${target_factor}_gsfm_mix_${mix_schedule}
output_dir=${OUTPUT_DIR:-/project/ricky/experiments/0806/overfit_sr_gsfm_mix_512/${out_name}}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-gsfm-mix.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --alignment=${alignment} \
    --attribute_init=${attribute_init} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --flow_steps=${flow_steps} \
    --flow_noise_std=${flow_noise_std} \
    --gin_param="flow_matching.flow_t_eps=${flow_t_eps}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/overfit/sr_gsfm_mix.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="training.image_l1_loss_weight=${image_l1_loss_weight}" \
    --gin_param="training.lpips_loss_weight=${lpips_loss_weight}" \
    --gin_param="loss_mixing.schedule='${mix_schedule}'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"

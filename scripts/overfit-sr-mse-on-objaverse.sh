#!/bin/bash

GPU_ID=${GPU_ID:-4}
echo "Using GPU: $GPU_ID"

# for scene_name in $(ls /project/ricky/splatformer-data/test-set-512/colmap)
# do

scene_name=3e288ee8aced4a0797e66d53536112b1

total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
alignment=${5:-emd}
attribute_init=${6:-3dgs}
input_factor=${7:-4}
target_factor=${8:-2}
loss_features=${9:-${LOSS_FEATURES:-means}}
post_activate_loss=${10:-${POST_ACTIVATE_LOSS:-true}}
direct_prediction=${11:-${DIRECT_PREDICTION:-false}}
means_origin_scale=${12:-${MEANS_ORIGIN_SCALE:-1.01}}
conda_env=${CONDA_ENV:-3dgs-sr}

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
post_activate_bit="$(to_bit ${post_activate_loss})"
direct_prediction_bit="$(to_bit ${direct_prediction})"
if [ "${direct_prediction_bit}" = "1" ]; then
    output_features_type=dc
    max_scale_normalized=${MAX_SCALE_NORMALIZED:--1}
else
    output_features_type=res
    max_scale_normalized=${MAX_SCALE_NORMALIZED:-1e-2}
fi

scale_name=$(sanitize_name "${means_origin_scale}")
out_name=${scene_name}_${attribute_init}_if${input_factor}_tf${target_factor}_mse_${loss_name}
output_dir=/project/ricky/experiments/objaverse_splatformer_overfit_sr_mse_512/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-mse.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --alignment=${alignment} \
    --attribute_init=${attribute_init} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --loss_features=${loss_features} \
    --post_activate_loss=${post_activate_loss} \
    --means_origin_scale=${means_origin_scale} \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr_mse.gin \
    --gin_param="FeaturePredictor.output_features_type='${output_features_type}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${max_scale_normalized}" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"
# done

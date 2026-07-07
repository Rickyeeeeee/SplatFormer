#!/bin/bash

GPU_ID=${GPU_ID:-5}
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
means_source=${9:-${MEANS_SOURCE:-gt}}
image_batch_size=${10:-${IMAGE_BATCH_SIZE:--1}}
conda_env=${CONDA_ENV:-3dgs-sr}

out_name=${scene_name}_${alignment}_${attribute_init}_ms${means_source}_ibs${image_batch_size}_if${input_factor}_tf${target_factor}
output_dir=/project/ricky/outputs/objaverse_splatformer_overfit_sr_2stage_512_gsplat/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-2stage.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --means_source=${means_source} \
    --image_batch_size=${image_batch_size} \
    --alignment=${alignment} \
    --attribute_init=${attribute_init} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr_2stage.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"
# done

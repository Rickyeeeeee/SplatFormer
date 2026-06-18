#!/bin/bash

GPU_ID=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '$1 == 0 {print NR-1}' \
    | head -n1)

if [ -z "$GPU_ID" ]; then
    echo "[ERROR] No available GPU found. Exiting."
    exit 1
fi
GPU_ID=5
echo "Using GPU: $GPU_ID"

# for scene_name in $(ls /project/ricky/splatformer-data/test-set-512/colmap)
# do

scene_name=3e288ee8aced4a0797e66d53536112b1

total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
gs_optim_steps=${5:-500}
point_loss_after_activation=${6:-True}

out_name=${scene_name}
output_dir=/project/ricky/outputs/objaverse_splatformer_overfit_sr512_point_act/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-point-loss.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.gs_optim_steps=${gs_optim_steps}" \
    --gin_param="training.point_loss_after_activation=${point_loss_after_activation}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"
# done

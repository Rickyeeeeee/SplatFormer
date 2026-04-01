#!/bin/bash

GPU_ID=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '$1 == 0 {print NR-1}' \
    | head -n1)

if [ -z "$GPU_ID" ]; then
    echo "[ERROR] No available GPU found. Exiting."
    exit 1
fi

echo "Using GPU: $GPU_ID"

scene_name=${1:-first_scene}
total_steps=${2:-1000}
save_interval=${3:-100}
eval_interval=${4:-100}
log_image_interval=${5:-10}

out_name=${scene_name}
output_dir=outputs/objaverse_splatformer_overfit_sr_128/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/sr.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}"

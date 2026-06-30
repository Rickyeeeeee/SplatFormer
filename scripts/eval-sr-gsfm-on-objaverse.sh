#!/bin/bash

GPU_ID=${GPU_ID:-5}
echo "Using GPU: $GPU_ID"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

alignment=${1:-emd}
attribute_init=${2:-3dgs}
input_factor=${3:-4}
target_factor=${4:-2}
# loss_features=${5:-${LOSS_FEATURES:-means,quats,scales,opacities,features_dc}}
loss_features=${5:-${LOSS_FEATURES:-scales,opacities}}
flow_steps=${6:-${FLOW_STEPS:-1}}
flow_noise_std=${7:-${FLOW_NOISE_STD:-0.0}}
post_activate_loss=${8:-${POST_ACTIVATE_LOSS:-false}}
eval_flow_steps=${EVAL_FLOW_STEPS:-1-5}
flow_t_eps=${FLOW_T_EPS:-1e-4}
conda_env=${CONDA_ENV:-3dgs-sr}

if [ -n "$CONDA_PREFIX" ]; then
    current_env=$(basename "$CONDA_PREFIX")
else
    current_env=""
fi
if [ "$current_env" != "$conda_env" ]; then
    echo "Warning: expected conda env '$conda_env', current env is '${current_env:-none}'"
fi

sanitize_name() {
    echo "$1" | tr ',' '-'
}

loss_name="$(sanitize_name "${loss_features}")"
out_name=${scene_name}_${alignment}_${attribute_init}_if${input_factor}_tf${target_factor}_gsfm_${loss_name}_raw_steps${flow_steps}_n${flow_noise_std}
output_dir=${OUTPUT_DIR:-/project/ricky/outputs/objaverse_splatformer_overfit_sr_gsfm_512/${out_name}}
checkpoint=${CHECKPOINT:-${output_dir}/checkpoints/model_last.pth}
eval_output_dir=${EVAL_OUTPUT_DIR:-${output_dir}/eval_flow_steps_${eval_flow_steps}}

CUDA_VISIBLE_DEVICES=$GPU_ID python eval-sr-gsfm.py \
    --output_dir=${output_dir} \
    --checkpoint=${checkpoint} \
    --eval_output_dir=${eval_output_dir} \
    --scene_name=${scene_name} \
    --alignment=${alignment} \
    --attribute_init=${attribute_init} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --loss_features=${loss_features} \
    --flow_space=raw \
    --flow_steps=${flow_steps} \
    --flow_noise_std=${flow_noise_std} \
    --post_activate_loss=${post_activate_loss} \
    --eval_flow_steps=${eval_flow_steps} \
    --gin_param="flow_matching.flow_t_eps=${flow_t_eps}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/overfit/sr_gsfm.gin \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"

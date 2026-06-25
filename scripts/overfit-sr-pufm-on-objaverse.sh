#!/bin/bash

GPU_ID=${GPU_ID:-5}
echo "Using GPU: $GPU_ID"

scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}

total_steps=${1:-2000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
alignment=${5:-nearest}
attribute_init=${6:-3dgs}
input_factor=${7:-4}
target_factor=${8:-2}
flow_space=${9:-bounded}
flow_steps=${10:-5}
flow_noise_std=${11:-0.0}
flow_loss_weight=${12:-0.0}
render_loss_weight=${13:-1.0}
conda_env=${CONDA_ENV:-3dgs-sr}
gt_features_dc=${GT_FEATURES_DC:-true}
gt_features_rest=${GT_FEATURES_REST:-true}
gt_opacities=${GT_OPACITIES:-false}
gt_scales=${GT_SCALES:-true}
gt_quats=${GT_QUATS:-true}

if [ -n "$CONDA_PREFIX" ]; then
    current_env=$(basename "$CONDA_PREFIX")
else
    current_env=""
fi
if [ "$current_env" != "$conda_env" ]; then
    echo "Warning: expected conda env '$conda_env', current env is '${current_env:-none}'"
fi

to_bit() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES) echo 1 ;;
        *) echo 0 ;;
    esac
}

gt_attr_bits="$(to_bit ${gt_features_dc})$(to_bit ${gt_features_rest})$(to_bit ${gt_opacities})$(to_bit ${gt_scales})$(to_bit ${gt_quats})"

out_name=${scene_name}_${alignment}_${attribute_init}_gt${gt_attr_bits}_if${input_factor}_tf${target_factor}_${flow_space}_fs${flow_steps}_n${flow_noise_std}_fw${flow_loss_weight}_rw${render_loss_weight}
output_dir=/project/ricky/outputs/objaverse_splatformer_overfit_sr_pufm_512_gsplat/${out_name}

CUDA_VISIBLE_DEVICES=$GPU_ID python overfit-sr-pufm.py \
    --output_dir=${output_dir} \
    --scene_name=${scene_name} \
    --alignment=${alignment} \
    --attribute_init=${attribute_init} \
    --input_factor=${input_factor} \
    --target_factor=${target_factor} \
    --flow_space=${flow_space} \
    --flow_steps=${flow_steps} \
    --flow_noise_std=${flow_noise_std} \
    --gin_param="flow_matching.flow_loss_weight=${flow_loss_weight}" \
    --gin_param="flow_matching.render_loss_weight=${render_loss_weight}" \
    --gt_features_dc=${gt_features_dc} \
    --gt_features_rest=${gt_features_rest} \
    --gt_opacities=${gt_opacities} \
    --gt_scales=${gt_scales} \
    --gt_quats=${gt_quats} \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/overfit/sr_pufm.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio'" \
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='/project/ricky/splatformer-data/test-set-512/objaverse/colmap'"

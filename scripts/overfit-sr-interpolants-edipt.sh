#!/bin/bash

set -euo pipefail
GPU_ID=${GPU_ID:-3}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
FIT_LR_TO_HR_ROOT=${FIT_LR_TO_HR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
FIT_HR_TO_LR_ROOT=${FIT_HR_TO_LR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
echo "Using GPU: ${GPU_ID}"

# Example: SCENE_MODE=many SCENE_COUNT=4 BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 bash scripts/overfit-sr-interpolants-edipt.sh
scene_mode=${SCENE_MODE:-one}
scene_count=${SCENE_COUNT:-1}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
lr_warmup_steps=${LR_WARMUP_STEPS:-500}
lr_warmup_start_factor=${LR_WARMUP_START_FACTOR:-0.03333333333333333}
total_steps=${1:-4000}
save_interval=${2:-4000}
eval_interval=${3:-500}
log_image_interval=${4:-500}
alignment=${5:-fit_lr_to_hr}
attribute_init=${6:-aligned}
input_resolution=${7:-32}
target_resolution=${8:-128}
mix_schedule=${9:-${MIX_SCHEDULE:-fm-only}}
flow_loss_type=${10:-${FLOW_LOSS_TYPE:-velocity}}
interpolant_type=${INTERPOLANT_TYPE:-linear}
default_flow_steps=50
if [[ "${interpolant_type}" == "encoding_decoding" ]]; then
    default_flow_steps=50
fi
flow_steps=${11:-${FLOW_STEPS:-${default_flow_steps}}}
loss_rollout_steps=${LOSS_ROLLOUT_STEPS:-10}
eval_noise_seed=${EVAL_NOISE_SEED:-0}
fixed_train_noise=${FIXED_TRAIN_NOISE:-False}
train_noise_seed=${TRAIN_NOISE_SEED:-0}
case "${fixed_train_noise}" in
    True|False) ;;
    *)
        echo "Unsupported FIXED_TRAIN_NOISE: ${fixed_train_noise}; expected True or False" >&2
        exit 2
        ;;
esac
flow_noise_std=${12:-${FLOW_NOISE_STD:-1.0}}
image_l1_loss_weight=${13:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
lpips_loss_weight=${14:-${LPIPS_LOSS_WEIGHT:-1.0}}
normalization_variance_floor=${NORMALIZATION_VARIANCE_FLOOR:-1e-8}
quaternion_representation=${QUATERNION_REPRESENTATION:-unit_unstandardized}
if [[ "${quaternion_representation}" != "unit_unstandardized" ]]; then
    echo "EDiPT requires QUATERNION_REPRESENTATION=unit_unstandardized" >&2
    exit 2
fi
quaternion_suffix=_unit_unstandardized
rotation_noise_std=${ROTATION_NOISE_STD:-0.3}
rotation_loss_weight=${ROTATION_LOSS_WEIGHT:-1.0}
gs_statistics_path=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
flow_t_eps=${FLOW_T_EPS:-1e-4}
grid_resolution=${GRID_RESOLUTION:-1536}
random_jitter=${RANDOM_JITTER:-False}
random_rotate=${RANDOM_ROTATE:-False}
rotation_mode=${ROTATION_MODE:-full}
rotation_max_degrees=${ROTATION_MAX_DEGREES:-None}
jitter_max_levels=${JITTER_MAX_LEVELS:-}
case "${random_jitter}" in
    True|False) ;;
    *) echo "RANDOM_JITTER must be True or False" >&2; exit 2 ;;
esac
case "${random_rotate}" in
    True|False) ;;
    *) echo "RANDOM_ROTATE must be True or False" >&2; exit 2 ;;
esac
case "${rotation_mode}" in
    full|gravity_consistent) ;;
    *) echo "ROTATION_MODE must be full or gravity_consistent" >&2; exit 2 ;;
esac

train_noise_suffix=
if [[ "${fixed_train_noise}" == "True" ]]; then
    train_noise_suffix=_fixedtrainnoise${train_noise_seed}
fi
lr_warmup_suffix=
if (( lr_warmup_steps > 0 )); then
    lr_warmup_suffix=_warmup${lr_warmup_steps}
fi
predictor_class=EquivariantGaussianDiPTPredictor
backbone_class=EquivariantGaussianDiPT
model_gin_file=configs/model/edipt_gaussian.gin
predictor_suffix=_edipt
echo "Using predictor: edipt"
run_date=$(date +%m%d)
custom_postfix=${CUSTOM_POSFIX:-run}

scene_label=${scene_name}
if [[ "${scene_mode}" == "many" ]]; then
    scene_label=many_${scene_count}
fi

augmentation_suffix=
if [[ "${random_jitter}" == "True" ]]; then
    augmentation_suffix+=_jitter
fi
if [[ "${random_rotate}" == "True" ]]; then
    if [[ "${rotation_mode}" == "gravity_consistent" ]]; then
        augmentation_suffix+=_gravity_rotate
    else
        augmentation_suffix+=_rotate
    fi
fi

out_name=${scene_label}_\
${alignment}_\
${attribute_init}_\
ir${input_resolution}_\
tr${target_resolution}_\
interpolants_${interpolant_type}_${mix_schedule}_\
noise${flow_noise_std}_rotnoise${rotation_noise_std}_rotweight${rotation_loss_weight}_steps${flow_steps}_seed${eval_noise_seed}${train_noise_suffix}_\
grid${grid_resolution}_\
batch_size${batch_size}${lr_warmup_suffix}${augmentation_suffix}_\
${custom_postfix}${predictor_suffix}${quaternion_suffix}

output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/overfit_sr_interpolants_edipt}
output_dir=${OUTPUT_DIR:-${output_root}/${out_name}}

# EDIPT_* and GS_* values are Gin literals; input width follows the SH degree.
network_gin_args=()
for network in "${backbone_class}" "${predictor_class}"; do
    case "${network}" in
        EquivariantGaussianDiPT)
            prefix=EDIPT
            parameters=(depth channels num_head patch_size order mlp_ratio frequency_embedding_size geometry_channels geometry_length_scale attn_drop proj_drop drop_path shuffle_orders)
            ;;
        EquivariantGaussianDiPTPredictor)
            prefix=GS
            parameters=(sh_degree input_feat_to_mlp output_head_width zeroinit)
            ;;
    esac
    for parameter in "${parameters[@]}"; do
        env_name=${prefix}_${parameter^^}
        value=${!env_name:-}
        if [[ -n "${value}" ]]; then
            network_gin_args+=("--gin_param=${network}.${parameter}=${value}")
        fi
    done
done
network_gin_args+=("--gin_param=${predictor_class}.grid_resolution=${grid_resolution}")
augmentation_gin_args=(
    "--gin_param=training_augmentation.random_jitter=${random_jitter}"
    "--gin_param=training_augmentation.random_rotate=${random_rotate}"
    "--gin_param=training_augmentation.rotation_max_degrees=${rotation_max_degrees}"
    "--gin_param=training_augmentation.rotation_mode='${rotation_mode}'"
)
if [[ -n "${jitter_max_levels}" ]]; then
    augmentation_gin_args+=("--gin_param=training_augmentation.jitter_max_levels=${jitter_max_levels}")
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-interpolants-edipt.py \
    --output_dir="${output_dir}" \
    --scene_name="${scene_name}" \
    --scene_mode="${scene_mode}" \
    --scene_count="${scene_count}" \
    --batch_size="${batch_size}" \
    --grad_accum_steps="${grad_accum_steps}" \
    --alignment="${alignment}" \
    --attribute_init="${attribute_init}" \
    --gin_param="flow_matching.quaternion_representation='${quaternion_representation}'" \
    --gin_param="flow_matching.rotation_noise_std=${rotation_noise_std}" \
    --gin_param="flow_matching.rotation_loss_weight=${rotation_loss_weight}" \
    --gin_param="flow_matching.normalization_variance_floor=${normalization_variance_floor}" \
    --gin_param="flow_matching.gs_statistics_path='${gs_statistics_path}'" \
    --gin_param="flow_matching.interpolant_type='${interpolant_type}'" \
    --gin_param="flow_matching.loss_rollout_steps=${loss_rollout_steps}" \
    --gin_param="flow_matching.eval_noise_seed=${eval_noise_seed}" \
    --gin_param="flow_matching.fixed_train_noise=${fixed_train_noise}" \
    --gin_param="flow_matching.train_noise_seed=${train_noise_seed}" \
    --gin_param="flow_matching.flow_steps=${flow_steps}" \
    --gin_param="flow_matching.flow_noise_std=${flow_noise_std}" \
    --gin_param="flow_matching.loss_type='${flow_loss_type}'" \
    --gin_param="flow_matching.flow_t_eps=${flow_t_eps}" \
    --gin_file="${model_gin_file}" \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/overfit/sr_interpolants_edipt.gin \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="test_fit_lr_to_hr_root='${FIT_LR_TO_HR_ROOT}'" \
    --gin_param="test_fit_hr_to_lr_root='${FIT_HR_TO_LR_ROOT}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${input_resolution}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${target_resolution}" \
    --gin_param="SplatFactoSRDataset.load_src_gs=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_gs=True" \
    --gin_param="SplatFactoSRDataset.load_src_images=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_images=True" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="train2D/build_scheduler.total_step=${total_steps}" \
    --gin_param="train2D/build_scheduler.warmup_step=${lr_warmup_steps}" \
    --gin_param="train2D/build_scheduler.warmup_start_factor=${lr_warmup_start_factor}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="training.image_l1_loss_weight=${image_l1_loss_weight}" \
    --gin_param="training.lpips_loss_weight=${lpips_loss_weight}" \
    --gin_param="loss_mixing.schedule='${mix_schedule}'" \
    "${augmentation_gin_args[@]}" \
    "${network_gin_args[@]}"

#!/bin/bash

set -euo pipefail
GPU_ID=${GPU_ID:-3}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
FIT_LR_TO_HR_ROOT=${FIT_LR_TO_HR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
FIT_HR_TO_LR_ROOT=${FIT_HR_TO_LR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
echo "Using GPU: ${GPU_ID}"

# Example: SCENE_MODE=many SCENE_COUNT=4 BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 bash scripts/overfit-sr-interpolants.sh
scene_mode=${SCENE_MODE:-one}
scene_count=${SCENE_COUNT:-1}
batch_size=${BATCH_SIZE:-16}
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
gs_statistics_path=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
flow_t_eps=${FLOW_T_EPS:-1e-4}
ptv3_drop_path=${PTV3_DROP_PATH:-0.0}
ptv3_shuffle_orders=${PTV3_SHUFFLE_ORDERS:-True}
ptv3_shuffle_orders_eval=${PTV3_SHUFFLE_ORDERS_EVAL:-False}
ptv3_turn_off_bn=${PTV3_TURN_OFF_BN:-True}
grid_resolution=${GRID_RESOLUTION:-1536}
random_jitter=${RANDOM_JITTER:-False}
random_rotate=${RANDOM_ROTATE:-False}
jitter_max_levels=${JITTER_MAX_LEVELS:-}
case "${random_jitter}" in
    True|False) ;;
    *) echo "RANDOM_JITTER must be True or False" >&2; exit 2 ;;
esac
case "${random_rotate}" in
    True|False) ;;
    *) echo "RANDOM_ROTATE must be True or False" >&2; exit 2 ;;
esac

train_noise_suffix=
if [[ "${fixed_train_noise}" == "True" ]]; then
    train_noise_suffix=_fixedtrainnoise${train_noise_seed}
fi
lr_warmup_suffix=
if (( lr_warmup_steps > 0 )); then
    lr_warmup_suffix=_warmup${lr_warmup_steps}
fi
# predictor=${PREDICTOR:-ptv3}
predictor=${PREDICTOR:-dipt}
case "${predictor}" in
    ptv3)
        predictor_class=GSFlowPredictor
        backbone_class=PointTransformerV3FlowModel
        model_gin_file=configs/model/ptv3_flow.gin
        predictor_suffix=
        ;;
    dipt)
        predictor_class=DiffusionGaussianPredictor
        backbone_class=DiffusionGaussianTransformer
        model_gin_file=configs/model/dipt_gaussian.gin
        predictor_suffix=_dipt
        ;;
    *)
        echo "Unsupported PREDICTOR: ${predictor}; expected ptv3 or dipt" >&2
        exit 2
        ;;
esac
echo "Using predictor: ${predictor}"
run_date=$(date +%m%d)
custom_postfix=${CUSTOM_POSFIX:-run}

scene_label=${scene_name}
if [[ "${scene_mode}" == "many" ]]; then
    scene_label=many_${scene_count}
fi

out_name=${scene_label}_\
${alignment}_\
${attribute_init}_\
ir${input_resolution}_\
tr${target_resolution}_\
interpolants_${interpolant_type}_${mix_schedule}_\
noise${flow_noise_std}_steps${flow_steps}_seed${eval_noise_seed}${train_noise_suffix}_\
grid${grid_resolution}}_\
batch_size${batch_size}${lr_warmup_suffix}_\
${custom_postfix}${predictor_suffix}

output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/overfit_sr_interpolants_512_dipt}
output_dir=${OUTPUT_DIR:-${output_root}/${out_name}}

# Optional architecture overrides retain Gin/model defaults when unset or empty.
# Use Gin literals for tuples/dicts and True/False for booleans; strings need no inner quotes.
# Keep stage counts, channel widths, and head counts compatible; input channels are derived.
# The output head supports mlp-relu; DiPT embeds continuous time in its transformer blocks.
# Fourier example: GS_FOURIER_INPUT_FEATURES="['means', 'scales']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 4}" CUSTOM_POSFIX=fourier bash scripts/overfit-sr-interpolants.sh
# Set GS_FOURIER_INCLUDE_RAW=False, GS_FOURIER_LOG_SAMPLING=False, or GS_FOURIER_MAX_FREQUENCY_LOG2="{'means': 5}" as needed.
# Example: PTV3_ENC_CHANNELS='(32, 64, 128, 256, 512)' PTV3_DEC_CHANNELS='(64, 64, 128, 256)' GS_OUTPUT_HEAD_WIDTH=64 GS_OUTPUT_HEAD_NLAYER=2 CUSTOM_POSFIX=small bash scripts/overfit-sr-interpolants.sh
network_gin_args=()
for network in "${backbone_class}" "${predictor_class}"; do
    case "${network}" in
        PointTransformerV3FlowModel)
            prefix=PTV3
            parameters=(enc_dim output_dim enc_channels dec_channels enc_depths dec_depths enc_num_head dec_num_head stride embedding_type T_dim enable_flash pdnorm_bn pdnorm_ln pretrained_ckpt)
            ;;
        DiffusionGaussianTransformer)
            prefix=DIPT
            parameters=(order depth channels num_head patch_size mlp_ratio frequency_embedding_size qkv_bias qk_scale attn_drop proj_drop drop_path pre_norm shuffle_orders shuffle_orders_eval enable_rpe enable_flash upcast_attention upcast_softmax pdnorm_bn pdnorm_ln pdnorm_decouple pdnorm_adaptive pdnorm_affine pdnorm_conditions pdnorm_condition)
            ;;
        GSFlowPredictor|DiffusionGaussianPredictor)
            prefix=GS
            parameters=(output_head_nlayer output_head_width output_head_type input_feat_to_mlp zeroinit res_feature_activation quat_residual_mode fourier_input_features fourier_num_frequencies fourier_include_raw fourier_log_sampling fourier_max_frequency_log2)
            ;;
    esac
    for parameter in "${parameters[@]}"; do
        env_name=${prefix}_${parameter^^}
        value=${!env_name:-}
        if [[ -n "${value}" ]]; then
            case "${parameter}" in
                embedding_type|pretrained_ckpt|output_head_type|quat_residual_mode|pdnorm_condition)
                    # Escape ordinary strings as Gin string literals, including checkpoint paths.
                    value=${value//\\/\\\\}
                    value=${value//\'/\\\'}
                    value=${value//$'\n'/\\n}
                    value=${value//$'\r'/\\r}
                    value=${value//$'\t'/\\t}
                    value="'${value}'"
                    ;;
            esac
            network_gin_args+=("--gin_param=${network}.${parameter}=${value}")
        fi
    done
done
if [[ "${predictor}" == "ptv3" ]]; then
    network_gin_args+=(
        "--gin_param=PointTransformerV3FlowModel.drop_path=${ptv3_drop_path}"
        "--gin_param=PointTransformerV3FlowModel.shuffle_orders=${ptv3_shuffle_orders}"
        "--gin_param=PointTransformerV3FlowModel.shuffle_orders_eval=${ptv3_shuffle_orders_eval}"
        "--gin_param=PointTransformerV3FlowModel.turn_off_bn=${ptv3_turn_off_bn}"
    )
fi
network_gin_args+=("--gin_param=${predictor_class}.grid_resolution=${grid_resolution}")
augmentation_gin_args=(
    "--gin_param=training_augmentation.random_jitter=${random_jitter}"
    "--gin_param=training_augmentation.random_rotate=${random_rotate}"
)
if [[ -n "${jitter_max_levels}" ]]; then
    augmentation_gin_args+=("--gin_param=training_augmentation.jitter_max_levels=${jitter_max_levels}")
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python overfit-sr-interpolants.py \
    --output_dir="${output_dir}" \
    --scene_name="${scene_name}" \
    --predictor="${predictor}" \
    --scene_mode="${scene_mode}" \
    --scene_count="${scene_count}" \
    --batch_size="${batch_size}" \
    --grad_accum_steps="${grad_accum_steps}" \
    --alignment="${alignment}" \
    --attribute_init="${attribute_init}" \
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
    --gin_file=configs/overfit/sr_interpolants.gin \
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

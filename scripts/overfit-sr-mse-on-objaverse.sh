#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-4}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
FIT_LR_TO_HR_ROOT=${FIT_LR_TO_HR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}
FIT_HR_TO_LR_ROOT=${FIT_HR_TO_LR_ROOT:-${DATASET_ROOT}/test-set-4x-up/objaverse}

scene_mode=${SCENE_MODE:-many}
scene_count=${SCENE_COUNT:-18}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
scene_name=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
total_steps=${1:-1000}
save_interval=${2:-200}
eval_interval=${3:-200}
log_image_interval=${4:-200}
alignment=${5:-emd}
attribute_init=${6:-3dgs}
input_resolution=${7:-128}
target_resolution=${8:-512}
post_activate_loss=${9:-${POST_ACTIVATE_LOSS:-true}}
direct_prediction=${10:-${DIRECT_PREDICTION:-false}}
gs_statistics_path=${11:-${GS_STATISTICS_PATH:-}}
image_l1_loss_weight=${12:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
lpips_loss_weight=${13:-${LPIPS_LOSS_WEIGHT:-1.0}}

grid_resolution=${GRID_RESOLUTION:-384}
run_date=$(date +%m%d)
custom_postfix=${CUSTOM_POSFIX:-run}

to_bit() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES) echo 1 ;;
        *) echo 0 ;;
    esac
}

case "${scene_mode}" in
    one|many) ;;
    *) echo "Unsupported SCENE_MODE=${scene_mode}. Use one or many." >&2; exit 1 ;;
esac
case "${alignment}" in
    emd|random|fit_lr_to_hr|fit_hr_to_lr) ;;
    *) echo "Unsupported alignment '${alignment}'. Use emd, random, fit_lr_to_hr, or fit_hr_to_lr." >&2; exit 1 ;;
esac
case "${attribute_init}" in
    aligned|3dgs) ;;
    *) echo "Unsupported attribute_init '${attribute_init}'. Use aligned or 3dgs." >&2; exit 1 ;;
esac
if (( scene_count < 1 || batch_size < 1 || grad_accum_steps < 1 || grad_accum_steps > batch_size )); then
    echo "Require SCENE_COUNT>=1 and 1<=GRAD_ACCUM_STEPS<=BATCH_SIZE." >&2
    exit 1
fi

post_activate_bit=$(to_bit "${post_activate_loss}")
direct_prediction_bit=$(to_bit "${direct_prediction}")
if [[ -n "${gs_statistics_path}" && "${post_activate_bit}" == "1" ]]; then
    echo "GS_STATISTICS_PATH requires POST_ACTIVATE_LOSS=false" >&2
    exit 1
fi
if [[ "${direct_prediction_bit}" == "1" ]]; then
    output_features_type=dc
    max_scale_normalized=${MAX_SCALE_NORMALIZED:--1}
else
    output_features_type=res
    max_scale_normalized=${MAX_SCALE_NORMALIZED:-1e-2}
fi

gs_statistics_args=()
stats_suffix=""
if [[ -n "${gs_statistics_path}" ]]; then
    gs_statistics_args=(--gs_statistics_path="${gs_statistics_path}")
    stats_suffix=_gsnorm
fi
scene_label=${scene_name}
if [[ "${scene_mode}" == "many" ]]; then
    scene_label=many_${scene_count}
fi
out_name=${scene_label}_${alignment}_${attribute_init}_ir${input_resolution}_tr${target_resolution}_grid${grid_resolution}_batch${batch_size}_l1${image_l1_loss_weight}_lpips${lpips_loss_weight}_${custom_postfix}${stats_suffix}
output_root=${OUTPUT_ROOT:-/project2/ricky/experiments/${run_date}/overfit_sr_mse}
output_dir=${OUTPUT_DIR:-${output_root}/${out_name}}

# Optional overrides use Gin literals; ordinary string parameters are escaped below.
network_gin_args=()
for network in PointTransformerV3Model FeaturePredictor; do
    case "${network}" in
        PointTransformerV3Model)
            prefix=PTV3
            parameters=(enc_dim output_dim enc_channels dec_channels enc_depths dec_depths enc_num_head dec_num_head stride embedding_type enable_flash pdnorm_bn pdnorm_ln pretrained_ckpt drop_path shuffle_orders shuffle_orders_eval turn_off_bn)
            ;;
        FeaturePredictor)
            prefix=GS
            parameters=(output_head_nlayer output_head_width output_head_type input_feat_to_mlp input_embed_to_mlp zeroinit res_feature_activation quat_residual_mode resume_ckpt fourier_input_features fourier_num_frequencies fourier_include_raw fourier_log_sampling fourier_max_frequency_log2)
            ;;
    esac
    for parameter in "${parameters[@]}"; do
        env_name=${prefix}_${parameter^^}
        value=${!env_name:-}
        if [[ -n "${value}" ]]; then
            case "${parameter}" in
                embedding_type|pretrained_ckpt|output_head_type|quat_residual_mode|resume_ckpt)
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

echo "Using GPU: ${GPU_ID}"
echo "Scenes: ${scene_mode} (${scene_count})"
echo "Batch: ${batch_size}, accumulation: ${grad_accum_steps}"
echo "Output: ${output_dir}"

TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" overfit-sr-mse.py \
    --output_dir="${output_dir}" \
    --scene_name="${scene_name}" \
    --scene_mode="${scene_mode}" \
    --scene_count="${scene_count}" \
    --batch_size="${batch_size}" \
    --grad_accum_steps="${grad_accum_steps}" \
    --alignment="${alignment}" \
    --attribute_init="${attribute_init}" \
    --post_activate_loss="${post_activate_loss}" \
    "${gs_statistics_args[@]}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/overfit/sr_mse.gin \
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
    --gin_param="FeaturePredictor.output_features_type='${output_features_type}'" \
    --gin_param="FeaturePredictor.max_scale_normalized=${max_scale_normalized}" \
    --gin_param="FeaturePredictor.grid_resolution=${grid_resolution}" \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="train2D/build_scheduler.total_step=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="training.image_l1_loss_weight=${image_l1_loss_weight}" \
    --gin_param="training.lpips_loss_weight=${lpips_loss_weight}" \
    "${network_gin_args[@]}"

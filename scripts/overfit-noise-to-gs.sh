#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-0}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-}
TARGET_GS_ROOT=${TARGET_GS_ROOT:-}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
SCENE_MODE=${SCENE_MODE:-one}
SCENE_COUNT=${SCENE_COUNT:-1}
GS_COUNT=${GS_COUNT:-1000}
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
PREDICTOR=${PREDICTOR:-dit}
GS_COORDINATE_MODE=${GS_COORDINATE_MODE:-morton_rank}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
INPUT_RESOLUTION=${INPUT_RESOLUTION:-32}
TARGET_RESOLUTION=${TARGET_RESOLUTION:-128}
TOTAL_STEPS=${TOTAL_STEPS:-10000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
FLOW_STEPS=${FLOW_STEPS:-50}
FLOW_COUPLING=${FLOW_COUPLING:-random}
FLOW_T_EPS=${FLOW_T_EPS:-1e-4}
EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-0}
EVAL_MAX_VIEWS=${EVAL_MAX_VIEWS:-0}
RANDOM_ROTATE=${RANDOM_ROTATE:-False}
ROTATION_MODE=${ROTATION_MODE:-gravity_consistent}
ROTATION_MAX_DEGREES=${ROTATION_MAX_DEGREES:-None}
ROTATION_PIVOT=${ROTATION_PIVOT:-"(0.5, 0.5, 0.5)"}
EVAL_ROTATION_SEED=${EVAL_ROTATION_SEED:-0}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
LR_WARMUP_START_FACTOR=${LR_WARMUP_START_FACTOR:-0.03333333333333333}

if [[ "${PREDICTOR}" != "dit" ]]; then
    echo "Unsupported PREDICTOR: ${PREDICTOR}; expected dit" >&2
    exit 2
fi
predictor_class=GaussianDiTPredictor
backbone_class=GaussianDiT
model_gin_file=configs/model/dit_gaussian.gin

scene_label=${SCENE_NAME}
if [[ "${SCENE_MODE}" == "many" ]]; then
    scene_label=many_${SCENE_COUNT}
fi
coordinate_label=_${GS_COORDINATE_MODE}
case "${RANDOM_ROTATE}" in
    true|True) RANDOM_ROTATE=True ;;
    false|False) RANDOM_ROTATE=False ;;
    *) echo "RANDOM_ROTATE must be True or False" >&2; exit 2 ;;
esac
load_target_images=True
if [[ "${RANDOM_ROTATE}" == "True" ]] || [[ "${SCENE_MODE}" == "many" && "${SCENE_COUNT}" -gt 1 ]]; then
    load_target_images=False
fi
rotation_label=
if [[ "${RANDOM_ROTATE}" == "True" ]]; then
    rotation_label=_rot_${ROTATION_MODE}
    if [[ "${ROTATION_MAX_DEGREES}" != "None" ]]; then
        rotation_label+=_deg${ROTATION_MAX_DEGREES}
    fi
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-output_overfit_noise_to_gs}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${scene_label}_${PREDICTOR}${coordinate_label}${rotation_label}_hr${TARGET_RESOLUTION}_n${GS_COUNT}}

# Forward architecture-specific Gin overrides without changing model defaults.
network_gin_args=()
if [[ -n "${GS_FEATURE_LOSS_WEIGHTS:-}" ]]; then
    network_gin_args+=("--gin_param=flow_matching.feature_loss_weights=${GS_FEATURE_LOSS_WEIGHTS}")
fi
for network in "${backbone_class}" "${predictor_class}"; do
    case "${network}" in
        GaussianDiT)
            prefix=DIT
            parameters=(depth channels num_heads patch_size mlp_ratio frequency_embedding_size)
            ;;
        GaussianDiTPredictor)
            prefix=GS
            parameters=(sh_degree coordinate_mode fourier_input_features fourier_num_frequencies fourier_include_raw fourier_log_sampling fourier_max_frequency_log2)
            ;;
    esac
    for parameter in "${parameters[@]}"; do
        env_name=${prefix}_${parameter^^}
        value=${!env_name:-}
        if [[ -n "${value}" ]]; then
            case "${parameter}" in
                coordinate_mode) value="'${value}'" ;;
            esac
            network_gin_args+=("--gin_param=${network}.${parameter}=${value}")
        fi
    done
done

network_gin_args+=("--gin_param=GaussianDiTPredictor.max_gaussians=${GS_COUNT}")

CUDA_VISIBLE_DEVICES="${GPU_ID}" python overfit-noise-to-gs.py \
    --output_dir="${OUTPUT_DIR}" \
    --scene_name="${SCENE_NAME}" \
    --source_gs_root="${SOURCE_GS_ROOT}" \
    --target_gs_root="${TARGET_GS_ROOT}" \
    --predictor="${PREDICTOR}" \
    --scene_mode="${SCENE_MODE}" \
    --scene_count="${SCENE_COUNT}" \
    --batch_size="${BATCH_SIZE}" \
    --grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    --eval_max_views="${EVAL_MAX_VIEWS}" \
    --gin_file="${model_gin_file}" \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/overfit/noise_to_gs.gin \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.max_gs_num=${GS_COUNT}" \
    --gin_param="SplatFactoSRDataset.load_src_gs=False" \
    --gin_param="SplatFactoSRDataset.load_src_images=False" \
    --gin_param="SplatFactoSRDataset.load_tgt_gs=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_images=${load_target_images}" \
    --gin_param="flow_matching.gs_statistics_path='${GS_STATISTICS_PATH}'" \
    --gin_param="flow_matching.flow_steps=${FLOW_STEPS}" \
    --gin_param="flow_matching.coupling='${FLOW_COUPLING}'" \
    --gin_param="flow_matching.flow_t_eps=${FLOW_T_EPS}" \
    --gin_param="flow_matching.eval_noise_seed=${EVAL_NOISE_SEED}" \
    --gin_param="training_augmentation.random_rotate=${RANDOM_ROTATE}" \
    --gin_param="training_augmentation.rotation_mode='${ROTATION_MODE}'" \
    --gin_param="training_augmentation.rotation_max_degrees=${ROTATION_MAX_DEGREES}" \
    --gin_param="training_augmentation.rotation_pivot=${ROTATION_PIVOT}" \
    --gin_param="training_augmentation.eval_rotation_seed=${EVAL_ROTATION_SEED}" \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.log_interval=${LOG_INTERVAL}" \
    --gin_param="train2D/build_scheduler.total_step=${TOTAL_STEPS}" \
    --gin_param="train2D/build_scheduler.warmup_step=${LR_WARMUP_STEPS}" \
    --gin_param="train2D/build_scheduler.warmup_start_factor=${LR_WARMUP_START_FACTOR}" \
    "${network_gin_args[@]}"

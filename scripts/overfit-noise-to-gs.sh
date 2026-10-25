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

case "${PREDICTOR}" in
    ptv3)
        predictor_class=GSFlowPredictor
        backbone_class=PointTransformerV3FlowModel
        model_gin_file=configs/model/ptv3_flow.gin
        ;;
    dipt)
        predictor_class=DiffusionGaussianPredictor
        backbone_class=DiffusionGaussianTransformer
        model_gin_file=configs/model/dipt_gaussian.gin
        ;;
    gdpt)
        predictor_class=DiffusionGaussianPredictor
        backbone_class=GlobalDiffusionPointTransformer
        model_gin_file=configs/model/gdpt_gaussian.gin
        ;;
    dit)
        predictor_class=GaussianDiTPredictor
        backbone_class=GaussianDiT
        model_gin_file=configs/model/dit_gaussian.gin
        ;;
    *)
        echo "Unsupported PREDICTOR: ${PREDICTOR}; expected ptv3, dipt, gdpt, or dit" >&2
        exit 2
        ;;
esac

scene_label=${SCENE_NAME}
if [[ "${SCENE_MODE}" == "many" ]]; then
    scene_label=many_${SCENE_COUNT}
fi
coordinate_label=
if [[ "${PREDICTOR}" == "dit" ]]; then
    coordinate_label=_${GS_COORDINATE_MODE}
fi
case "${RANDOM_ROTATE}" in
    true|True) RANDOM_ROTATE=True ;;
    false|False) RANDOM_ROTATE=False ;;
    *) echo "RANDOM_ROTATE must be True or False" >&2; exit 2 ;;
esac
rotation_label=
if [[ "${RANDOM_ROTATE}" == "True" ]]; then
    rotation_label=_rot_${ROTATION_MODE}
    if [[ "${ROTATION_MAX_DEGREES}" != "None" ]]; then
        rotation_label+=_deg${ROTATION_MAX_DEGREES}
    fi
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-output_overfit_noise_to_gs}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${scene_label}_${PREDICTOR}${coordinate_label}${rotation_label}_hr${TARGET_RESOLUTION}_n1000}

# Forward architecture-specific Gin overrides without changing model defaults.
network_gin_args=()
for network in "${backbone_class}" "${predictor_class}"; do
    case "${network}" in
        PointTransformerV3FlowModel)
            prefix=PTV3
            parameters=(enc_dim output_dim enc_channels dec_channels enc_depths dec_depths enc_num_head dec_num_head stride enable_flash drop_path)
            ;;
        DiffusionGaussianTransformer)
            prefix=DIPT
            parameters=(depth channels num_head patch_size order mlp_ratio frequency_embedding_size enable_flash partition_mode)
            ;;
        GlobalDiffusionPointTransformer)
            prefix=GDPT
            parameters=(depth channels num_head mlp_ratio frequency_embedding_size attention_backend use_sparse_conv)
            ;;
        GaussianDiT)
            prefix=DIT
            parameters=(depth channels num_heads mlp_ratio frequency_embedding_size)
            ;;
        GSFlowPredictor|DiffusionGaussianPredictor)
            prefix=GS
            parameters=(sh_degree output_head_nlayer output_head_width input_feat_to_mlp zeroinit fourier_input_features fourier_num_frequencies)
            ;;
        GaussianDiTPredictor)
            prefix=GS
            parameters=(sh_degree coordinate_mode)
            ;;
    esac
    for parameter in "${parameters[@]}"; do
        env_name=${prefix}_${parameter^^}
        value=${!env_name:-}
        if [[ -n "${value}" ]]; then
            case "${parameter}" in
                attention_backend|partition_mode|coordinate_mode) value="'${value}'" ;;
            esac
            network_gin_args+=("--gin_param=${network}.${parameter}=${value}")
        fi
    done
done

if [[ "${PREDICTOR}" == "ptv3" ]]; then
    network_gin_args+=("--gin_param=GSFlowPredictor.shift_negative_grid_coords=True")
elif [[ "${PREDICTOR}" == "dipt" || "${PREDICTOR}" == "gdpt" ]]; then
    network_gin_args+=("--gin_param=DiffusionGaussianPredictor.shift_negative_grid_coords=True")
fi

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
    --gin_param="SplatFactoSRDataset.max_gs_num=1000" \
    --gin_param="SplatFactoSRDataset.load_src_gs=False" \
    --gin_param="SplatFactoSRDataset.load_src_images=False" \
    --gin_param="SplatFactoSRDataset.load_tgt_gs=True" \
    --gin_param="SplatFactoSRDataset.load_tgt_images=True" \
    --gin_param="flow_matching.gs_statistics_path='${GS_STATISTICS_PATH}'" \
    --gin_param="flow_matching.flow_steps=${FLOW_STEPS}" \
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

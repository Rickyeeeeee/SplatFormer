#!/bin/bash
set -euo pipefail

# Example: GPU_IDS=0,1 NGPUS=2 BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 bash scripts/train-sr-gsfm-on-objaverse.sh
GPU_ID=${GPU_ID:-8}
GPU_IDS=${GPU_IDS:-${GPU_ID}}
NGPUS=${NGPUS:-1}
MASTER_PORT=${MASTER_PORT:-29519}
# Workers are per GPU; prefetch factor counts microbatches per worker.
NUM_WORKERS=${NUM_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
PIN_MEMORY=${PIN_MEMORY:-true}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
# Classify scenes by capped CSV Gaussian counts before constructing microbatches.
SCENE_SAMPLING=${SCENE_SAMPLING:-big_small}
BIG_SCENE_THRESHOLD=${BIG_SCENE_THRESHOLD:-25000}
case "${SCENE_SAMPLING}" in
    random|big_small|avoid_big) ;;
    *) echo "Unsupported SCENE_SAMPLING=${SCENE_SAMPLING}" >&2; exit 1 ;;
esac
# Opt in to optimizer-state sharding; single-GPU runs retain the ordinary optimizer.
ZERO_OPTIMIZER=${ZERO_OPTIMIZER:-false}
case "${ZERO_OPTIMIZER,,}" in
    true) ZERO_OPTIMIZER_GIN=True ;;
    false) ZERO_OPTIMIZER_GIN=False ;;
    *) echo "Unsupported ZERO_OPTIMIZER=${ZERO_OPTIMIZER}; expected true or false" >&2; exit 1 ;;
esac
TOTAL_STEPS=${1:-1000000}
SAVE_INTERVAL=${2:-5000}
EVAL_INTERVAL=${3:-5000}
LOG_IMAGE_INTERVAL=${4:-1000}
ALIGNMENT=${5:-${ALIGNMENT:-fit_lr_to_hr}}
ATTRIBUTE_INIT=${6:-${ATTRIBUTE_INIT:-aligned}}
INPUT_RESOLUTION=${7:-${INPUT_RESOLUTION:-32}}
TARGET_RESOLUTION=${8:-${TARGET_RESOLUTION:-128}}
MIX_SCHEDULE=${9:-${MIX_SCHEDULE:-fm-only}}
FLOW_LOSS_TYPE=${10:-${FLOW_LOSS_TYPE:-velocity}}
FLOW_STEPS=${11:-${FLOW_STEPS:-10}}
FLOW_NOISE_STD=${12:-${FLOW_NOISE_STD:-0.0}}
IMAGE_L1_LOSS_WEIGHT=${13:-${IMAGE_L1_LOSS_WEIGHT:-1.0}}
LPIPS_LOSS_WEIGHT=${14:-${LPIPS_LOSS_WEIGHT:-1.0}}

run_date=$(date +%m%d)
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/${run_date}-gpu7/objaverse_sr_gsfm_${INPUT_RESOLUTION}to${TARGET_RESOLUTION}_${ALIGNMENT}_${MIX_SCHEDULE}}

case "${MIX_SCHEDULE}" in
    linear|free-range-gs|fm-only) ;;
    *) echo "Unsupported MIX_SCHEDULE=${MIX_SCHEDULE}" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --rdzv-endpoint="localhost:${MASTER_PORT}" \
    train-sr-gsfm.py \
    --num_workers="${NUM_WORKERS}" \
    --prefetch_factor="${PREFETCH_FACTOR}" \
    --pin_memory="${PIN_MEMORY}" \
    --batch_size="${BATCH_SIZE}" \
    --scene_sampling="${SCENE_SAMPLING}" \
    --big_scene_threshold="${BIG_SCENE_THRESHOLD}" \
    --grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    --output_dir="${OUTPUT_DIR}" \
    --alignment="${ALIGNMENT}" \
    --attribute_init="${ATTRIBUTE_INIT}" \
    --gin_file=configs/model/ptv3_flow.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/train/sr_gsfm.gin \
    --gin_param="GSFlowPredictor.grid_resolution=1536" \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    --gin_param="flow_matching.gs_statistics_path='${GS_STATISTICS_PATH}'" \
    --gin_param="flow_matching.flow_steps=${FLOW_STEPS}" \
    --gin_param="flow_matching.flow_noise_std=${FLOW_NOISE_STD}" \
    --gin_param="flow_matching.loss_type='${FLOW_LOSS_TYPE}'" \
    --gin_param="loss_mixing.schedule='${MIX_SCHEDULE}'" \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}" \
    --gin_param="training.image_l1_loss_weight=${IMAGE_L1_LOSS_WEIGHT}" \
    --gin_param="training.lpips_loss_weight=${LPIPS_LOSS_WEIGHT}" \
    --gin_param="train2D/build_optimizer.use_zero=${ZERO_OPTIMIZER_GIN}" \
    --gin_param="train2D/build_scheduler.total_step=${TOTAL_STEPS}"

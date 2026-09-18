#!/bin/bash
set -euo pipefail

NGPUS=${NGPUS:-1}
GPU_IDS=${GPU_IDS:-}
MASTER_PORT=${MASTER_PORT:-29518}
# Workers are per GPU; prefetch factor counts microbatches per worker.
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-true}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
# Classify scenes by capped CSV Gaussian counts before constructing microbatches.
SCENE_SAMPLING=${SCENE_SAMPLING:-big_small}
BIG_SCENE_THRESHOLD=${BIG_SCENE_THRESHOLD:-25000}
case "${SCENE_SAMPLING}" in
    random|big_small|avoid_big) ;;
    *) echo "Unsupported SCENE_SAMPLING=${SCENE_SAMPLING}" >&2; exit 1 ;;
esac
TOTAL_STEPS=${1:-100000}
SAVE_INTERVAL=${2:-1000}
EVAL_INTERVAL=${3:-1000}
LOG_IMAGE_INTERVAL=${4:-1000}
INPUT_RESOLUTION=${5:-${INPUT_RESOLUTION:-32}}
TARGET_RESOLUTION=${6:-${TARGET_RESOLUTION:-128}}
run_date=$(date +%m%d)
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/${run_date}-gpu7/objaverse_splatformer_sr_${INPUT_RESOLUTION}to${TARGET_RESOLUTION}}

DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}

if [[ -n "${GPU_IDS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --rdzv-endpoint="localhost:${MASTER_PORT}" \
    train-sr.py \
    --num_workers="${NUM_WORKERS}" \
    --prefetch_factor="${PREFETCH_FACTOR}" \
    --pin_memory="${PIN_MEMORY}" \
    --batch_size="${BATCH_SIZE}" \
    --scene_sampling="${SCENE_SAMPLING}" \
    --big_scene_threshold="${BIG_SCENE_THRESHOLD}" \
    --grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    --output_dir="${OUTPUT_DIR}" \
    --input_resolution="${INPUT_RESOLUTION}" \
    --target_resolution="${TARGET_RESOLUTION}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/train/sr.gin \
    --gin_param="dataset_root=\"${DATASET_ROOT}\"" \
    --gin_param="train_scene_list=\"${TRAIN_SCENE_LIST}\"" \
    --gin_param="test_scene_list=\"${TEST_SCENE_LIST}\"" \
    --gin_param="SplatFactoSRDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    --gin_param="train_dataset/SplatFactoSRDataset.image_per_scene=8" \
    --gin_param="total_steps=${TOTAL_STEPS}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.log_image_interval=${LOG_IMAGE_INTERVAL}"

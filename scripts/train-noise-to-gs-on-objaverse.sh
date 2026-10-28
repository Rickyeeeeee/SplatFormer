#!/bin/bash
set -euo pipefail

# Example: GPU_IDS=0,1 EVAL_SCENE_NAME=<test-scene-id> bash scripts/train-noise-to-gs-on-objaverse.sh
GPU_IDS=${GPU_IDS:-0}
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
NGPUS=${NGPUS:-${#gpu_ids[@]}}
MASTER_PORT=${MASTER_PORT:-29521}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-true}
BATCH_SIZE=${BATCH_SIZE:-64}
COORDINATE_MODE=${COORDINATE_MODE:-morton_rank}
case "${COORDINATE_MODE}" in
    morton_rank|current) ;;
    *) echo "COORDINATE_MODE must be morton_rank or current" >&2; exit 2 ;;
esac
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
GS_COUNT=${GS_COUNT:-1000}
RESOLUTION=${RESOLUTION:-32}
DIT_PATCH_SIZE=${DIT_PATCH_SIZE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
FLOW_STEPS=${FLOW_STEPS:-25}
EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-0}
RANDOM_ROTATE=${RANDOM_ROTATE:-False}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-noise-to-gs}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
GS_DATASET_ROOT=${GS_DATASET_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${GS_DATASET_ROOT}/train-set/objaverse/${RESOLUTION}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
TRAIN_SPLAT_DIR=${TRAIN_SPLAT_DIR:-${GS_DATASET_ROOT}/train-set/objaverse/${RESOLUTION}/gsplat}
TEST_SPLAT_DIR=${TEST_SPLAT_DIR:-${GS_DATASET_ROOT}/test-set/objaverse/${RESOLUTION}/gsplat}
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-${GS_DATASET_ROOT}/train-set/objaverse/${RESOLUTION}/gs_statistics.json}
: "${EVAL_SCENE_NAME:?Set EVAL_SCENE_NAME to a scene in TEST_SCENE_LIST}"
OUTPUT_DIR=${OUTPUT_DIR:-/project2/ricky/outputs/noise_to_gs_${RESOLUTION}_n${GS_COUNT}_patch${DIT_PATCH_SIZE}}
init_args=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    init_args=("--init_checkpoint=${INIT_CHECKPOINT}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --rdzv-endpoint="localhost:${MASTER_PORT}" \
    train-noise-to-gs.py \
    --output_dir="${OUTPUT_DIR}" \
    --train_splat_dir="${TRAIN_SPLAT_DIR}" \
    --test_splat_dir="${TEST_SPLAT_DIR}" \
    --eval_scene_name="${EVAL_SCENE_NAME}" \
    --gs_count="${GS_COUNT}" \
    --batch_size="${BATCH_SIZE}" \
    --grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --prefetch_factor="${PREFETCH_FACTOR}" \
    --pin_memory="${PIN_MEMORY}" \
    --use_wandb="${USE_WANDB}" \
    --wandb_project="${WANDB_PROJECT}" \
    "${init_args[@]}" \
    --gin_file=configs/model/dit_gaussian.gin \
    --gin_file=configs/dataset/objaverse-sr.gin \
    --gin_file=configs/train/noise_to_gs.gin \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.max_gs_num=${GS_COUNT}" \
    --gin_param="GaussianDiTPredictor.max_gaussians=${GS_COUNT}" \
    --gin_param="GaussianDiTPredictor.coordinate_mode='${COORDINATE_MODE}'" \
    --gin_param="GaussianDiT.patch_size=${DIT_PATCH_SIZE}" \
    --gin_param="flow_matching.gs_statistics_path='${GS_STATISTICS_PATH}'" \
    --gin_param="flow_matching.flow_steps=${FLOW_STEPS}" \
    --gin_param="flow_matching.eval_noise_seed=${EVAL_NOISE_SEED}" \
    --gin_param="training_augmentation.random_rotate=${RANDOM_ROTATE}" \
    --gin_param="training.total_steps=${TOTAL_STEPS}" \
    --gin_param="training.eval_interval=${EVAL_INTERVAL}" \
    --gin_param="training.save_interval=${SAVE_INTERVAL}" \
    --gin_param="training.log_interval=${LOG_INTERVAL}" \
    --gin_param="train2D/build_scheduler.total_step=${TOTAL_STEPS}"

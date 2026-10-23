#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

GPU_ID=${GPU_ID:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/project/ricky/splatformer-sr-data-scaled}
SOURCE_GS_ROOT=${SOURCE_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-500}
TARGET_GS_ROOT=${TARGET_GS_ROOT:-/project/ricky/splatformer-sr-data-mcmc-1000}
TRAIN_SCENE_LIST=${TRAIN_SCENE_LIST:-${DATASET_ROOT}/psnr_filtered_scenes.csv}
TEST_SCENE_LIST=${TEST_SCENE_LIST:-${DATASET_ROOT}/test_psnr_filtered_scenes.csv}
GS_STATISTICS_PATH=${GS_STATISTICS_PATH:-${DATASET_ROOT}/gs_statistics.json}
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
SPLIT=${SPLIT:-test}
INPUT_RESOLUTION=${INPUT_RESOLUTION:-32}
TARGET_RESOLUTION=${TARGET_RESOLUTION:-64}
EVAL_NOISE_SEED=${EVAL_NOISE_SEED:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8084}
VIEW_MODE=${VIEW_MODE:-pointcloud}
EXPORT_SETTINGS=${EXPORT_SETTINGS:-noise_to_gs_preset.json}

preset_args=()
if [[ -n "${PRESET:-}" ]]; then
    preset_args+=("--preset=${PRESET}")
fi

# Extra CLI arguments can override launcher defaults, including --dry_run.
CUDA_VISIBLE_DEVICES="${GPU_ID}" exec "${PYTHON_BIN}" scripts/visualize_noise_to_gs.py \
    --config configs/dataset/objaverse-sr.gin \
    --scene_name="${SCENE_NAME}" \
    --split="${SPLIT}" \
    --source_gs_root="${SOURCE_GS_ROOT}" \
    --target_gs_root="${TARGET_GS_ROOT}" \
    --statistics_path="${GS_STATISTICS_PATH}" \
    --noise_seed="${EVAL_NOISE_SEED}" \
    --host="${HOST}" \
    --port="${PORT}" \
    --initial_view_mode="${VIEW_MODE}" \
    --export_settings="${EXPORT_SETTINGS}" \
    --gin_param="dataset_root='${DATASET_ROOT}'" \
    --gin_param="train_scene_list='${TRAIN_SCENE_LIST}'" \
    --gin_param="test_scene_list='${TEST_SCENE_LIST}'" \
    --gin_param="SplatFactoSRDataset.src_resolution=${INPUT_RESOLUTION}" \
    --gin_param="SplatFactoSRDataset.tgt_resolution=${TARGET_RESOLUTION}" \
    "${preset_args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
RUNS_ROOT=${1:-${RUNS_ROOT:-}}
OUTPUT_DIR=${OUTPUT_DIR:-}
SOURCE_EVAL_SUBDIR=${SOURCE_EVAL_SUBDIR:-eval_final}
SCENE_NAME=${SCENE_NAME:-}
INPUT_FACTOR=${INPUT_FACTOR:-4}
TARGET_FACTOR=${TARGET_FACTOR:-1}
EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-16}
NERFSTUDIO_FOLDER=${NERFSTUDIO_FOLDER:-/project/ricky/splatformer-data/test-set-512/objaverse/nerfstudio}
COLMAP_FOLDER=${COLMAP_FOLDER:-/project/ricky/splatformer-data/test-set-512/objaverse/colmap}

if [[ -z "${RUNS_ROOT}" ]]; then
    echo "Usage: $0 <runs_root>" >&2
    echo "RUNS_ROOT must contain one direct single-attribute run for every GS parameter." >&2
    exit 1
fi

args=(
    --runs_root="${RUNS_ROOT}"
    --source_eval_subdir="${SOURCE_EVAL_SUBDIR}"
    --eval_chunk_size="${EVAL_CHUNK_SIZE}"
    --gin_file=configs/model/ptv3.gin
    --gin_file=configs/overfit/sr_mse.gin
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${NERFSTUDIO_FOLDER}'"
    --gin_param="train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${COLMAP_FOLDER}'"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
    args+=(--output_dir="${OUTPUT_DIR}")
fi
if [[ -n "${SCENE_NAME}" ]]; then
    args+=(--scene_name="${SCENE_NAME}")
fi
if [[ "${INPUT_FACTOR}" != "-1" ]]; then
    args+=(--input_factor="${INPUT_FACTOR}")
fi
if [[ "${TARGET_FACTOR}" != "-1" ]]; then
    args+=(--target_factor="${TARGET_FACTOR}")
fi

echo "Merging final single-attribute PLYs from: ${RUNS_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python merge-sr-mse-ply.py "${args[@]}"

#!/bin/bash
set -euo pipefail

DIPT_PARTITION_MODE=rotated

viewer_gin_args=()
if [[ "${BACKEND:-dipt}" == "dipt" && -n "${DIPT_PARTITION_MODE:-}" ]]; then
    viewer_gin_args+=("--gin_param=DiffusionGaussianTransformer.partition_mode='${DIPT_PARTITION_MODE}'")
fi

CUDA_VISIBLE_DEVICES=${GPU_ID:-1} python scripts/visualize_augmentations.py \
    --backend "${BACKEND:-dipt}" \
    --config "${DATASET_CONFIG:-configs/dataset/objaverse-sr.gin}" \
    --scene_name "${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}" \
    --split "${SPLIT:-test}" --port "${PORT:-8083}" "${viewer_gin_args[@]}" "$@"

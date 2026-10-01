#!/bin/bash
set -euo pipefail


CUDA_VISIBLE_DEVICES=${GPU_ID:-3} python scripts/visualize_augmentations.py \
    --backend "${BACKEND:-dipt}" \
    --config "${DATASET_CONFIG:-configs/dataset/objaverse-sr.gin}" \
    --scene_name "${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}" \
    --split "${SPLIT:-test}" --port "${PORT:-8083}" "$@"

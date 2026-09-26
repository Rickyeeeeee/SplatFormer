#!/bin/bash
set -euo pipefail
# Pass additional --gin_file/--gin_param bindings or CLI overrides as arguments.
args=(
    --mode configured
    --gin_file "${GIN_FILE:-configs/dataset/objaverse-sr.gin}"
    --scene_name "${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}"
    --dataset_scope "${DATASET_SCOPE:-test_dataset}"
    --gs_resolution "${GS_RESOLUTION:-source}"
    --random_jitter "${RANDOM_JITTER:-False}"
    --random_rotate "${RANDOM_ROTATE:-False}"
    --rotation_mode "${ROTATION_MODE:-full}"
    --trials "${TRIALS:-3}" --seed "${SEED:-42}"
    --device "${DEVICE:-cuda}" --chunk_size "${CHUNK_SIZE:-8}"
    --preview_views "${PREVIEW_VIEWS:-4}"
    --output_dir "${OUTPUT_DIR:-output_data_augmentation}"
    --gin_param "SplatFactoSRDataset.src_resolution=${INPUT_RESOLUTION:-32}"
    --gin_param "SplatFactoSRDataset.tgt_resolution=${TARGET_RESOLUTION:-128}"
)
if [[ -n "${JITTER_MAX_LEVELS:-}" ]]; then
    args+=(--jitter_max_levels "$JITTER_MAX_LEVELS")
fi
if [[ -n "${ROTATION_PIVOT:-}" ]]; then
    read -r -a pivot_values <<< "$ROTATION_PIVOT"
    args+=(--rotation_pivot "${pivot_values[@]}")
fi
CUDA_VISIBLE_DEVICES=${GPU_ID:-3} "${PYTHON:-python}" -m sr.data_augmentation "${args[@]}" "$@"

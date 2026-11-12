#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Shared settings for all scene-count experiments.
export GPU_IDS=9 BATCH_SIZE=32 GRAD_ACCUM_STEPS=1 EVAL_SAMPLE_COUNT=4 
export EMA_ENABLED=True EMA_DECAY=0.99
export COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False
export MODEL_OUTPUT_TYPE=x PREDICTION_TYPE=x
export USE_WANDB=False 
export CACHE_DATASET_ON_GPU=true GPU_CACHE_HEADROOM_GIB=4 

# Evaluation defaults to the first selected training scene; set EVAL_SCENE_NAME for a test scene.
# Train on the first scene in the training list.
# TRAIN_SCENE_COUNT=1 EPOCHS=64000 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Train on the first 10 scenes in the training list.
# TRAIN_SCENE_COUNT=10 EPOCHS=32000 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Train on the first scene in the training list.
# TRAIN_SCENE_COUNT=1 EPOCHS=64000 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_bottleneck \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Train on the first 10 scenes in the training list.
# TRAIN_SCENE_COUNT=10 EPOCHS=32000 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema_bottleneck \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Patch size overfitting
# Train on the first 1 scenes in the training list.
# TRAIN_SCENE_COUNT=1 EPOCHS=32000 DIT_PATCH_SIZE=1 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_p1 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=1 EPOCHS=32000 DIT_PATCH_SIZE=2 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_p2 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=1 EPOCHS=32000 DIT_PATCH_SIZE=4 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_p4 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=1 EPOCHS=32000 DIT_PATCH_SIZE=8 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_p8 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=1 EPOCHS=32000 DIT_PATCH_SIZE=16 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1_scenes_ema_p16 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Patch size overfitting: 10 scenes
# TRAIN_SCENE_COUNT=10 EPOCHS=32000 DIT_PATCH_SIZE=1 BATCH_SIZE=32 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema_p1 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=10 EPOCHS=32000 D/shapenet.sh

TRAIN_SCENE_COUNT=10 EPOCHS=64000 DIT_PATCH_SIZE=16 BATCH_SIZE=128 \
OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema_p16 \
bash scripts/train-noise-to-gs-on-shapenet.sh

# Patch size overfitting: 100 scenes
# TRAIN_SCENE_COUNT=100 EPOCHS=32000 DIT_PATCH_SIZE=1 BATCH_SIZE=32 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p1 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

TRAIN_SCENE_COUNT=100 EPOCHS=64000 DIT_PATCH_SIZE=4 BATCH_SIZE=128 \
OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p4 \
bash scripts/train-noise-to-gs-on-shapenet.sh

TRAIN_SCENE_COUNT=100 EPOCHS=128000 DIT_PATCH_SIZE=16 BATCH_SIZE=256 \
OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p16 \
bash scripts/train-noise-to-gs-on-shapenet.sh


# Train on the first 10 scenes in the training list.
# TRAIN_SCENE_COUNT=10 EPOCHS=32000 DIT_PATCH_SIZE=4 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema_p4 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Train on the first 100 scenes in the training list.
# TRAIN_SCENE_COUNT=100 EPOCHS=32000 BATCH_SIZE=128 GRAD_ACCUM_STEPS=4 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Train on the first 100 scenes in the training list.
# TRAIN_SCENE_COUNT=1000 EPOCHS=3200 BATCH_SIZE=128 GRAD_ACCUM_STEPS=4 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_1000_scenes_ema \
# bash scripts/train-noise-to-gs-on-shapenet.sh

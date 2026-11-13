#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Shared settings for all scene-count experiments.
export GPU_IDS=2 BATCH_SIZE=32 GRAD_ACCUM_STEPS=1 EVAL_SAMPLE_COUNT=4 
export MASTER_PORT=29524
export EMA_ENABLED=True EMA_DECAY=0.99
export COORDINATE_MODE=morton_rank FLOW_COUPLING=morton RANDOM_ROTATE=False
export MODEL_OUTPUT_TYPE=x PREDICTION_TYPE=x
export USE_WANDB=True 
export CACHE_DATASET_ON_GPU=True GPU_CACHE_HEADROOM_GIB=4 

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

# TRAIN_SCENE_COUNT=10 EPOCHS=64000 DIT_PATCH_SIZE=16 BATCH_SIZE=128 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_10_scenes_ema_p16 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# Patch size overfitting: 100 scenes
# TRAIN_SCENE_COUNT=100 EPOCHS=32000 DIT_PATCH_SIZE=1 BATCH_SIZE=32 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p1 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=12800 DIT_PATCH_SIZE=4 BATCH_SIZE=128 \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p4 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=12800 DIT_PATCH_SIZE=16 BATCH_SIZE=128 FLOW_COUPLING=morton DIT_SIZE=B \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p16 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=32 BATCH_SIZE=256 FLOW_COUPLING=morton DIT_SIZE=B \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p32 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=12800 DIT_PATCH_SIZE=8 BATCH_SIZE=128 FLOW_COUPLING=morton DIT_SIZE=B \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p8 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=32 BATCH_SIZE=256 FLOW_COUPLING=morton DIT_SIZE=B \
# GS_FOURIER_INPUT_FEATURES="['means','scales']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 6}" \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p32_fourier \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=32 BATCH_SIZE=256 FLOW_COUPLING=morton DIT_SIZE=B \
# PREDICTION_TYPE=v \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p32_vloss \
# bash scripts/train-noise-to-gs-on-shapenet.sh

TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=32 BATCH_SIZE=256 FLOW_COUPLING=random DIT_SIZE=B \
PREDICTION_TYPE=v \
OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_100_scenes_ema_p32_random \
bash scripts/train-noise-to-gs-on-shapenet.sh

TRAIN_SCENE_COUNT=1000 EPOCHS=51200 DIT_PATCH_SIZE=32 BATCH_SIZE=512 GRAD_ACCUM_STEPS=2 DIT_SIZE=B \
PREDICTION_TYPE=v FLOW_COUPLING=random \
LR_WARMUP_STEPS=2000 \
LR_WARMUP_START_FACTOR=0.03333333333333333 \
OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_1000_scenes_ema_p32_random \
bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=1000 EPOCHS=12800 DIT_PATCH_SIZE=32 BATCH_SIZE=128 FLOW_COUPLING=morton DIT_SIZE=B \
# GS_FOURIER_INPUT_FEATURES="['means','scales']" GS_FOURIER_NUM_FREQUENCIES="{'means': 6, 'scales': 6}" \
# GS_FOURIER_INCLUDE_RAW=True GS_FOURIER_LOG_SAMPLING=True GS_FOURIER_MAX_FREQUENCY_LOG2='{}' \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_B_1000_scenes_ema_p32_fourier \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=16 BATCH_SIZE=256 FLOW_COUPLING=morton \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p16 \
# bash scripts/train-noise-to-gs-on-shapenet.sh

# TRAIN_SCENE_COUNT=100 EPOCHS=25600 DIT_PATCH_SIZE=32 BATCH_SIZE=256 FLOW_COUPLING=morton \
# OUTPUT_DIR=/project2/ricky/outputs/$(date +%m%d)/shapenet_noise_to_gs_100_scenes_ema_p32 \
# bash scripts/train-noise-to-gs-on-shapenet.sh


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

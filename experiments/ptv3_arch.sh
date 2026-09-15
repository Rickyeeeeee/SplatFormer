#!/bin/bash

set -euo pipefail

# Shared data and optimization settings for all six comparisons.
export GPU_ID=3
export BATCH_SIZE=4
export GRAD_ACCUM_STEPS=1
export GRID_RESOLUTION=768
export SCENE_MODE=many
export SCENE_COUNT=18
export VELOCITY_VARIANCE_SOURCE=precomputed_aggregate
export MIX_SCHEDULE=fm-only
export FLOW_LOSS_TYPE=velocity
export FLOW_STEPS=1
export FLOW_NOISE_STD=0.0

# Shared backbone and output-head settings; the stride comparison overrides pooling below.
export PTV3_ENC_NUM_HEAD='(2,4,8,16,32)'
export PTV3_DEC_NUM_HEAD='(4,4,8,16)'
export PTV3_STRIDE='(1,2,2,2)'
export PTV3_EMBEDDING_TYPE=MLP
export PTV3_T_DIM=3
export PTV3_ENABLE_FLASH=True
export PTV3_TURN_OFF_BN=True
export PTV3_PDNORM_BN=False
export PTV3_PDNORM_LN=False
export PTV3_DROP_PATH=0.0
export PTV3_SHUFFLE_ORDERS=True
export PTV3_SHUFFLE_ORDERS_EVAL=False
export GS_OUTPUT_HEAD_WIDTH=128
export GS_OUTPUT_HEAD_NLAYER=4
export GS_OUTPUT_HEAD_TYPE=mlp-relu
export GS_INPUT_FEAT_TO_MLP=True
export GS_ZEROINIT=True
export GS_QUAT_RESIDUAL_MODE=add

# Run from the repository root; the Gin configuration fixes the seed at 42.
# Each comparison uses 20,000 steps and 32-to-128 resolution with its own postfix.

# baseline: use baseline widths.
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
PTV3_DEC_CHANNELS='(128,128,256,256)' \
CUSTOM_POSFIX=baseline \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# enc_deep: use baseline widths.
PTV3_ENC_DEPTHS='(3,3,3,9,3)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
PTV3_DEC_CHANNELS='(128,128,256,256)' \
CUSTOM_POSFIX=enc_deep \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# dec_deep: use baseline widths.
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(3,3,3,3)' \
PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
PTV3_DEC_CHANNELS='(128,128,256,256)' \
CUSTOM_POSFIX=dec_deep \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# both_deep: use baseline widths.
PTV3_ENC_DEPTHS='(3,3,3,9,3)' \
PTV3_DEC_DEPTHS='(3,3,3,3)' \
PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
PTV3_DEC_CHANNELS='(128,128,256,256)' \
CUSTOM_POSFIX=both_deep \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# wide: increase internal widths while preserving the 128-channel head input.
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(96,144,192,384,768)' \
PTV3_DEC_CHANNELS='(128,192,384,384)' \
CUSTOM_POSFIX=wide \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# stride_2222: baseline depth and width with downsampling at every pooling stage.
PTV3_STRIDE='(2,2,2,2)' \
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
PTV3_DEC_CHANNELS='(128,128,256,256)' \
CUSTOM_POSFIX=stride_2222 \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

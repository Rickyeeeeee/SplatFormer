# 0909
# GPU_ID=0 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# GS_STATISTICS_PATH=/project/ricky/splatformer-sr-data/gs_statistics.json \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# VELOCITY_VARIANCE_SOURCE=precomputed_scene \
# GS_STATISTICS_PATH=/project/ricky/splatformer-sr-data-scaled/test_gs_statistics.json \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# VELOCITY_VARIANCE_SOURCE=matching \
# GS_STATISTICS_PATH=/project/ricky/splatformer-sr-data-scaled/test_gs_statistics.json \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# 0910
# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# VELOCITY_VARIANCE_SOURCE=matching \
# GS_STATISTICS_PATH=/project/ricky/splatformer-sr-data-scaled/test_gs_statistics.json \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=2 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=4 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=8 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=19 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=15 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# 0911
# GPU_ID=0 \
# BATCH_SIZE=4 \
# GRID_RESOLUTION=1024 \
# SCENE_MODE=many \
# SCENE_COUNT=19 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# BATCH_SIZE=4 \
# GRID_RESOLUTION=4096 \
# SCENE_MODE=many \
# SCENE_COUNT=19 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# GPU_ID=0 \
# BATCH_SIZE=1 \
# GRID_RESOLUTION=2048 \
# SCENE_MODE=many \
# SCENE_COUNT=19 \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 

# 0913
# GPU_ID=0 \
# BATCH_SIZE=1 \
# GRID_RESOLUTION=768 \
# SCENE_MODE=one \
# VELOCITY_VARIANCE_SOURCE=matching \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 10000 10000 1000 1000 fit_lr_to_hr aligned 32 128

# GPU_ID=0 \
# BATCH_SIZE=1 \
# GRID_RESOLUTION=768 \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

# GPU_ID=0 \
# BATCH_SIZE=4 \
# GRID_RESOLUTION=768 \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

# GPU_ID=0 \
# BATCH_SIZE=8 \
# GRID_RESOLUTION=768 \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

# 0915
# GPU_ID=6 \
# BATCH_SIZE=8 \
# GRAD_ACCUM_STEPS=1 \
# GRID_RESOLUTION=1536 \
# SCENE_MODE=many \
# SCENE_COUNT=18 \
# VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
# PTV3_ENC_DEPTHS='(3,3,3,9,3)' \
# PTV3_DEC_DEPTHS='(3,3,3,3)' \
# PTV3_ENC_CHANNELS='(64,96,128,256,512)' \
# PTV3_DEC_CHANNELS='(128,128,256,256)' \
# CUSTOM_POSFIX=both_deep \
# bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

GPU_ID=6 \
BATCH_SIZE=6 \
GRAD_ACCUM_STEPS=1 \
GRID_RESOLUTION=1536 \
SCENE_MODE=many \
SCENE_COUNT=18 \
VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(96,144,192,384,768)' \
PTV3_DEC_CHANNELS='(128,192,384,384)' \
CUSTOM_POSFIX=larger \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

GPU_ID=6 \
BATCH_SIZE=6 \
GRAD_ACCUM_STEPS=1 \
GRID_RESOLUTION=1024 \
SCENE_MODE=many \
SCENE_COUNT=18 \
VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
PTV3_ENC_DEPTHS='(2,2,2,6,2)' \
PTV3_DEC_DEPTHS='(2,2,2,2)' \
PTV3_ENC_CHANNELS='(96,144,192,384,768)' \
PTV3_DEC_CHANNELS='(128,192,384,384)' \
CUSTOM_POSFIX=larger \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 20000 20000 4000 4000 fit_lr_to_hr aligned 32 128

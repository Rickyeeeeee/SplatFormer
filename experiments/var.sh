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
GPU_ID=0 \
BATCH_SIZE=4 \
GRID_RESOLUTION=1024 \
SCENE_MODE=many \
SCENE_COUNT=19 \
VELOCITY_VARIANCE_SOURCE=precomputed_aggregate \
bash scripts/overfit-sr-gsfm-on-objaverse.sh 

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

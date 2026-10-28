#!/bin/bash
set -euo pipefail

# CHECKPOINT=/project2/ricky/experiments/1025/3e288ee8aced4a0797e66d53536112b1_dit_morton_rank_hr32_n1000/checkpoints/model_00009999.pth
# CHECKPOINT=/project2/ricky/experiments/1025/many_18_dit_morton_rank_hr32_n1000/checkpoints/model_00009999.pth
# CHECKPOINT=/project2/ricky/experiments/1025/3e288ee8aced4a0797e66d53536112b1_dit_morton_rank_rot_gravity_consistent_hr32_n1000/checkpoints/model_00009999.pth
CHECKPOINT=/project2/ricky/experiments/1028/dit_hr32_n1000_morton_coupling/checkpoints/model_00009999.pth
PREDICTOR=dit

CUDA_VISIBLE_DEVICES=0 python scripts/visualize_noise_to_gs_checkpoint.py \
    --checkpoint="${CHECKPOINT}" \
    --predictor="${PREDICTOR}" \
    --gs_count=1000 

GPU_ID=${GPU_ID:-3}


# FIXED_TRAIN_NOISE=True \
# TRAIN_NOISE_SEED=0 \
# EVAL_NOISE_SEED=0 \
# SPATIAL_COORD_MODE=dynamic \
# DYNAMIC_GRID_SIZE=0.02 \

# INTERPOLANT_TYPE=one_sided \
# FLOW_NOISE_STD=0.1 \
# BATCH_SIZE=64 \
# GRAD_ACCUM_STEPS=4 \
# bash scripts/overfit-sr-interpolants.sh

# INTERPOLANT_TYPE=linear \
# bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=latent \
FLOW_NOISE_STD=0.1 \
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=encoding_decoding \
FLOW_NOISE_STD=0.1 \
bash scripts/overfit-sr-interpolants.sh

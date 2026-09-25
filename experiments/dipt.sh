GPU_ID=${GPU_ID:-3}


# FIXED_TRAIN_NOISE=True \
# TRAIN_NOISE_SEED=0 \
# EVAL_NOISE_SEED=0 \
# SPATIAL_COORD_MODE=dynamic \
# DYNAMIC_GRID_SIZE=0.02 \


# INTERPOLANT_TYPE=linear \
# LR_WARMUP_STEPS=500 \
# LR_WARMUP_START_FACTOR=0.03333333333333333 \
# bash scripts/overfit-sr-interpolants.sh

# INTERPOLANT_TYPE=one_sided \
# FLOW_NOISE_STD=1.0 \
# LR_WARMUP_STEPS=500 \
# LR_WARMUP_START_FACTOR=0.03333333333333333 \
# bash scripts/overfit-sr-interpolants.sh

# INTERPOLANT_TYPE=latent \
# FLOW_NOISE_STD=0.1 \
# bash scripts/overfit-sr-interpolants.sh

# INTERPOLANT_TYPE=encoding_decoding \
# FLOW_NOISE_STD=0.1 \
# bash scripts/overfit-sr-interpolants.sh

# 0925 jitter
INTERPOLANT_TYPE=linear \
RANDOM_JITTER=True \
JITTER_MAX_LEVELS="{
  'means': 0.01,
  'scales': 0.0,
  'opacities': 0.0,
  'quats': 0.0,
  'features_dc': 0.0,
  'features_rest': 0.0
}" \
CUSTOM_POSFIX=jmeans
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
RANDOM_JITTER=True \
JITTER_MAX_LEVELS="{
  'means': 0.0,
  'scales': 0.05,
  'opacities': 0.0,
  'quats': 0.0,
  'features_dc': 0.0,
  'features_rest': 0.0
}" \
CUSTOM_POSFIX=jscales
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
RANDOM_JITTER=True \
JITTER_MAX_LEVELS="{
  'means': 0.0,
  'scales': 0.0,
  'opacities': 0.1,
  'quats': 0.0,
  'features_dc': 0.0,
  'features_rest': 0.0
}" \
CUSTOM_POSFIX=jopacities
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
RANDOM_JITTER=True \
JITTER_MAX_LEVELS="{
  'means': 0.0,
  'scales': 0.0,
  'opacities': 0.0,
  'quats': 0.05,
  'features_dc': 0.0,
  'features_rest': 0.0
}" \
CUSTOM_POSFIX=jquats
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
RANDOM_JITTER=True \
JITTER_MAX_LEVELS="{
  'means': 0.01,
  'scales': 0.05,
  'opacities': 0.1,
  'quats': 0.05,
  'features_dc': 0.05,
  'features_rest': 0.1
}" \
CUSTOM_POSFIX=jall
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
RANDOM_ROTATE=True \
bash scripts/overfit-sr-interpolants.sh
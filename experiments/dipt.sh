GPU_ID=${GPU_ID:-4}


INTERPOLANT_TYPE=one_sided \
FLOW_NOISE_STD=1.0 \
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=linear \
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=latent \
FLOW_NOISE_STD=1.0 \
bash scripts/overfit-sr-interpolants.sh

INTERPOLANT_TYPE=encoding_decoding \
FLOW_NOISE_STD=1.0 \
bash scripts/overfit-sr-interpolants.sh

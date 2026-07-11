# Predicted means from stage-1 means model
# GPU_ID=4 bash scripts/overfit-sr-2stage-on-objaverse.sh 1000 200 200 200 emd aligned 4 2 predicted 16
# GPU_ID=4 bash scripts/overfit-sr-2stage-on-objaverse.sh 1000 200 200 200 emd aligned 4 2 gt 16
# GPU_ID=4 bash scripts/overfit-sr-2stage-on-objaverse.sh 1000 200 200 200 emd aligned 4 2 predicted 16

GPU_ID=4 bash scripts/train-sr-2stage-on-objaverse.sh 20000 1000 1000 1000 emd aligned 4 2 predicted
# GPU_ID=4 bash scripts/train-sr-on-objaverse.sh 20000 1000 1000 1000 
# GPU_ID=4 bash scripts/train-sr-2stage-on-objaverse.sh 20000 1000 1000 1000 emd aligned 4 2 gt

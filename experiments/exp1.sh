# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 means
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 means,scales
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 means,scales,opacities
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 means,scales,opacities,features_dc,features_rest

# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats,scales
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats,means,scales
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats,means,scales,opacities
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats,means,scales,opacities,features_dc,features_rest

# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 scales
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 opacities
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 features_dc,features_rest

# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 quats,scales,opacities,features_dc,features_rest
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 2 scales,opacities,features_dc,features_rest
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd 3dgs 4 1 scales,opacities true
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 quats,scales,opacities true
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 1 means,scales,opacities true
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means,quats,scales,opacities,features_dc
# GPU_ID=5 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 nearest 3dgs 4 2 means,quats,scales,opacities,features_dc

# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means,quats,scales,opacities,features_dc true false 1.01
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd aligned 4 2 means,quats,scales,opacities,features_dc true false 1.001
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means true false 1.01
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means true false 1.005
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 nearest 3dgs 4 2 means true false 1.001
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means true false 1.0001
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd 3dgs 4 2 means true false 1.00001
# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 nearest 3dgs 4 2 means,quats,scales,opacities,features_dc true false 1.001

# GPU_ID=2 bash scripts/overfit-sr-mse-on-objaverse.sh 2000 200 200 200 emd aligned 4 2 means,quats,scales,opacities,features_dc true false 1.001


# 0809
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 means
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 scales
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 quats
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 opacities
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 features_dc
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 features_rest
# GPU_ID=0 bash scripts/overfit-sr-mse-noemd-on-objaverse.sh 4000 4000 400 400 4 1 means,opacities,features_dc,features_rest,scales,quats

# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 4000 4000 400 400 4 1 means,opacities,features_dc,features_rest,scales,quats
# GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 4000 4000 400 400 4 1 means,opacities,features_dc,features_rest,scales,quats
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 means
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 scales
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 quats
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 opacities
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 features_dc
# MIX_SCHEDULE=fm-only GPU_ID=0 bash scripts/overfit-sr-gsfm-noemd-on-objaverse.sh 3000 3000 400 400 4 1 features_rest

# 0809
# GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 means
# GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 scales
# GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 opacities
# GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 features_dc
# GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 features_rest
GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 3000 3000 400 400 emd aligned 4 1 quats
GPU_ID=3 bash scripts/overfit-sr-mse-on-objaverse.sh 4000 4000 400 400 emd aligned 4 1 means,opacities,features_dc,features_rest,scales,quats
# GPU_ID=0 bash scripts/overfit-sr-mse-on-objaverse.sh 5000 200 200 200 emd aligned 4 1 \
#     means,opacities,features_dc,features_rest,scales,quats \
#     False True 1.01 true ./gs_statistics.json
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities,features_dc
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities,scales,features_dc
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities,features_dc,scales
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities,features_dc,scales,quats
# GPU_ID=4 bash scripts/overfit-sr-mse-on-objaverse.sh 1000 200 200 200 emd aligned 4 1 means,opacities,features_dc

# GPU_ID=4 bash scripts/overfit-sr-mse-mix-on-objaverse.sh 2000 200 200 200 emd aligned 4 1 \
#     post_activate false 1.01 1.0 1.0
# GPU_ID=5 bash scripts/overfit-sr-gsfm-mix-on-objaverse.sh 8000 400 400 400 emd aligned 4 1

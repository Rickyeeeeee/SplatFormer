CUDA_VISIBLE_DEVICES=1 python3 dataset_statistics.py \
  --colmap_root /project/ricky/splatformer-data/train-set-512/objaverse/colmap \
  --nerfstudio_root /project/ricky/splatformer-data/train-set-512/objaverse/nerfstudio \
  --output_csv ./dataset_stats.csv \
  --device cuda \
  --disable_metrics

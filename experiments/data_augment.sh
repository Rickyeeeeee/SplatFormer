CUDA_VISIBLE_DEVICES=3 python -m sr.data_augmentation \
  --gin_file configs/dataset/objaverse-sr.gin \
  --scene_name 3e288ee8aced4a0797e66d53536112b1 \
  --jitter_levels 0.01 0.05 0.1 \
  --trials 3 \
  --output_dir output_data_augmentation
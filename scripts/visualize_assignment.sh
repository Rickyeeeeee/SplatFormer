CUDA_VISIBLE_DEVICES=5
python scripts/visualize_sr_assignments.py \
  --densify_init /project/ricky/experiments/objaverse_splatformer_overfit_sr_mse_512/3e288ee8aced4a0797e66d53536112b1_aligned_if4_tf2_mse_means_lossout/densify_init \
  --sample_count 512 \
  --port 8080
  # --point_cloud_folder /project/ricky/experiments/objaverse_splatformer_overfit_sr_mse_512/3e288ee8aced4a0797e66d53536112b1_aligned_if4_tf1_mse_means_lossout/eval_final/viewer/3e288ee8aced4a0797e66d53536112b1/point_cloud \
  # --densify_init /project/ricky/experiments/objaverse_splatformer_overfit_sr_mse_512/3e288ee8aced4a0797e66d53536112b1_emd_aligned_if4_tf2_mse_means-quats-scales-opacities-features_dc_pa1_direct0_mos1.001/densify_init \
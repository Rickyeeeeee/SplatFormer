# python scripts/visualize_gs_displacement.py \
#   --viewer_dir /home/ricky/AnchorSplat/outputs/anchorsplat_1x_multi_eval/eval_final_4to1/3e288ee8aced4a0797e66d53536112b1/viewer/3e288ee8aced4a0797e66d53536112b1 \
#   --initial_view_mode gsplat \

# python scripts/visualize_gs_displacement.py \
#   --viewer_dir outputs/objaverse_train_sr_2stage_4to1_low_res/eval/00136000/3e288ee8aced4a0797e66d53536112b1/viewer/3e288ee8aced4a0797e66d53536112b1 \
#   --initial_view_mode gsplat \
  
# python scripts/visualize_gs_displacement.py \
#   --input_ply /home/ricky/AnchorSplat/examples/3dgs_sr_demo.ply \
#   --output_ply /home/ricky/AnchorSplat/outputs/3dgs_sr_demo_1x.ply \
#   --initial_view_mode gsplat \
#   --render_height 720

# python scripts/visualize_gs_displacement.py \
#   --initial_view_mode gsplat \
#   --input_ply /project/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means-opacities-features_dc-features_rest-scales-quats_lossout/matching_init/00_input_low_res_gs.ply \
#   --output_ply /project/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means-opacities-features_dc-features_rest-scales-quats_lossout/matching_init/01_fitted_target_gs.ply

# python scripts/visualize_gs_displacement.py \
#   --initial_view_mode gsplat \
#   --input_ply /project/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means-opacities-features_dc-features_rest-scales-quats_lossout/eval_final/viewer/3e288ee8aced4a0797e66d53536112b1/point_cloud/input.ply \
#   --output_ply /project/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means-opacities-features_dc-features_rest-scales-quats_lossout/eval_final/viewer/3e288ee8aced4a0797e66d53536112b1/point_cloud/output.ply

# python scripts/visualize_gs_displacement.py \
#   --initial_view_mode gsplat \
#   --input_ply /project2/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means_lossout/matching_init/00_input_low_res_gs.ply \
#   --output_ply /project2/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means_lossout/matching_init/01_fitted_target_gs.ply

python scripts/visualize_gs_displacement.py \
  --initial_view_mode gsplat \
  --input_ply /project2/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means_lossout/eval_final/viewer/3e288ee8aced4a0797e66d53536112b1/point_cloud/input.ply \
  --output_ply /project2/ricky/experiments/0809/overfit_sr_mse_noemd_512/3e288ee8aced4a0797e66d53536112b1_if4_tf1_noemd_mse_means_lossout/eval_final/viewer/3e288ee8aced4a0797e66d53536112b1/point_cloud/output.ply

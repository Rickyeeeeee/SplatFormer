
RUN=/project2/ricky/outputs/0915-gpu7/objaverse_sr_gsfm_32to128_fit_lr_to_hr_fm-only

CUDA_VISIBLE_DEVICES=3 python scripts/visualize_gs_flow.py \
  --config "$RUN/config.gin" \
  --checkpoint "$RUN/checkpoints/model_00179999.pth" \
  --scene_name 3e288ee8aced4a0797e66d53536112b1 \
  --split test \
  --flow_steps 10 \
  --device cuda \
  --port 8082

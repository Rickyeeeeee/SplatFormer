torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir test --compare_with_input --save_viewer --save_residuals \
    --gin_param="FeaturePredictor.resume_ckpt='checkpoints/train-on-objaverse.pth'" 


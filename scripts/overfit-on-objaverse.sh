out_name=first_scene
output_dir=outputs/objaverse_splatformer_overfit/${out_name}

total_steps=${1:-1000}
save_interval=${2:-100}
eval_interval=${2:-100}
log_image_interval=${3:-10}

CUDA_VISIBLE_DEVICES=7 python overfit.py \
    --output_dir=${output_dir} \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/overfit/default.gin \
    --gin_param="training.total_steps=${total_steps}" \
    --gin_param="training.save_interval=${save_interval}" \
    --gin_param="training.eval_interval=${eval_interval}" \
    --gin_param="training.log_image_interval=${log_image_interval}" \
    --gin_param="build_trainloader.batch_size=1"

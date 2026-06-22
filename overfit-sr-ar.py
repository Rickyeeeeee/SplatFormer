import json
import os
import random

import cv2
import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from gs_flow_model import GSFlowModel
from utils import gpu_utils, gs_utils, loss_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

INPUT_FACTOR = 4
TARGET_FACTOR = 2


@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@gin.configurable
def training(
    output_dir=None,
    total_steps=gin.REQUIRED,
    pretrain_steps=gin.REQUIRED,
    eval_interval=gin.REQUIRED,
    log_interval=gin.REQUIRED,
    save_interval=gin.REQUIRED,
    log_image_interval=gin.REQUIRED,
    grad_clip_norm=gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0.0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
    ar_num_stages=10,
    ar_num_rollouts=None,
    ar_steps_per_stage=None,
    # Accepted for compatibility with configs/overfit/sr-flow.gin. The AR
    # curriculum does not use teacher GS paths or velocity supervision.
    flow_loss_weight=1.0,
    flow_loss_type="mse",
    flow_eval_steps=10,
    flow_trajectory_mode="optimized_target_discrete",
    flow_num_timesteps=10,
    flow_segment_steps=1000,
    flow_integration_timestep_mode="left",
):
    del (
        pretrain_steps,
        flow_loss_weight,
        flow_loss_type,
        flow_eval_steps,
        flow_trajectory_mode,
        flow_num_timesteps,
        flow_segment_steps,
        flow_integration_timestep_mode,
    )
    if ar_num_stages <= 0:
        raise ValueError("training.ar_num_stages must be positive")
    used_legacy_rollout_alias = ar_num_rollouts is None and ar_steps_per_stage is not None
    if ar_num_rollouts is None:
        ar_num_rollouts = 100 if ar_steps_per_stage is None else ar_steps_per_stage
    elif ar_steps_per_stage is not None:
        raise ValueError(
            "Configure training.ar_num_rollouts or the deprecated "
            "training.ar_steps_per_stage, not both"
        )
    if ar_num_rollouts <= 0:
        raise ValueError("training.ar_num_rollouts must be positive")
    return {
        "output_dir": output_dir,
        "legacy_total_steps": total_steps,
        "total_steps": int(ar_num_stages) * int(ar_num_rollouts),
        "ar_num_stages": int(ar_num_stages),
        "ar_num_rollouts": int(ar_num_rollouts),
        "used_legacy_rollout_alias": used_legacy_rollout_alias,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "log_image_interval": log_image_interval,
        "grad_clip_norm": grad_clip_norm,
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "resume_from_step": resume_from_step,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


def make_grid(imgs, nrow=3, ncols=3):
    img_h, img_w = imgs[0].shape[:2]
    channels = 3 if imgs[0].ndim == 3 else None
    shape = (img_h * nrow, img_w * ncols, channels) if channels else (img_h * nrow, img_w * ncols)
    grid = np.zeros(shape, dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncols):
            index = i * ncols + j
            if index >= len(imgs):
                break
            grid[i * img_h : (i + 1) * img_h, j * img_w : (j + 1) * img_w] = imgs[index]
    return grid


def _to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu(value) for value in data)
    return data


def _sanitize_for_filename(value):
    return str(value).replace("/", "_").replace("\\", "_")


def stage_timestep(stage_index, num_stages):
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if stage_index < 0 or stage_index >= num_stages:
        raise ValueError(f"stage_index {stage_index} is outside [0, {num_stages})")
    return float(stage_index + 1) / float(num_stages)


def interpolate_resolution(source_size, target_size, t):
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    source_h, source_w = map(int, source_size)
    target_h, target_w = map(int, target_size)
    height = int(round(source_h + (target_h - source_h) * float(t)))
    width = int(round(source_w + (target_w - source_w) * float(t)))
    return height, width


def build_resolution_schedule(source_size, target_size, num_stages):
    return [
        {
            "stage_index": stage_index,
            "t": stage_timestep(stage_index, num_stages),
            "size": interpolate_resolution(
                source_size,
                target_size,
                stage_timestep(stage_index, num_stages),
            ),
        }
        for stage_index in range(num_stages)
    ]


def training_position(global_step, num_stages):
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    return global_step // num_stages, global_step % num_stages


def resize_images_and_cameras(images, cameras, output_size):
    if len(images) == 0:
        raise ValueError("Cannot resize an empty image list")
    output_h, output_w = map(int, output_size)
    if output_h <= 0 or output_w <= 0:
        raise ValueError(f"Invalid output size: {output_size}")

    input_h, input_w = images[0].shape[:2]
    for image in images:
        if tuple(image.shape[:2]) != (input_h, input_w):
            raise ValueError("All images in a payload must have the same dimensions")

    if (input_h, input_w) == (output_h, output_w):
        resized_images = images
    else:
        image_batch = torch.stack([image.permute(2, 0, 1) for image in images], dim=0)
        resized_batch = F.interpolate(
            image_batch,
            size=(output_h, output_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        resized_images = [image.permute(1, 2, 0) for image in resized_batch]

    camera_width = float(torch.as_tensor(cameras["width"]).item())
    camera_height = float(torch.as_tensor(cameras["height"]).item())
    scale_x = float(output_w) / camera_width
    scale_y = float(output_h) / camera_height
    resized_cameras = dict(cameras)
    resized_cameras["fx"] = cameras["fx"] * scale_x
    resized_cameras["fy"] = cameras["fy"] * scale_y
    resized_cameras["cx"] = cameras["cx"] * scale_x
    resized_cameras["cy"] = cameras["cy"] * scale_y
    resized_cameras["width"] = torch.as_tensor(
        output_w,
        dtype=torch.as_tensor(cameras["width"]).dtype,
        device=torch.as_tensor(cameras["width"]).device,
    )
    resized_cameras["height"] = torch.as_tensor(
        output_h,
        dtype=torch.as_tensor(cameras["height"]).dtype,
        device=torch.as_tensor(cameras["height"]).device,
    )
    return resized_images, resized_cameras


def apply_gs_delta(gs, delta):
    next_gs = dict(gs)
    for key, value in delta.items():
        if key not in gs:
            raise KeyError(f"Predicted GS delta key '{key}' is not present in the input GS")
        if not torch.is_tensor(gs[key]) or not torch.is_tensor(value):
            raise TypeError(f"GS update '{key}' must contain tensors")
        next_gs[key] = gs[key] + value
    return next_gs


def detach_gs(gs):
    return {key: value.detach() if torch.is_tensor(value) else value for key, value in gs.items()}


def rollout_ar(model, input_gs, num_steps, num_stages):
    if num_steps < 0 or num_steps > num_stages:
        raise ValueError(f"num_steps must be in [0, {num_stages}], got {num_steps}")
    current_gs = input_gs
    for stage_index in range(num_steps):
        t = torch.tensor(
            [stage_timestep(stage_index, num_stages)],
            device=current_gs["means"].device,
            dtype=torch.float32,
        )
        delta = model(batch_normalized_gs=[current_gs], timestep=t)[0]
        current_gs = apply_gs_delta(current_gs, delta)
    return current_gs


def rollout_detached_prefix(model, input_gs, stage_index, num_stages, enable_amp=False):
    del enable_amp
    current_gs = detach_gs(input_gs)
    was_training = model.training
    model.eval()
    try:
        # This spconv build has no suitable eval-mode FP16 implicit-GEMM
        # algorithm for some sparse shapes. Match full evaluation and run the
        # detached inference path in FP32.
        with torch.no_grad():
            current_gs = rollout_ar(
                model, current_gs, num_steps=stage_index, num_stages=num_stages
            )
    finally:
        model.train(was_training)
    return detach_gs(current_gs)


def recompute_detached_stage(model, current_gs, stage_index, num_stages, enable_amp=False):
    del enable_amp
    was_training = model.training
    model.eval()
    try:
        # See rollout_detached_prefix: eval-mode spconv must remain FP32.
        with torch.no_grad():
            t = torch.tensor(
                [stage_timestep(stage_index, num_stages)],
                device=current_gs["means"].device,
                dtype=torch.float32,
            )
            delta = model(batch_normalized_gs=[current_gs], timestep=t)[0]
            next_gs = apply_gs_delta(current_gs, delta)
    finally:
        model.train(was_training)
    return detach_gs(next_gs)


def _build_dataset():
    with gin.config_scope("train_dataset"):
        return SplatFactoMultiLevelDataset()


def _scene_name_from_dataset(dataset, idx):
    return dataset.folders[idx]["scene_name"]


def _find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for idx in range(len(dataset.folders)):
        if _scene_name_from_dataset(dataset, idx) == scene_name:
            return idx
    return 0


def _build_split_payload(dataset, scene_idx, scene_name, factor_entry, split):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    total_num = len(meta["camera_to_worlds"])
    if split == "train":
        sample_num = total_num if dataset.image_per_scene is None else min(dataset.image_per_scene, total_num)
        cam_ids = np.random.permutation(total_num)[:sample_num]
    elif split == "test":
        cam_ids = np.arange(total_num)
    else:
        raise ValueError(f"Unsupported split: {split}")

    images = [dataset.read_image(imgs_path[i], background=background) for i in cam_ids]
    cameras = {
        "camera_to_worlds": torch.as_tensor(meta["camera_to_worlds"][cam_ids]).float(),
        "fx": torch.as_tensor(meta["fx"]).float(),
        "fy": torch.as_tensor(meta["fy"]).float(),
        "cx": torch.as_tensor(meta["cx"]).float(),
        "cy": torch.as_tensor(meta["cy"]).float(),
        "width": torch.as_tensor(meta["width"]).float(),
        "height": torch.as_tensor(meta["height"]).float(),
        "background_color": background,
    }
    return {
        "gs_params": factor_entry["gs_params"],
        "images": images,
        "images_name": [imgs_name[i] for i in cam_ids],
        "cameras": cameras,
        "scene_idx": scene_idx,
        "scene_name": scene_name,
    }


def _resize_payload(payload, output_size):
    images, cameras = resize_images_and_cameras(payload["images"], payload["cameras"], output_size)
    resized = dict(payload)
    resized["images"] = images
    resized["cameras"] = cameras
    return resized


def _render_losses(gs, images, cameras, lpips_loss_func=None):
    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, cameras)
    if len(pred_imgs) == 0:
        raise ValueError("Training payload contains zero images")

    device = gs["means"].device
    image_l1 = torch.zeros((), device=device)
    lpips_loss = torch.zeros((), device=device)
    train_psnr = torch.zeros((), device=device)
    for pred_img, gt_img in zip(pred_imgs, images):
        gt_rgb = gt_img[..., :3]
        if gt_img.shape[-1] == 4:
            mask = gt_img[..., 3:].to(pred_img.dtype)
            pred_eval = pred_img * mask
            gt_eval = gt_rgb
            image_l1 = image_l1 + ((pred_img - gt_rgb) * mask).abs().mean()
        else:
            pred_eval = pred_img
            gt_eval = gt_rgb
            image_l1 = image_l1 + (pred_img - gt_rgb).abs().mean()
        train_psnr = train_psnr + psnr(pred_eval.unsqueeze(0), gt_eval.unsqueeze(0)).mean()
        if lpips_loss_func is not None:
            lpips_loss = lpips_loss + lpips_loss_func(
                pred_eval.unsqueeze(0), gt_eval.unsqueeze(0)
            ).mean()

    denominator = float(len(pred_imgs))
    return (
        image_l1 / denominator,
        lpips_loss / denominator,
        train_psnr / denominator,
        pred_imgs,
    )


def evaluate_single_scene(
    model,
    input_gs,
    num_rollout_steps,
    num_stages,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
):
    model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    predicted_keys = list(getattr(model, "output_features", []))
    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")
    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)
    residual_dir = os.path.join(output_dir, "residuals") if save_residuals else None
    if residual_dir is not None:
        os.makedirs(residual_dir, exist_ok=True)
    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, f"compare/{scene_name}") if compare_with_input else None
    if compare_dir is not None:
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = rollout_ar(model, input_gs, num_rollout_steps, num_stages)
        pred_preview = []
        gt_preview = []
        for start in range(0, num_views, eval_chunk_size):
            end = min(start + eval_chunk_size, num_views)
            chunk_images = gpu_utils.move_to_device(eval_images[start:end], device)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in eval_cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            gt_imgs = torch.stack(chunk_images, dim=0)

            if gt_imgs.shape[-1] == 4:
                masks = gt_imgs[..., 3].unsqueeze(-1)
                pred_imgs = pred_imgs * masks
                gt_imgs = (gt_imgs[..., :3] * 255).to(torch.uint8)
                pred_imgs = (pred_imgs * 255).to(torch.uint8)
            else:
                masks = None
                gt_imgs = (gt_imgs * 255).to(torch.uint8)
                pred_imgs = (pred_imgs * 255).to(torch.uint8)

            slots = 9 - len(pred_preview)
            if slots > 0:
                pred_preview.extend([im.cpu().numpy().astype(np.uint8) for im in pred_imgs[:slots]])
                if output_gt:
                    gt_preview.extend([im.cpu().numpy().astype(np.uint8) for im in gt_imgs[:slots]])

            metric_computer.update(pred_imgs, gt_imgs, name=f"{scene_idx}_{start:06d}")
            for name, pred_img in zip(image_names[start:end], pred_imgs):
                image = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(pred_single_dir, name), image[:, :, ::-1])

            if compare_with_input:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                input_imgs = (input_imgs * 255).to(torch.uint8)
                metric_computer_input.update(input_imgs, gt_imgs, name=f"{scene_idx}_{start:06d}")
                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    comparison = np.concatenate(
                        [
                            gt_img.cpu().numpy().astype(np.uint8),
                            input_img.cpu().numpy().astype(np.uint8),
                            pred_img.cpu().numpy().astype(np.uint8),
                        ],
                        axis=1,
                    )
                    cv2.imwrite(
                        os.path.join(compare_dir, f"{global_idx:04d}.png"), comparison[:, :, ::-1]
                    )

        if pred_preview:
            cv2.imwrite(
                os.path.join(output_dir, f"scene{scene_idx}_pred.png"),
                cv2.cvtColor(make_grid(pred_preview), cv2.COLOR_RGB2BGR),
            )
        if output_gt and gt_preview:
            cv2.imwrite(
                os.path.join(output_dir, f"scene{scene_idx}_gt.png"),
                cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR),
            )

        if save_viewer:
            viewer_dir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewer_dir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewer_dir, model.sh_degree)
            gs_utils.export_ply_forviewer(
                input_gs, os.path.join(viewer_dir, "point_cloud/iteration_0/point_cloud.ply")
            )
            gs_utils.export_ply_forviewer(
                out_gs, os.path.join(viewer_dir, "point_cloud/iteration_1/point_cloud.ply")
            )
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(
                    gt_gs, os.path.join(viewer_dir, "point_cloud/iteration_2/point_cloud.ply")
                )

        if save_residuals:
            residual_keys = [key for key in predicted_keys if key in out_gs and key in input_gs]
            if not residual_keys:
                residual_keys = sorted(key for key in out_gs if key in input_gs)
            residuals = {key: out_gs[key] - input_gs[key] for key in residual_keys}
            residual_stats = {
                key: {
                    "mean": float(value.mean().item()),
                    "abs_mean": float(value.abs().mean().item()),
                }
                for key, value in residuals.items()
            }
            scene_stem = f"{int(scene_idx)}_{_sanitize_for_filename(scene_name)}"
            torch.save(
                {
                    "scene_idx": int(scene_idx),
                    "scene_name": scene_name,
                    "residual_type": "ar_out_minus_input",
                    "num_rollout_steps": int(num_rollout_steps),
                    "num_stages": int(num_stages),
                    "residual_keys": residual_keys,
                    "residuals": _to_cpu(residuals),
                    "input_gs": _to_cpu(input_gs),
                    "output_gs": _to_cpu(out_gs),
                    "cameras": _to_cpu(eval_cameras),
                },
                os.path.join(residual_dir, f"{scene_stem}.pt"),
            )
            with open(os.path.join(residual_dir, f"{scene_stem}.json"), "w") as file:
                json.dump(
                    {
                        "scene_idx": int(scene_idx),
                        "scene_name": scene_name,
                        "num_gaussians": int(input_gs["means"].shape[0]),
                        "residual_type": "ar_out_minus_input",
                        "num_rollout_steps": int(num_rollout_steps),
                        "num_stages": int(num_stages),
                        "residual_keys": residual_keys,
                        "residual_stats": residual_stats,
                    },
                    file,
                    indent=2,
                )

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))
    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}
    model.train()
    return metrics, metrics_input


def _log_eval_metrics(logger, prefix, metrics, metrics_input):
    logger.info(f"{prefix}: " + " ".join(f"{key}: {value:.4f}" for key, value in metrics.items()))
    if FLAGS.compare_with_input:
        logger.info(
            f"{prefix} input: "
            + " ".join(f"{key}: {value:.4f}" for key, value in metrics_input.items())
        )


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as file:
        file.writelines(gin.operative_config_str())
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    if train_cfg["used_legacy_rollout_alias"]:
        logger.warning(
            "training.ar_steps_per_stage is deprecated; its value is being used as "
            "training.ar_num_rollouts"
        )
    device = torch.device("cuda")

    dataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][INPUT_FACTOR]
    target_factor_entry = scene["factor_data"][TARGET_FACTOR]
    source_size = (
        int(torch.as_tensor(input_factor_entry["meta"]["height"]).item()),
        int(torch.as_tensor(input_factor_entry["meta"]["width"]).item()),
    )
    target_size = (
        int(torch.as_tensor(target_factor_entry["meta"]["height"]).item()),
        int(torch.as_tensor(target_factor_entry["meta"]["width"]).item()),
    )
    schedule = build_resolution_schedule(source_size, target_size, train_cfg["ar_num_stages"])
    with open(os.path.join(FLAGS.output_dir, "ar_schedule.json"), "w") as file:
        json.dump(
            {
                "source_size": list(source_size),
                "target_size": list(target_size),
                "num_stages": train_cfg["ar_num_stages"],
                "num_rollouts": train_cfg["ar_num_rollouts"],
                "updates_per_rollout": train_cfg["ar_num_stages"],
                "total_steps": train_cfg["total_steps"],
                "legacy_total_steps": train_cfg["legacy_total_steps"],
                "stages": [
                    {**entry, "size": list(entry["size"])} for entry in schedule
                ],
            },
            file,
            indent=2,
        )

    if int(train_cfg["legacy_total_steps"]) != train_cfg["total_steps"]:
        logger.info(
            "Ignoring legacy training.total_steps=%s; AR schedule uses %s rollouts x %s stages = %s",
            train_cfg["legacy_total_steps"],
            train_cfg["ar_num_rollouts"],
            train_cfg["ar_num_stages"],
            train_cfg["total_steps"],
        )

    eval_hr_payload = _build_split_payload(
        dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test"
    )
    if len(eval_hr_payload["images"]) == 0:
        eval_hr_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
        )
    full_eval_payload = _resize_payload(eval_hr_payload, target_size)
    full_eval_chunk_size = (
        dataset.image_per_scene
        if dataset.image_per_scene is not None
        else len(full_eval_payload["images"])
    )

    batch_gs = gpu_utils.move_to_device([input_factor_entry["gs_params"]], device)
    model = GSFlowModel().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total_step=train_cfg["total_steps"])

    total_steps = train_cfg["total_steps"]
    resume_from_step = int(train_cfg["resume_from_step"])
    if resume_from_step < 0 or resume_from_step >= total_steps:
        raise ValueError(f"resume_from_step must be in [0, {total_steps}), got {resume_from_step}")
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["enable_amp"])
    lpips_loss_func = (
        loss_utils.lpips_loss_fn() if train_cfg["lpips_loss_weight"] > 0 else None
    )

    print(
        f"AR overfit scene={scene['scene_name']} idx={scene['idx']} "
        f"eval_views={len(eval_hr_payload['images'])} "
        f"gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"resolution={source_size}->{target_size} stages={train_cfg['ar_num_stages']} "
        f"rollouts={train_cfg['ar_num_rollouts']}"
    )
    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    current_gs = None
    train_hr_payload = None
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        rollout_index, stage_index = training_position(step, train_cfg["ar_num_stages"])
        stage_info = schedule[stage_index]
        t_value = stage_info["t"]
        output_h, output_w = stage_info["size"]

        if stage_index == 0:
            current_gs = detach_gs(batch_gs[0])
            train_hr_payload = _build_split_payload(
                dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
            )
        elif current_gs is None:
            # A resumed mid-rollout run has no cached handoff state. Rebuild the
            # prefix with the loaded model, then continue with a fresh view sample.
            current_gs = rollout_detached_prefix(
                model,
                batch_gs[0],
                stage_index=stage_index,
                num_stages=train_cfg["ar_num_stages"],
                enable_amp=train_cfg["enable_amp"],
            )
            train_hr_payload = _build_split_payload(
                dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
            )

        train_payload = _resize_payload(train_hr_payload, stage_info["size"])
        batch_cameras = gpu_utils.move_to_device(train_payload["cameras"], device)
        batch_images = gpu_utils.move_to_device(train_payload["images"], device)

        t = torch.tensor([t_value], device=device, dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=train_cfg["enable_amp"]):
            predicted_delta = model(batch_normalized_gs=[current_gs], timestep=t)[0]
            predicted_gs = apply_gs_delta(current_gs, predicted_delta)
            image_l1_raw, lpips_raw, train_psnr, pred_imgs = _render_losses(
                predicted_gs, batch_images, batch_cameras, lpips_loss_func
            )
            image_l1 = image_l1_raw * train_cfg["image_l1_loss_weight"]
            lpips_loss = lpips_raw * train_cfg["lpips_loss_weight"]
            total_loss = image_l1 + lpips_loss

        optimizer_step_applied = True
        if train_cfg["enable_amp"]:
            scale_before_update = scaler.get_scale()
            scaler.scale(total_loss).backward()
            if train_cfg["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_applied = scaler.get_scale() >= scale_before_update
        else:
            total_loss.backward()
            if train_cfg["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_step_applied:
            scheduler.step()
        completed_step = step + 1

        if stage_index + 1 < train_cfg["ar_num_stages"]:
            current_gs = recompute_detached_stage(
                model,
                current_gs,
                stage_index=stage_index,
                num_stages=train_cfg["ar_num_stages"],
                enable_amp=train_cfg["enable_amp"],
            )
        else:
            current_gs = None
            train_hr_payload = None

        pbar.set_postfix(
            {
                "rollout": f"{rollout_index + 1}/{train_cfg['ar_num_rollouts']}",
                "stage": f"{stage_index + 1}/{train_cfg['ar_num_stages']}",
                "t": f"{t_value:.2f}",
                "res": f"{output_w}x{output_h}",
                "loss": f"{total_loss.item():.4f}",
                "psnr": f"{train_psnr.item():.2f}",
                "skipped": str(not optimizer_step_applied),
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if train_cfg["empty_cache_fre"] > 0 and completed_step % train_cfg["empty_cache_fre"] == 0:
            torch.cuda.empty_cache()
        if completed_step % train_cfg["log_interval"] == 0:
            logger.info(
                "step=%s rollout=%s/%s stage=%s/%s t=%.4f resolution=%sx%s "
                "total=%.6f l1=%.6f "
                "lpips=%.6f psnr=%.4f optimizer_step_applied=%s lr=%.8f",
                completed_step,
                rollout_index + 1,
                train_cfg["ar_num_rollouts"],
                stage_index + 1,
                train_cfg["ar_num_stages"],
                t_value,
                output_w,
                output_h,
                total_loss.item(),
                image_l1.item(),
                lpips_loss.item(),
                train_psnr.item(),
                optimizer_step_applied,
                optimizer.param_groups[0]["lr"],
            )
        if completed_step % train_cfg["log_image_interval"] == 0:
            pred_uint8 = [(image * 255).detach().cpu().numpy().astype(np.uint8) for image in pred_imgs]
            cv2.imwrite(
                os.path.join(FLAGS.output_dir, "train", f"{completed_step:08d}_pred.png"),
                cv2.cvtColor(make_grid(pred_uint8), cv2.COLOR_RGB2BGR),
            )
            gt_uint8 = [
                (image[..., :3] * 255).detach().cpu().numpy().astype(np.uint8)
                for image in batch_images
            ]
            cv2.imwrite(
                os.path.join(FLAGS.output_dir, "train", f"{completed_step:08d}_gt.png"),
                cv2.cvtColor(make_grid(gt_uint8), cv2.COLOR_RGB2BGR),
            )
        if completed_step % train_cfg["eval_interval"] == 0:
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=batch_gs[0],
                num_rollout_steps=train_cfg["ar_num_stages"],
                num_stages=train_cfg["ar_num_stages"],
                scene_idx=full_eval_payload["scene_idx"],
                scene_name=full_eval_payload["scene_name"],
                eval_images=full_eval_payload["images"],
                eval_cameras=full_eval_payload["cameras"],
                image_names=full_eval_payload["images_name"],
                output_dir=os.path.join(FLAGS.output_dir, "eval", f"{completed_step:08d}"),
                eval_chunk_size=full_eval_chunk_size,
                gt_gs=target_factor_entry["gs_params"],
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                save_residuals=FLAGS.save_residuals,
                output_gt=True,
            )
            _log_eval_metrics(
                logger,
                f"Eval step {completed_step} rollout {rollout_index + 1} full_hr",
                metrics,
                metrics_input,
            )
        if completed_step % train_cfg["save_interval"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.output_dir, "checkpoints", f"model_{completed_step:08d}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))
    metrics, metrics_input = evaluate_single_scene(
        model=model,
        input_gs=batch_gs[0],
        num_rollout_steps=train_cfg["ar_num_stages"],
        num_stages=train_cfg["ar_num_stages"],
        scene_idx=full_eval_payload["scene_idx"],
        scene_name=full_eval_payload["scene_name"],
        eval_images=full_eval_payload["images"],
        eval_cameras=full_eval_payload["cameras"],
        image_names=full_eval_payload["images_name"],
        output_dir=os.path.join(FLAGS.output_dir, FLAGS.eval_subdir),
        eval_chunk_size=full_eval_chunk_size,
        gt_gs=target_factor_entry["gs_params"],
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
    )
    _log_eval_metrics(logger, "Final eval", metrics, metrics_input)


if __name__ == "__main__":
    app.run(main)

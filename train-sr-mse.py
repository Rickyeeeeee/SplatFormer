import json
import os
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_predictor import FeaturePredictor
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import scale_means_origin, unscale_means_origin
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS, feature_loss_value
from utils.metrics import MetricComputer
from utils.optimizers import build_3DGSoptimizer, build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_sr_mse", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("compare_with_input", True, "Compare predictions with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", False, "Save residual tensors and stats")
flags.DEFINE_boolean("use_wandb", True, "Log training and evaluation metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_integer("input_resolution", 128, "Low-resolution GS/image resolution used as alignment source")
flags.DEFINE_integer("target_resolution", 512, "High-resolution GS/image resolution used as training target")
flags.DEFINE_enum(
    "alignment",
    "emd",
    ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"],
    "Input/target Gaussian alignment method",
)
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_string(
    "matching_cache_root",
    "/project2/ricky/splatformer-data-to-4x",
    "Persistent cache for reverse fitted alignment",
)
flags.DEFINE_boolean(
    "post_activate_loss",
    False,
    "Use feature-specific loss transforms: sigmoid opacities and geodesic quats.",
)
flags.DEFINE_float(
    "means_origin_scale",
    1.0,
    "Training-only origin scale for target Gaussian means when computing means MSE loss. "
    "Predicted means are divided by this value before render/export.",
)
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

WANDB_EVAL_IMAGE_SCENE = "3e288ee8aced4a0797e66d53536112b1"
MEANS_LOSS_REDUCTION = "mean"


@gin.configurable
def set_seed(seed):
    seed_everything(seed)


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
):
    return {
        "output_dir": output_dir,
        "total_steps": total_steps,
        "pretrain_steps": pretrain_steps,
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


@gin.configurable
def matching_fit(
    total_steps=1000,
    image_per_step=16,
    log_interval=20,
    preview_interval=200,
    grad_clip_norm=0.0,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=1.0,
    enable_amp=True,
    empty_cache_fre=-1,
):
    return {
        "total_steps": total_steps,
        "image_per_step": image_per_step,
        "log_interval": log_interval,
        "preview_interval": preview_interval,
        "grad_clip_norm": grad_clip_norm,
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


def build_matching_optimization(gaussians):
    with gin.config_scope("matching_fit"):
        optimizer = build_3DGSoptimizer(gaussians)
        scheduler = build_scheduler(optimizer)
    return optimizer, scheduler


@gin.configurable
def feature_mse_loss(loss_weights=None, quat_direct_mse=False):
    default_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
    if loss_weights is None:
        loss_weights = {}
    unknown_keys = sorted(set(loss_weights) - set(SUPPORTED_GS_KEYS))
    if unknown_keys:
        raise ValueError(
            f"Unsupported feature_mse_loss.loss_weights keys: {unknown_keys}. "
            f"Supported keys are: {SUPPORTED_GS_KEYS}"
        )
    default_weights.update({key: float(value) for key, value in loss_weights.items()})
    return {"loss_weights": default_weights, "quat_direct_mse": quat_direct_mse}


def compute_all_feature_mse_loss(
    out_gs,
    target_gs,
    loss_weights,
    post_activate_loss,
    quat_direct_mse,
):
    losses = {}
    weighted_losses = {}
    total_loss = None
    for key in SUPPORTED_GS_KEYS:
        pred = out_gs[key]
        target = target_gs[key].to(
            device=pred.device, dtype=pred.dtype
        )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for Gaussian attribute {key!r}: "
                f"output {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        loss = feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=post_activate_loss,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=MEANS_LOSS_REDUCTION,
        )
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    return total_loss, losses, weighted_losses


def to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(to_cpu(v) for v in data)
    return data


def init_wandb(output_dir):
    if not FLAGS.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

    normalized_output_dir = output_dir.rstrip("/")
    output_parts = Path(normalized_output_dir).parts
    default_name = (
        "/".join(output_parts[-2:])
        if len(output_parts) >= 2
        else normalized_output_dir
    )
    return wandb.init(
        project=FLAGS.wandb_project,
        dir=FLAGS.wandb_dir,
        name=FLAGS.wandb_name or default_name,
        config={
            "output_dir": output_dir,
            "eval_subdir": FLAGS.eval_subdir,
            "compare_with_input": FLAGS.compare_with_input,
            "save_viewer": FLAGS.save_viewer,
            "save_residuals": FLAGS.save_residuals,
            "input_resolution": FLAGS.input_resolution,
            "target_resolution": FLAGS.target_resolution,
            "alignment": FLAGS.alignment,
            "matching_cache_root": FLAGS.matching_cache_root,
            "attribute_init": FLAGS.attribute_init,
            "loss_attributes": list(SUPPORTED_GS_KEYS),
            "post_activate_loss": FLAGS.post_activate_loss,
            "means_origin_scale": FLAGS.means_origin_scale,
            "gin_config": gin.operative_config_str(),
        },
    )


def wandb_log(data, step=None):
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def build_dataset(scope):
    dataset = SplatFactoSRDataset.from_gin_scope(scope)
    required_resolutions = {FLAGS.input_resolution, FLAGS.target_resolution}
    missing = sorted(required_resolutions - set(dataset.resolutions))
    if missing:
        raise ValueError(
            f"{scope} dataset resolutions {sorted(dataset.resolutions)} do not include required resolutions {missing}"
        )
    configured_pair = (
        dataset.fit_source_resolution, dataset.fit_target_resolution
    )
    requested_pair = (FLAGS.input_resolution, FLAGS.target_resolution)
    if configured_pair != requested_pair:
        raise ValueError(
            f"{scope} dataset fitted pair {configured_pair[0]}->{configured_pair[1]} "
            f"does not match requested pair {requested_pair[0]}->{requested_pair[1]}"
        )
    return dataset


def write_filtered_scenes(output_dir, datasets_by_split):
    filtered_scenes = []
    for split, dataset in datasets_by_split.items():
        for item in dataset.filtered_scenes:
            filtered_scenes.append({"split": split, **item})
    with open(os.path.join(output_dir, "filtered_scenes.json"), "w") as f:
        json.dump(filtered_scenes, f, indent=2)
    return filtered_scenes


def evaluate_single_scene(
    model,
    input_gs,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=False,
    save_residuals=False,
    output_gt=True,
    wandb_step=None,
    means_origin_scale=1.0,
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
    residual_dir = None
    if save_residuals:
        residual_dir = os.path.join(output_dir, "residuals")
        os.makedirs(residual_dir, exist_ok=True)

    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input:
        compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        input_gs_device = gpu_utils.move_to_device(input_gs, device)
        out_gs = model(batch_normalized_gs=[input_gs_device], batch_scene_idx=[scene_idx])[0]
        out_gs = unscale_means_origin(out_gs, means_origin_scale)

        pred_preview = []
        gt_preview = []
        compare_preview = []

        for start in range(0, num_views, eval_chunk_size):
            end = min(start + eval_chunk_size, num_views)
            chunk_name = f"{scene_idx}_{start:06d}"

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

            preview_slots = 9 - len(pred_preview)
            if preview_slots > 0:
                pred_preview.extend([im.cpu().numpy().astype(np.uint8) for im in pred_imgs[:preview_slots]])
                if output_gt:
                    gt_preview.extend([im.cpu().numpy().astype(np.uint8) for im in gt_imgs[:preview_slots]])

            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)

            for name, pred_img in zip(image_names[start:end], pred_imgs):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(pred_single_dir, name), pred_img[:, :, ::-1])

            if compare_with_input:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs_device, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                    input_imgs = (input_imgs * 255).to(torch.uint8)
                else:
                    input_imgs = (input_imgs * 255).to(torch.uint8)
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)

                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    gt_img = gt_img.cpu().numpy().astype(np.uint8)
                    input_img = input_img.cpu().numpy().astype(np.uint8)
                    pred_img = pred_img.cpu().numpy().astype(np.uint8)
                    cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1)
                    if len(compare_preview) < 9:
                        compare_preview.append(cmp_img)
                    cv2.imwrite(os.path.join(compare_dir, f"{global_idx:04d}.png"), cmp_img[:, :, ::-1])

        if len(pred_preview) > 0:
            pred_grid_rgb = gs_utils.make_grid(pred_preview)
            pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid_rgb = gs_utils.make_grid(gt_preview)
            gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        if wandb is not None and wandb.run is not None and scene_name == WANDB_EVAL_IMAGE_SCENE:
            wandb_images = {}
            if len(pred_preview) > 0:
                wandb_images[f"eval_images/{scene_name}/pred_grid"] = wandb.Image(
                    pred_grid_rgb,
                    caption=f"{scene_name} prediction",
                )
            if output_gt and len(gt_preview) > 0:
                wandb_images[f"eval_images/{scene_name}/gt_grid"] = wandb.Image(
                    gt_grid_rgb,
                    caption=f"{scene_name} ground truth",
                )
            if compare_with_input and len(compare_preview) > 0:
                compare_grid = gs_utils.make_grid(compare_preview)
                wandb_images[f"eval_images/{scene_name}/compare_grid"] = wandb.Image(
                    compare_grid,
                    caption=f"{scene_name} GT | input | pred",
                )
            if wandb_images:
                wandb_log(wandb_images, step=wandb_step)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, model.sh_degree)
            gs_utils.export_ply_forviewer(
                gs_params=input_gs_device,
                filename=os.path.join(viewerdir, "point_cloud/iteration_0/point_cloud.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(viewerdir, "point_cloud/iteration_1/point_cloud.ply"),
            )
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(
                    gs_params=gt_gs,
                    filename=os.path.join(viewerdir, "point_cloud/iteration_2/point_cloud.ply"),
                )

        if save_residuals:
            residual_type = "out_minus_input"
            residual_keys = [key for key in predicted_keys if key in out_gs and key in input_gs_device]
            if len(residual_keys) == 0:
                residual_keys = sorted([key for key in out_gs.keys() if key in input_gs_device])

            residuals = {}
            residual_stats = {}
            for key in residual_keys:
                residual = out_gs[key] - input_gs_device[key]
                residuals[key] = residual
                residual_stats[key] = {
                    "mean": float(residual.mean().item()),
                    "abs_mean": float(residual.abs().mean().item()),
                }

            scene_stem = f"{int(scene_idx)}_{gs_utils.sanitize_for_filename(scene_name)}"
            pt_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "residual_type": residual_type,
                "residual_keys": residual_keys,
                "residuals": to_cpu(residuals),
                "input_gs": to_cpu(input_gs_device),
                "output_gs": to_cpu(out_gs),
                "cameras": to_cpu(eval_cameras),
            }
            torch.save(pt_payload, os.path.join(residual_dir, f"{scene_stem}.pt"))

            stats_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "num_gaussians": int(input_gs_device["means"].shape[0]),
                "residual_type": residual_type,
                "residual_keys": residual_keys,
                "residual_stats": residual_stats,
            }
            with open(os.path.join(residual_dir, f"{scene_stem}.json"), "w") as f:
                json.dump(stats_payload, f, indent=2)

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))

    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}

    model.train()
    return metrics, metrics_input


def evaluate_dataset(
    model,
    dataset,
    output_dir,
    compare_with_input=True,
    save_viewer=False,
    save_residuals=False,
    output_gt=False,
    wandb_step=None,
    means_origin_scale=1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    all_metrics_input = []
    logger = ProcessSafeLogger(os.path.join(output_dir, "eval.log")).get_logger()
    device = next(model.parameters()).device
    matching_config = matching_fit()

    for scene_idx in tqdm(range(len(dataset.folders)), desc="Evaluating"):
        scene = dataset.load_scene(scene_idx)
        input_resolution_entry = scene["resolution_data"][FLAGS.input_resolution]
        target_resolution_entry = scene["resolution_data"][FLAGS.target_resolution]
        input_images, _, input_cameras = dataset.load_resolution_views(
            input_resolution_entry
        )
        eval_images, image_names, eval_cameras = dataset.load_resolution_views(
            target_resolution_entry
        )
        target_gs = gpu_utils.move_to_device(
            target_resolution_entry["gs_params"], device
        )
        scene_output_dir = os.path.join(output_dir, scene["scene_name"])
        input_gs, attribute_target_gs, alignment_info = prepare_alignment(
            dataset=dataset,
            scene=scene,
            input_resolution_entry=input_resolution_entry,
            target_resolution_entry=target_resolution_entry,
            target_images=eval_images,
            target_cameras=eval_cameras,
            target_gs=target_gs,
            output_dir=scene_output_dir,
            logger=logger,
            device=device,
            eval_chunk_size=len(eval_images),
            alignment=FLAGS.alignment,
            attribute_init=FLAGS.attribute_init,
            emd_eps=FLAGS.emd_eps,
            emd_iters=FLAGS.emd_iters,
            input_resolution=FLAGS.input_resolution,
            target_resolution=FLAGS.target_resolution,
            matching_cache_root=FLAGS.matching_cache_root,
            force_matching_fit=False,
            matching_config=matching_config,
            matching_optimizer_factory=build_matching_optimization,
            input_images=input_images,
            input_cameras=input_cameras,
            write_artifacts=False,
        )
        metrics, metrics_input = evaluate_single_scene(
            model=model,
            input_gs=input_gs,
            gt_gs=attribute_target_gs,
            scene_idx=scene["idx"],
            scene_name=scene["scene_name"],
            eval_images=eval_images,
            eval_cameras=eval_cameras,
            image_names=image_names,
            output_dir=scene_output_dir,
            eval_chunk_size=len(eval_images),
            compare_with_input=compare_with_input,
            save_viewer=save_viewer,
            save_residuals=save_residuals,
            output_gt=output_gt,
            wandb_step=wandb_step,
            means_origin_scale=means_origin_scale,
        )
        all_metrics.append(metrics)
        if compare_with_input:
            all_metrics_input.append(metrics_input)
        logger.info(
            f"Scene {scene['scene_name']} alignment={alignment_info}: "
            + " ".join(
                f"{key}: {value:.4f}" for key, value in metrics.items()
            )
        )

    reduced_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            reduced_metrics[key] = float(
                np.mean([metrics[key] for metrics in all_metrics])
            )

    reduced_metrics_input = {}
    if compare_with_input and all_metrics_input:
        for key in all_metrics_input[0]:
            reduced_metrics_input[key] = float(
                np.mean([metrics[key] for metrics in all_metrics_input])
            )

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(reduced_metrics, f, indent=2)
    if compare_with_input:
        with open(os.path.join(output_dir, "metrics_input.json"), "w") as f:
            json.dump(reduced_metrics_input, f, indent=2)

    return reduced_metrics, reduced_metrics_input


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.bind_parameter("training.output_dir", FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    mse_loss_cfg = feature_mse_loss()
    matching_config = matching_fit()
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger()
    wandb_run = init_wandb(FLAGS.output_dir)
    device = torch.device("cuda")

    train_dataset = build_dataset("train_dataset")
    test_dataset = build_dataset("test_dataset")
    filtered_scenes = write_filtered_scenes(
        FLAGS.output_dir,
        {"train": train_dataset, "test": test_dataset},
    )
    if filtered_scenes:
        logger.info(
            f"Saved {len(filtered_scenes)} filtered scenes to filtered_scenes.json"
        )

    model = FeaturePredictor().to(device)
    missing_outputs = sorted(
        set(SUPPORTED_GS_KEYS) - set(model.output_features)
    )
    if missing_outputs:
        raise ValueError(
            f"FeaturePredictor.output_features must include all Gaussian "
            f"attributes; missing {missing_outputs}"
        )
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
        logger.info(f"Loaded model checkpoint from {model.resume_ckpt}")

    if FLAGS.only_eval:
        model.eval()
    else:
        model.train()

    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = (
        requested_means_origin_scale
        if "means" in model.output_features
        else 1.0
    )

    training_brief = (
        f"Train SR MSE input_resolution={FLAGS.input_resolution} target_resolution={FLAGS.target_resolution}\n"
        f"train_scenes={len(train_dataset.folders)} test_scenes={len(test_dataset.folders)}\n"
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init}\n"
        f"matching_cache_root={FLAGS.matching_cache_root}\n"
        f"loss_attributes={','.join(SUPPORTED_GS_KEYS)}\n"
        f"post_activate_loss={FLAGS.post_activate_loss}\n"
        f"means_origin_scale={requested_means_origin_scale}\n"
        f"effective_means_origin_scale={means_origin_scale}\n"
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']}\n"
        f"loss_weights={mse_loss_cfg['loss_weights']}"
    )
    print(training_brief)
    logger.info(training_brief)

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]
    # Keep render-loss training gin keys accepted for compatibility; this script optimizes GS MSE only.
    _ = train_cfg["image_l1_loss_weight"]
    _ = train_cfg["lpips_loss_weight"]
    _ = train_cfg["pretrain_steps"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    if not FLAGS.only_eval:
        optimizer.zero_grad(set_to_none=True)
        train_iter = iter(train_dataset)
        pbar = tqdm(range(resume_from_step, total_steps), desc="Training")
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                batch = next(train_iter)
            input_resolution_entry = batch["multiresolution"][FLAGS.input_resolution]
            target_resolution_entry = batch["multiresolution"][FLAGS.target_resolution]

            target_gs = gpu_utils.move_to_device(
                target_resolution_entry["gs_params"], device
            )
            aligned_input_gs, attribute_target_gs, alignment_info = prepare_alignment(
                dataset=train_dataset,
                scene={
                    "scene_name": batch["scene_name"],
                    "fit_lr_to_hr": batch["fit_lr_to_hr"],
                },
                input_resolution_entry=input_resolution_entry,
                target_resolution_entry=target_resolution_entry,
                target_images=target_resolution_entry["images"],
                target_cameras=target_resolution_entry["cameras"],
                target_gs=target_gs,
                output_dir=os.path.join(FLAGS.output_dir, "train", batch["scene_name"]),
                logger=logger,
                device=device,
                eval_chunk_size=len(target_resolution_entry["images"]),
                alignment=FLAGS.alignment,
                attribute_init=FLAGS.attribute_init,
                emd_eps=FLAGS.emd_eps,
                emd_iters=FLAGS.emd_iters,
                input_resolution=FLAGS.input_resolution,
                target_resolution=FLAGS.target_resolution,
                matching_cache_root=FLAGS.matching_cache_root,
                force_matching_fit=False,
                matching_config=matching_config,
                matching_optimizer_factory=build_matching_optimization,
                input_images=input_resolution_entry["images"],
                input_cameras=input_resolution_entry["cameras"],
                write_artifacts=False,
            )
            loss_target_gs = scale_means_origin(
                attribute_target_gs, means_origin_scale
            )
            batch_gs = [aligned_input_gs]
            batch_scene_idx = [batch["scene_idx"]]

            with torch.cuda.amp.autocast(enabled=enable_amp):
                out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                total_loss, feature_losses, weighted_feature_losses = compute_all_feature_mse_loss(
                    out_gs=out_gs,
                    target_gs=loss_target_gs,
                    loss_weights=mse_loss_cfg["loss_weights"],
                    post_activate_loss=FLAGS.post_activate_loss,
                    quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                )

            if enable_amp:
                scaler.scale(total_loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            feature_loss_values = {key: loss.item() for key, loss in feature_losses.items()}
            weighted_loss_values = {key: loss.item() for key, loss in weighted_feature_losses.items()}
            pbar.set_postfix(
                {
                    "scene": batch["scene_name"],
                    "loss": f"{total_loss.item():.3e}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
                torch.cuda.empty_cache()

            if step % log_interval == 0:
                train_log = {
                    "train/total_loss": total_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/scene_idx": batch["scene_idx"],
                    "train/input_gaussians": input_resolution_entry["gs_params"]["means"].shape[0],
                    "train/aligned_gaussians": aligned_input_gs["means"].shape[0],
                    "train/target_gaussians": attribute_target_gs["means"].shape[0],
                }
                for key, value in feature_loss_values.items():
                    train_log[f"train/{key}_loss"] = value
                    train_log[f"train/{key}_weighted"] = weighted_loss_values[key]
                wandb_log(train_log, step=step)

                feature_loss_str = " ".join(
                    f"{key}_loss={value:.6f} {key}_weighted={weighted_loss_values[key]:.6f}"
                    for key, value in feature_loss_values.items()
                )
                logger.info(
                    f"step={step} scene={batch['scene_name']} total={total_loss.item():.6f} "
                    f"alignment={alignment_info} {feature_loss_str} lr={optimizer.param_groups[0]['lr']:.8f}"
                )

            if step % log_image_interval == 0:
                batch_cameras = gpu_utils.move_to_device(target_resolution_entry["cameras"], device)
                batch_images = gpu_utils.move_to_device(target_resolution_entry["images"], device)
                with torch.no_grad():
                    log_out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                    log_out_gs = unscale_means_origin(log_out_gs, means_origin_scale)
                    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(log_out_gs, batch_cameras)

                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
                pred_grid_rgb = gs_utils.make_grid(pred_imgs_uint8)
                pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
                pred_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png")
                cv2.imwrite(pred_path, pred_grid)

                gt_imgs_uint8 = [
                    (img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in batch_images
                ]
                gt_grid_rgb = gs_utils.make_grid(gt_imgs_uint8)
                gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
                gt_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_gt.png")
                cv2.imwrite(gt_path, gt_grid)

                if wandb is not None and wandb.run is not None:
                    wandb_log(
                        {
                            "train/pred_grid": wandb.Image(pred_grid_rgb, caption=f"step={step} pred"),
                            "train/gt_grid": wandb.Image(gt_grid_rgb, caption=f"step={step} gt"),
                        },
                        step=step,
                    )

            if step % eval_interval == 0:
                eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
                metrics, metrics_input = evaluate_dataset(
                    model=model,
                    dataset=test_dataset,
                    output_dir=eval_dir,
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    save_residuals=FLAGS.save_residuals,
                    output_gt=(step == 0),
                    wandb_step=step,
                    means_origin_scale=means_origin_scale,
                )
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Eval step {step}: {metric_str}")
                wandb_log({f"eval/{key}": value for key, value in metrics.items()}, step=step)
                if FLAGS.compare_with_input:
                    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                    logger.info(f"Eval step {step} input: {metric_str}")
                    wandb_log(
                        {f"eval_input/{key}": value for key, value in metrics_input.items()},
                        step=step,
                    )
                model.train()

            if (step + 1) % save_interval == 0:
                ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved model checkpoint to {ckpt_path}")

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = evaluate_dataset(
        model=model,
        dataset=test_dataset,
        output_dir=final_eval_dir,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
        means_origin_scale=means_origin_scale,
    )
    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    wandb_log({f"final_eval/{key}": value for key, value in metrics.items()})
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")
        wandb_log({f"final_eval_input/{key}": value for key, value in metrics_input.items()})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)

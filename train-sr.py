import json
import os
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
import torch.distributed as dist
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_sr", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("compare_with_input", True, "Compare predictions with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", False, "Save residual tensors and stats")
flags.DEFINE_boolean("use_wandb", True, "Log training and evaluation metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_integer("input_resolution", 128, "Low-resolution GS/image resolution used as model input")
flags.DEFINE_integer("target_resolution", 512, "High-resolution GS/image resolution used as training target")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

WANDB_EVAL_IMAGE_SCENE = "3e288ee8aced4a0797e66d53536112b1"


@gin.configurable
def set_seed(seed, rank=0):
    seed_everything(seed + rank)


def reduce_mean(value):
    reduced = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
    return reduced


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


def to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(to_cpu(value) for value in data)
    return data


def init_wandb(output_dir):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if not FLAGS.use_wandb or rank != 0:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

    output_dir_parts = Path(output_dir.rstrip("/")).parts
    return wandb.init(
        project=FLAGS.wandb_project,
        dir=FLAGS.wandb_dir,
        name=FLAGS.wandb_name or (
            "/".join(output_dir_parts[-2:]) if len(output_dir_parts) >= 2 else output_dir
        ),
        config={
            "output_dir": output_dir,
            "eval_subdir": FLAGS.eval_subdir,
            "compare_with_input": FLAGS.compare_with_input,
            "save_viewer": FLAGS.save_viewer,
            "save_residuals": FLAGS.save_residuals,
            "input_resolution": FLAGS.input_resolution,
            "target_resolution": FLAGS.target_resolution,
            "gin_config": gin.operative_config_str(),
        },
    )

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


def normalize_filtered_scene(split, filtered_scene):
    reason = filtered_scene.get("reason")
    exception_reason = filtered_scene.get("exception_reason")
    return {
        "split": split,
        "scene_idx": filtered_scene.get("scene_idx"),
        "scene_name": filtered_scene.get("scene_name"),
        "reason": reason,
        "exception_reason": exception_reason,
        "exception_type": filtered_scene.get("exception_type"),
        "gsplat_dir": filtered_scene.get("gsplat_dir"),
        "colmap_dir": filtered_scene.get("colmap_dir"),
    }


def write_filtered_scenes(output_dir, datasets_by_split, extra_filtered_scenes=None):
    filtered_scenes = []
    for split, dataset in datasets_by_split.items():
        for filtered_scene in getattr(dataset, "filtered_scenes", []):
            filtered_scenes.append(normalize_filtered_scene(split, filtered_scene))

    if extra_filtered_scenes is not None:
        filtered_scenes.extend(extra_filtered_scenes)

    with open(os.path.join(output_dir, "filtered_scenes.json"), "w") as f:
        json.dump(filtered_scenes, f, indent=2)
    return filtered_scenes


def compute_render_loss(pred_imgs, batch_images, lpips_loss_func, image_l1_loss_weight, lpips_loss_weight):
    image_l1 = 0
    lpips_loss = 0
    train_psnr = 0
    num_images = len(pred_imgs)
    if num_images == 0:
        raise ValueError("Cannot compute render loss on zero images")

    for pred_img, gt_img in zip(pred_imgs, batch_images):
        gt_rgb = gt_img[..., :3]
        image_l1 += (pred_img - gt_rgb).abs().mean()
        train_psnr += psnr(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()
        if lpips_loss_func is not None:
            lpips_loss += lpips_loss_func(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()

    image_l1 = image_l1 / num_images * image_l1_loss_weight
    train_psnr = train_psnr / num_images
    total_loss = image_l1
    if lpips_loss_func is not None:
        lpips_loss = lpips_loss / num_images * lpips_loss_weight
        total_loss = total_loss + lpips_loss
    return total_loss, image_l1, lpips_loss, train_psnr


def image_png_name(image_names, image_id):
    if image_id < len(image_names):
        image_name = os.path.basename(str(image_names[image_id]))
        stem, ext = os.path.splitext(image_name)
        if ext.lower() == ".png":
            return image_name
        if stem:
            return f"{stem}.png"
    return f"{image_id:04d}.png"


def append_image_metric_records(records, metric_computer, previous_counts, start, image_names):
    metric_values = {}
    for metric, values in metric_computer.results.items():
        new_values = values[previous_counts[metric]:]
        metric_values[metric] = torch.cat([value.reshape(-1) for value in new_values]).detach().cpu().tolist()

    num_records = len(metric_values["psnr"])
    for offset in range(num_records):
        image_id = start + offset
        records.append(
            {
                "image_id": image_id,
                "image_name": image_png_name(image_names, image_id),
                "psnr": float(metric_values["psnr"][offset]),
                "ssim": float(metric_values["ssim"][offset]),
                "lpips": float(metric_values["lpips"][offset]),
            }
        )


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
    low_res_gt_gs=None,
    evaluate_baselines=False,
    compare_with_input=True,
    save_viewer=False,
    save_residuals=False,
    output_gt=True,
    wandb_step=None,
):
    model.eval()
    model_module = model.module if isinstance(model, DDP) else model
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if evaluate_baselines else None
    metric_computer_gt_low_res = MetricComputer() if evaluate_baselines else None
    metric_computer_gt_high_res = MetricComputer() if evaluate_baselines else None
    predicted_keys = list(getattr(model_module, "output_features", []))

    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)
    pred_single_dir = os.path.join(output_dir, "pred")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input and evaluate_baselines:
        compare_dir = os.path.join(output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)

    residual_dir = None
    if save_residuals:
        residual_dir = os.path.join(output_dir, "residuals")
        os.makedirs(residual_dir, exist_ok=True)

    image_metrics = []
    image_metrics_input = []
    image_metrics_gt_low_res = []
    image_metrics_gt_high_res = []

    with torch.no_grad():
        input_gs_device = gpu_utils.move_to_device(input_gs, device)
        gt_gs_device = (
            gpu_utils.move_to_device(gt_gs, device) if evaluate_baselines or save_viewer else None
        )
        low_res_gt_gs_device = (
            gpu_utils.move_to_device(low_res_gt_gs, device) if evaluate_baselines else None
        )
        out_gs = model(batch_normalized_gs=[input_gs_device], batch_scene_idx=[scene_idx])[0]

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

            metric_counts = {
                metric: len(values) for metric, values in metric_computer.results.items()
            }
            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)
            append_image_metric_records(image_metrics, metric_computer, metric_counts, start, image_names)

            if evaluate_baselines:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs_device, chunk_cameras)
                low_res_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(low_res_gt_gs_device, chunk_cameras)
                gt_high_res_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gt_gs_device, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                low_res_imgs = torch.stack(low_res_imgs, dim=0)
                gt_high_res_imgs = torch.stack(gt_high_res_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                    low_res_imgs = low_res_imgs * masks
                    gt_high_res_imgs = gt_high_res_imgs * masks
                input_imgs = (input_imgs * 255).to(torch.uint8)
                low_res_imgs = (low_res_imgs * 255).to(torch.uint8)
                gt_high_res_imgs = (gt_high_res_imgs * 255).to(torch.uint8)

                input_counts = {
                    metric: len(values) for metric, values in metric_computer_input.results.items()
                }
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)
                append_image_metric_records(
                    image_metrics_input, metric_computer_input, input_counts, start, image_names
                )
                low_res_counts = {
                    metric: len(values)
                    for metric, values in metric_computer_gt_low_res.results.items()
                }
                metric_computer_gt_low_res.update(low_res_imgs, gt_imgs, name=chunk_name)
                append_image_metric_records(
                    image_metrics_gt_low_res, metric_computer_gt_low_res,
                    low_res_counts, start, image_names
                )
                high_res_counts = {
                    metric: len(values)
                    for metric, values in metric_computer_gt_high_res.results.items()
                }
                metric_computer_gt_high_res.update(gt_high_res_imgs, gt_imgs, name=chunk_name)
                append_image_metric_records(
                    image_metrics_gt_high_res, metric_computer_gt_high_res,
                    high_res_counts, start, image_names
                )

            for global_idx, pred_img in enumerate(pred_imgs, start=start):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(
                    os.path.join(pred_single_dir, image_png_name(image_names, global_idx)),
                    pred_img[:, :, ::-1],
                )

            if compare_with_input and evaluate_baselines:
                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    gt_img = gt_img.cpu().numpy().astype(np.uint8)
                    input_img = input_img.cpu().numpy().astype(np.uint8)
                    pred_img = pred_img.cpu().numpy().astype(np.uint8)
                    cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1)
                    if len(compare_preview) < 9:
                        compare_preview.append(cmp_img)
                    cv2.imwrite(
                        os.path.join(compare_dir, image_png_name(image_names, global_idx)),
                        cmp_img[:, :, ::-1],
                    )

        if len(pred_preview) > 0:
            pred_grid_rgb = gs_utils.make_grid(pred_preview)
            pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid_rgb = gs_utils.make_grid(gt_preview)
            gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0 and wandb is not None and wandb.run is not None and scene_name == WANDB_EVAL_IMAGE_SCENE:
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
            if compare_with_input and evaluate_baselines and len(compare_preview) > 0:
                compare_grid = gs_utils.make_grid(compare_preview)
                wandb_images[f"eval_images/{scene_name}/compare_grid"] = wandb.Image(
                    compare_grid,
                    caption=f"{scene_name} GT | input | pred",
                )
            if wandb_images:
                wandb.log(wandb_images, step=wandb_step)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, model_module.sh_degree)
            gs_utils.export_ply_forviewer(
                gs_params=input_gs_device,
                filename=os.path.join(viewerdir, "point_cloud/00_input_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(viewerdir, "point_cloud/01_output_gs.ply"),
            )
            if gt_gs_device is not None:
                gs_utils.export_ply_forviewer(
                    gs_params=gt_gs_device,
                    filename=os.path.join(viewerdir, "point_cloud/02_gt_gs.ply"),
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
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(image_metrics, f, indent=2)
    if evaluate_baselines:
        metrics_input = metric_computer_input.finalize()
        with open(os.path.join(output_dir, "metrics_input.json"), "w") as f:
            json.dump(image_metrics_input, f, indent=2)
        metrics_gt_low_res = metric_computer_gt_low_res.finalize()
        with open(os.path.join(output_dir, "metrics_gt_low_res.json"), "w") as f:
            json.dump(image_metrics_gt_low_res, f, indent=2)
        metrics_gt_high_res = metric_computer_gt_high_res.finalize()
        with open(os.path.join(output_dir, "metrics_gt_high_res.json"), "w") as f:
            json.dump(image_metrics_gt_high_res, f, indent=2)
    else:
        metrics_input = {}
        metrics_gt_low_res = {}
        metrics_gt_high_res = {}

    model.train()
    return metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res


def evaluate_dataset(
    model,
    dataset,
    output_dir,
    compare_with_input=True,
    save_viewer=False,
    save_residuals=False,
    output_gt=False,
    wandb_step=None,
    evaluate_baselines=False,
):
    os.makedirs(output_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    all_metrics = []
    all_metrics_input = []
    all_metrics_gt_low_res = []
    all_metrics_gt_high_res = []
    scene_average_metrics = []
    eval_filtered_scenes = []
    logger_name = "eval.log" if rank == 0 else f"eval.rank{rank}.log"
    logger = ProcessSafeLogger(os.path.join(output_dir, logger_name)).get_logger()

    scene_indices = range(rank, len(dataset.folders), world_size)
    for scene_idx in tqdm(scene_indices, desc=f"Evaluating rank {rank}", disable=rank != 0):
        scene_info = dataset.folders[scene_idx]
        scene_name = scene_info["scene_name"]
        try:
            scene = dataset.load_scene(scene_idx)
            input_resolution_entry = scene["resolution_data"][FLAGS.input_resolution]
            target_resolution_entry = scene["resolution_data"][FLAGS.target_resolution]
            eval_images, image_names, eval_cameras = dataset.load_resolution_views(target_resolution_entry)
            input_gs = gs_utils.convert_gaussian_frame(
                input_resolution_entry["gs_params"],
                input_resolution_entry["scaler"],
                target_resolution_entry["scaler"],
            )

            scene_output_dir = os.path.join(output_dir, scene["scene_name"])
            metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_single_scene(
                model=model,
                input_gs=input_gs,
                gt_gs=target_resolution_entry["gs_params"],
                low_res_gt_gs=input_gs,
                evaluate_baselines=evaluate_baselines,
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
            )
        except Exception as exc:
            model.train()
            filtered_scene = normalize_filtered_scene(
                "test",
                {
                    "scene_idx": scene_idx,
                    "scene_name": scene_name,
                    "reason": "eval_exception",
                    "exception_reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "gsplat_dir": scene_info.get("resolution_paths", {})
                    .get(FLAGS.target_resolution, {})
                    .get("gsplat_dir"),
                    "colmap_dir": scene_info.get("resolution_paths", {})
                    .get(FLAGS.target_resolution, {})
                    .get("colmap_dir"),
                },
            )
            eval_filtered_scenes.append(filtered_scene)
            logger.exception(f"Filtering eval scene {scene_name} after exception")
            continue

        all_metrics.append(metrics)
        if evaluate_baselines:
            all_metrics_gt_low_res.append(metrics_gt_low_res)
            all_metrics_gt_high_res.append(metrics_gt_high_res)
            all_metrics_input.append(metrics_input)
        scene_metrics = {
            "scene_idx": scene["idx"],
            "scene_name": scene["scene_name"],
            "output_gs": metrics,
        }
        if evaluate_baselines:
            scene_metrics.update({
                "gt_low_res_gs": metrics_gt_low_res,
                "input_gs": metrics_input,
                "gt_high_res_gs": metrics_gt_high_res,
            })
        scene_average_metrics.append(scene_metrics)
        logger.info(
            f"Scene {scene['scene_name']}: "
            + " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        )
        if evaluate_baselines:
            logger.info(f"Scene {scene['scene_name']} GT low-res GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()]
            ))
            logger.info(f"Scene {scene['scene_name']} input GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_input.items()]
            ))
            logger.info(f"Scene {scene['scene_name']} GT high-res GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()]
            ))

    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "scene_metrics": scene_average_metrics,
                "filtered_scenes": eval_filtered_scenes,
            },
        )
        scene_average_metrics = sorted(
            [item for payload in gathered for item in payload["scene_metrics"]],
            key=lambda item: item["scene_idx"],
        )
        eval_filtered_scenes = [
            item for payload in gathered for item in payload["filtered_scenes"]
        ]

    all_metrics = [item["output_gs"] for item in scene_average_metrics]
    if evaluate_baselines:
        all_metrics_input = [item["input_gs"] for item in scene_average_metrics]
        all_metrics_gt_low_res = [item["gt_low_res_gs"] for item in scene_average_metrics]
        all_metrics_gt_high_res = [item["gt_high_res_gs"] for item in scene_average_metrics]

    reduced_metrics = {}
    if len(all_metrics) > 0:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            reduced_metrics[key] = float(np.mean([metrics[key] for metrics in all_metrics]))

    reduced_metrics_input = {}
    if evaluate_baselines and len(all_metrics_input) > 0:
        metric_keys = all_metrics_input[0].keys()
        for key in metric_keys:
            reduced_metrics_input[key] = float(np.mean([metrics[key] for metrics in all_metrics_input]))

    reduced_metrics_gt_low_res = {}
    if evaluate_baselines and len(all_metrics_gt_low_res) > 0:
        metric_keys = all_metrics_gt_low_res[0].keys()
        for key in metric_keys:
            reduced_metrics_gt_low_res[key] = float(np.mean(
                [metrics[key] for metrics in all_metrics_gt_low_res]
            ))

    reduced_metrics_gt_high_res = {}
    if evaluate_baselines and len(all_metrics_gt_high_res) > 0:
        metric_keys = all_metrics_gt_high_res[0].keys()
        for key in metric_keys:
            reduced_metrics_gt_high_res[key] = float(np.mean(
                [metrics[key] for metrics in all_metrics_gt_high_res]
            ))

    if rank == 0:
        write_filtered_scenes(output_dir, {"test": dataset}, eval_filtered_scenes)
        with open(os.path.join(output_dir, "scene_average_metrics.json"), "w") as f:
            json.dump(scene_average_metrics, f, indent=2)

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(reduced_metrics, f, indent=2)
        if evaluate_baselines:
            with open(os.path.join(output_dir, "metrics_input.json"), "w") as f:
                json.dump(reduced_metrics_input, f, indent=2)
            with open(os.path.join(output_dir, "metrics_gt_low_res.json"), "w") as f:
                json.dump(reduced_metrics_gt_low_res, f, indent=2)
            with open(os.path.join(output_dir, "metrics_gt_high_res.json"), "w") as f:
                json.dump(reduced_metrics_gt_high_res, f, indent=2)

    return (
        reduced_metrics, reduced_metrics_input,
        reduced_metrics_gt_low_res, reduced_metrics_gt_high_res,
    )


def main(argv):
    del argv
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.bind_parameter("training.output_dir", FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    set_seed(rank=rank)

    logger = (
        ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger()
        if rank == 0
        else None
    )
    wandb_run = init_wandb(FLAGS.output_dir)

    train_dataset = build_dataset("train_dataset")
    test_dataset = build_dataset("test_dataset")
    filtered_scenes = []
    if rank == 0:
        filtered_scenes = write_filtered_scenes(
            FLAGS.output_dir,
            {"train": train_dataset, "test": test_dataset},
        )
    if filtered_scenes and rank == 0:
        logger.info(f"Saved {len(filtered_scenes)} filtered scenes to filtered_scenes.json")

    model = FeaturePredictor()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
        if rank == 0:
            logger.info(f"Loaded model checkpoint from {model.resume_ckpt}")
    model = model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    model_module = model.module if isinstance(model, DDP) else model

    if FLAGS.only_eval:
        model.eval()
    else:
        model.train()

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model_module)
        scheduler = build_scheduler(optimizer)

    if rank == 0:
        with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
            f.writelines(gin.operative_config_str())

    training_brief = (
        f"Train SR input_resolution={FLAGS.input_resolution} target_resolution={FLAGS.target_resolution}\n"
        f"train_scenes={len(train_dataset.folders)} test_scenes={len(test_dataset.folders)}\n"
        f"world_size={world_size}\n"
        f"model_input_features={','.join(model_module.input_features)}\n"
        f"model_output_features={','.join(model_module.output_features)}"
    )
    if rank == 0:
        print(training_brief)
        logger.info(training_brief)

    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]
    image_l1_loss_weight = train_cfg["image_l1_loss_weight"]
    lpips_loss_weight = train_cfg["lpips_loss_weight"]
    _ = train_cfg["pretrain_steps"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if lpips_loss_weight > 0 else None

    if rank == 0:
        os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)
    if distributed:
        dist.barrier()

    if not FLAGS.only_eval:
        optimizer.zero_grad(set_to_none=True)
        train_iter = iter(train_dataset)
        pbar = tqdm(
            range(resume_from_step, total_steps),
            desc="Training",
            disable=rank != 0,
        )
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                batch = next(train_iter)
            input_resolution_entry = batch["multiresolution"][FLAGS.input_resolution]
            target_resolution_entry = batch["multiresolution"][FLAGS.target_resolution]

            input_gs = gpu_utils.move_to_device(input_resolution_entry["gs_params"], device)
            input_gs = gs_utils.convert_gaussian_frame(
                input_gs,
                input_resolution_entry["scaler"],
                target_resolution_entry["scaler"],
            )
            batch_scene_idx = [batch["scene_idx"]]
            batch_cameras = gpu_utils.move_to_device(target_resolution_entry["cameras"], device)
            batch_images = gpu_utils.move_to_device(target_resolution_entry["images"], device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                out_gs = model(batch_normalized_gs=[input_gs], batch_scene_idx=batch_scene_idx)[0]
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, batch_cameras)
                total_loss, image_l1, lpips_loss, train_psnr = compute_render_loss(
                    pred_imgs,
                    batch_images,
                    lpips_loss_func,
                    image_l1_loss_weight,
                    lpips_loss_weight,
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

            lpips_value = lpips_loss.item() if lpips_loss_func is not None else 0.0
            if rank == 0:
                pbar.set_postfix(
                    {
                        "scene": batch["scene_name"],
                        "loss": f"{total_loss.item():.4f}",
                        "l1": f"{image_l1.item():.4f}",
                        "lpips": f"{lpips_value:.4f}",
                        "psnr": f"{train_psnr.item():.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

            if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
                torch.cuda.empty_cache()

            if step % log_interval == 0:
                lpips_tensor = lpips_loss if torch.is_tensor(lpips_loss) else total_loss.new_tensor(0.0)
                reduced_values = {
                    "total_loss": reduce_mean(total_loss),
                    "image_l1": reduce_mean(image_l1),
                    "lpips": reduce_mean(lpips_tensor),
                    "psnr": reduce_mean(train_psnr),
                    "input_gaussians": reduce_mean(
                        total_loss.new_tensor(input_resolution_entry["gs_params"]["means"].shape[0])
                    ),
                    "target_gaussians": reduce_mean(
                        total_loss.new_tensor(target_resolution_entry["gs_params"]["means"].shape[0])
                    ),
                    "views": reduce_mean(total_loss.new_tensor(len(batch_images))),
                }
                if rank == 0:
                    train_log = {
                        f"train/{key}": value.item() for key, value in reduced_values.items()
                    }
                    train_log["train/lr"] = optimizer.param_groups[0]["lr"]
                    if wandb is not None and wandb.run is not None:
                        wandb.log(train_log, step=step)
                    logger.info(
                        f"step={step} total={reduced_values['total_loss'].item():.6f} "
                        f"l1={reduced_values['image_l1'].item():.6f} "
                        f"lpips={reduced_values['lpips'].item():.6f} "
                        f"psnr={reduced_values['psnr'].item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}"
                    )

            if step % log_image_interval == 0 and rank == 0:
                with torch.no_grad():
                    log_pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, batch_cameras)

                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in log_pred_imgs]
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
                    wandb.log(
                        {
                            "train/pred_grid": wandb.Image(pred_grid_rgb, caption=f"step={step} pred"),
                            "train/gt_grid": wandb.Image(gt_grid_rgb, caption=f"step={step} gt"),
                        },
                        step=step,
                    )

            if step % eval_interval == 0:
                eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
                metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_dataset(
                    model=model,
                    dataset=test_dataset,
                    output_dir=eval_dir,
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    save_residuals=FLAGS.save_residuals,
                    output_gt=(step == 0),
                    wandb_step=step,
                    evaluate_baselines=(step == 0),
                )
                if rank == 0:
                    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                    logger.info(f"Eval step {step}: {metric_str}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log({f"eval/{key}": value for key, value in metrics.items()}, step=step)
                    if step == 0:
                        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_input.items()])
                        logger.info(f"Eval step {step} input: {metric_str}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log({f"eval_input/{key}": value for key, value in metrics_input.items()}, step=step)
                        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()])
                        logger.info(f"Eval step {step} GT low-res GS: {metric_str}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log({f"eval_gt_low_res/{key}": value for key, value in metrics_gt_low_res.items()}, step=step)
                        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()])
                        logger.info(f"Eval step {step} GT high-res GS: {metric_str}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log({f"eval_gt_high_res/{key}": value for key, value in metrics_gt_high_res.items()}, step=step)
                if distributed:
                    dist.barrier()
                model.train()

            if (step + 1) % save_interval == 0:
                if rank == 0:
                    ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth")
                    torch.save(model_module.state_dict(), ckpt_path)
                    logger.info(f"Saved model checkpoint to {ckpt_path}")
                if distributed:
                    dist.barrier()

        if rank == 0:
            last_ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth")
            torch.save(model_module.state_dict(), last_ckpt_path)
            logger.info(f"Saved model checkpoint to {last_ckpt_path}")
        if distributed:
            dist.barrier()

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_dataset(
        model=model,
        dataset=test_dataset,
        output_dir=final_eval_dir,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
        evaluate_baselines=True,
    )
    if rank == 0:
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        logger.info(f"Final eval: {metric_str}")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"final_eval/{key}": value for key, value in metrics.items()})
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"final_eval_input/{key}": value for key, value in metrics_input.items()})
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()])
        logger.info(f"Final eval GT low-res GS: {metric_str}")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"final_eval_gt_low_res/{key}": value for key, value in metrics_gt_low_res.items()})
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()])
        logger.info(f"Final eval GT high-res GS: {metric_str}")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"final_eval_gt_high_res/{key}": value for key, value in metrics_gt_high_res.items()})

    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)

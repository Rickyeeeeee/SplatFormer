"""Fixed-scene SR overfitting with attribute and optional render supervision."""

import json
import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_predictor import FeaturePredictor
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import (
    SUPPORTED_GS_KEYS,
    compute_gaussian_attribute_loss,
    gaussian_attribute_loss_config,
    load_gs_statistics_normalizers,
)
from utils.metrics import MetricComputer, render_gs_average_metrics
from utils.optimizers import build_optimizer, build_scheduler

flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit in one mode")
flags.DEFINE_enum("scene_mode", "one", ["one", "many"], "Overfit one scene or a fixed random scene set")
flags.DEFINE_integer("scene_count", 1, "Number of scenes selected in many mode")
flags.DEFINE_integer("batch_size", 1, "Total scene samples per optimizer update")
flags.DEFINE_integer("grad_accum_steps", 1, "Forward/backward passes used to split one effective batch")
flags.register_validator("batch_size", lambda value: value >= 1, message="batch_size must be at least 1")
flags.register_multi_flags_validator(
    ["batch_size", "grad_accum_steps"],
    lambda values: 1 <= values["grad_accum_steps"] <= values["batch_size"],
    message="grad_accum_steps must be between 1 and batch_size",
)
flags.DEFINE_boolean("compare_with_input", False, "Compare with aligned input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_enum("alignment", "emd", ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"], "Matching input/target Gaussian pair")
flags.DEFINE_enum("attribute_init", "aligned", ["aligned", "3dgs"], "For emd and random only")
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_boolean("post_activate_loss", True, "Post activation loss")
flags.DEFINE_string("gs_statistics_path", None, "Channel-normalized attribute MSE")
flags.DEFINE_multi_string("gin_file", None, "List of paths to Gin config files")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter bindings")

FLAGS = flags.FLAGS


@gin.configurable
def set_seed(seed):
    seed_everything(seed)
    return seed


def select_scene_indices(dataset, scene_mode, scene_name, scene_count, seed):
    """Select a stable set of unique scene identities."""
    if scene_mode == "one":
        return [dataset.scene_index(scene_name)]
    unique_scenes = {}
    for index, entry in enumerate(dataset.folders):
        unique_scenes.setdefault(entry["scene_name"], index)
    if not 1 <= scene_count <= len(unique_scenes):
        raise ValueError(f"scene_count must be between 1 and {len(unique_scenes)}, got {scene_count}")
    return np.random.default_rng(seed).choice(list(unique_scenes.values()), size=scene_count, replace=False).tolist()


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
    save_viewer=True,
    output_gt=True,
):
    """Render one predicted scene and write image metrics and viewer artifacts."""
    model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation has zero views")
    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)
    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)
    compare_dir = None
    if compare_with_input:
        compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = model(batch_normalized_gs=[input_gs], batch_scene_idx=[scene_idx])[0]
        pred_preview = []
        gt_preview = []
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
                pred_preview.extend(image.cpu().numpy().astype(np.uint8) for image in pred_imgs[:preview_slots])
                if output_gt:
                    gt_preview.extend(image.cpu().numpy().astype(np.uint8) for image in gt_imgs[:preview_slots])
            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)
            for name, pred_img in zip(image_names[start:end], pred_imgs):
                cv2.imwrite(os.path.join(pred_single_dir, name), pred_img.cpu().numpy().astype(np.uint8)[:, :, ::-1])

            if compare_with_input:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                input_imgs = (input_imgs * 255).to(torch.uint8)
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)
                for global_idx, (gt_img, input_img, pred_img) in enumerate(zip(gt_imgs, input_imgs, pred_imgs), start=start):
                    comparison = np.concatenate([
                        gt_img.cpu().numpy().astype(np.uint8),
                        input_img.cpu().numpy().astype(np.uint8),
                        pred_img.cpu().numpy().astype(np.uint8),
                    ], axis=1)
                    cv2.imwrite(os.path.join(compare_dir, f"{global_idx:04d}.png"), comparison[:, :, ::-1])

        if pred_preview:
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), cv2.cvtColor(make_grid(pred_preview), cv2.COLOR_RGB2BGR))
        if output_gt and gt_preview:
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR))
        if save_viewer:
            viewer_dir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewer_dir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewer_dir, model.sh_degree)
            point_cloud_dir = os.path.join(viewer_dir, "point_cloud")
            gs_utils.export_ply_forviewer(input_gs, os.path.join(point_cloud_dir, "input.ply"))
            gs_utils.export_ply_forviewer(out_gs, os.path.join(point_cloud_dir, "output.ply"))
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(gt_gs, os.path.join(point_cloud_dir, "gt.ply"))

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))
    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}
    model.train()
    return metrics, metrics_input


def prepare_overfit_scene(dataset, scene, output_dir, logger, device, gs_statistics_path, loss_config):
    """Prepare fixed alignment, loss normalization, and render baselines."""
    os.makedirs(output_dir, exist_ok=True)
    input_resolution = dataset.src_resolution
    target_resolution = dataset.tgt_resolution
    if scene["coordinate_frame"] != "input_resolution":
        raise ValueError("SR dev overfitting requires input_resolution coordinates")
    input_data = scene["data"][input_resolution]
    target_data = scene["data"][target_resolution]
    target_images = target_data["images"]
    target_image_names = target_data["images_name"]
    target_cameras = target_data["cameras"]
    target_gs = gpu_utils.move_to_device(target_data["gs_params"], device)
    eval_chunk_size = dataset.image_per_scene or len(target_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(target_images)

    if FLAGS.alignment == "fit_lr_to_hr":
        aligned_input_gs = gpu_utils.move_to_device(input_data["gs_params"], device)
        attribute_target_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
        alignment_info = {"status": "dataset_preloaded", "direction": FLAGS.alignment}
    elif FLAGS.alignment == "fit_hr_to_lr":
        aligned_input_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
        attribute_target_gs = target_gs
        alignment_info = {"status": "dataset_preloaded", "direction": FLAGS.alignment}
    else:
        aligned_input_gs, attribute_target_gs, alignment_info = prepare_alignment(
            dataset=dataset,
            scene=scene,
            input_resolution_entry=input_data,
            target_resolution_entry=target_data,
            target_images=target_images,
            target_cameras=target_cameras,
            target_gs=target_gs,
            output_dir=output_dir,
            logger=logger,
            device=device,
            eval_chunk_size=eval_chunk_size,
            alignment=FLAGS.alignment,
            attribute_init=FLAGS.attribute_init,
            emd_eps=FLAGS.emd_eps,
            emd_iters=FLAGS.emd_iters,
            input_resolution=input_resolution,
            target_resolution=target_resolution,
        )

    component_normalizers = None
    selected_statistics = None
    effective_loss_weights = dict(loss_config.loss_weights)
    if gs_statistics_path is not None:
        component_normalizers, selected_statistics = load_gs_statistics_normalizers(
            gs_statistics_path,
            target_resolution,
            SUPPORTED_GS_KEYS,
            attribute_target_gs,
            alignment=FLAGS.alignment,
            resolutions=[input_resolution, target_resolution],
        )
        effective_loss_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}

    metric_devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=metric_devices):
        original_input_gs = gpu_utils.move_to_device(input_data["gs_params"], device)
        source_metrics = render_gs_average_metrics(original_input_gs, target_images, target_cameras, eval_chunk_size, device)
        target_metrics = render_gs_average_metrics(target_gs, target_images, target_cameras, eval_chunk_size, device)
        matching_metrics = target_metrics if attribute_target_gs is target_gs else render_gs_average_metrics(
            attribute_target_gs, target_images, target_cameras, eval_chunk_size, device
        )

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    gt_images = [(image[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for image in target_images[:9]]
    if gt_images:
        cv2.imwrite(os.path.join(train_dir, "00000000_gt.png"), cv2.cvtColor(make_grid(gt_images), cv2.COLOR_RGB2BGR))
    logger.info(
        f"Prepared scene={scene['scene_name']} idx={scene['scene_idx']} alignment={FLAGS.alignment} "
        f"input_gaussians={input_data['gs_params']['means'].shape[0]} aligned_gaussians={aligned_input_gs['means'].shape[0]} "
        f"target_gaussians={attribute_target_gs['means'].shape[0]} alignment_info={alignment_info}"
    )
    return {
        "scene_idx": scene["scene_idx"],
        "scene_name": scene["scene_name"],
        "source_gs": aligned_input_gs,
        "target_gs": attribute_target_gs,
        "component_normalizers": component_normalizers,
        "effective_loss_weights": effective_loss_weights,
        "target_images": target_images,
        "target_cameras": target_cameras,
        "target_image_names": target_image_names,
        "eval_chunk_size": eval_chunk_size,
        "render_view_count": min(dataset.image_per_scene or len(target_images), len(target_images)),
        "train_dir": train_dir,
        "baseline_metrics": {
            "scene_idx": scene["scene_idx"],
            "view_count": len(target_images),
            "source_gs": source_metrics,
            "target_gs": target_metrics,
            "matching_target_gs": matching_metrics,
        },
        "loss_statistics": {
            "configured_loss_weights": dict(loss_config.loss_weights),
            "effective_loss_weights": effective_loss_weights,
            "selected_statistics": selected_statistics,
            "effective_normalizers": None if component_normalizers is None else {
                key: value.detach().cpu().tolist() for key, value in component_normalizers.items()
            },
        },
    }


def compute_microbatch_loss(model, scenes, device, loss_config, image_l1_loss_weight, lpips_loss_weight, lpips_loss_func, enable_amp):
    """Compute summed scene losses while keeping only one microbatch on the GPU."""
    samples = []
    render_enabled = image_l1_loss_weight > 0 or lpips_loss_weight > 0
    for scene in scenes:
        source = gpu_utils.move_to_device(scene["source_gs"], device)
        target = gpu_utils.move_to_device(scene["target_gs"], device)
        normalizers = gpu_utils.move_to_device(scene["component_normalizers"], device)
        images, cameras = None, None
        if render_enabled:
            view_count = scene["render_view_count"]
            if view_count <= 0:
                raise ValueError(f"Scene {scene['scene_name']!r} has no render-loss views")
            indices = np.random.permutation(len(scene["target_images"]))[:view_count]
            images = gpu_utils.move_to_device([scene["target_images"][index] for index in indices], device)
            cameras = dict(scene["target_cameras"])
            cameras["camera_to_worlds"] = cameras["camera_to_worlds"][indices]
            cameras = gpu_utils.move_to_device(cameras, device)
        samples.append({"source": source, "target": target, "normalizers": normalizers, "images": images, "cameras": cameras})

    summed_loss = None
    statistics = {}
    with torch.cuda.amp.autocast(enabled=enable_amp):
        outputs = model(
            batch_normalized_gs=[sample["source"] for sample in samples],
            batch_scene_idx=[scene["scene_idx"] for scene in scenes],
        )
        for scene, sample, out_gs in zip(scenes, samples, outputs):
            attribute_loss, feature_losses, weighted_feature_losses = compute_gaussian_attribute_loss(
                out_gs=out_gs,
                target_gs=sample["target"],
                loss_weights=scene["effective_loss_weights"],
                post_activate_loss=FLAGS.post_activate_loss,
                quat_direct_mse=loss_config.quat_direct_mse,
                means_loss_reduction=loss_config.means_loss_reduction,
                component_normalizers=sample["normalizers"],
            )
            render_l1 = attribute_loss.new_zeros(())
            render_lpips = attribute_loss.new_zeros(())
            if render_enabled:
                pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, sample["cameras"])
                render_l1_terms = []
                render_lpips_terms = []
                for pred_image, target_image in zip(pred_images, sample["images"]):
                    target_rgb = target_image[..., :3]
                    if target_image.shape[-1] == 4:
                        pred_image = pred_image * target_image[..., 3].unsqueeze(-1)
                    render_l1_terms.append((pred_image - target_rgb).abs().mean())
                    if lpips_loss_func is not None:
                        render_lpips_terms.append(lpips_loss_func(pred_image.unsqueeze(0), target_rgb.unsqueeze(0)).mean())
                render_l1 = torch.stack(render_l1_terms).mean()
                if render_lpips_terms:
                    render_lpips = torch.stack(render_lpips_terms).mean()
            weighted_render_l1 = float(image_l1_loss_weight) * render_l1
            weighted_render_lpips = float(lpips_loss_weight) * render_lpips
            total_loss = attribute_loss + weighted_render_l1 + weighted_render_lpips
            summed_loss = total_loss if summed_loss is None else summed_loss + total_loss
            values = {
                "total": total_loss,
                "attribute_loss": attribute_loss,
                "render_l1": render_l1,
                "weighted_render_l1": weighted_render_l1,
                "render_lpips": render_lpips,
                "weighted_render_lpips": weighted_render_lpips,
                "sampled_views": 0 if sample["images"] is None else len(sample["images"]),
            }
            values.update({f"{key}_loss": value for key, value in feature_losses.items()})
            values.update({f"{key}_weighted": value for key, value in weighted_feature_losses.items()})
            for key, value in values.items():
                scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                statistics[key] = statistics.get(key, 0.0) + scalar
    return summed_loss, statistics


def write_aggregate_metrics(output_dir, filename, per_scene):
    """Write equal-scene aggregate evaluation metrics."""
    mean = {
        key: sum(values[key] for values in per_scene.values()) / len(per_scene)
        for key in next(iter(per_scene.values()))
    }
    with open(os.path.join(output_dir, filename), "w") as metric_file:
        json.dump({"mean": mean, "scenes": per_scene}, metric_file, indent=2)
        metric_file.write("\n")
    return mean


@gin.configurable
def training(
    dataset,
    scene_indices,
    output_dir,
    gs_statistics_path,
    logger,
    device,
    total_steps=gin.REQUIRED,
    eval_interval=gin.REQUIRED,
    log_interval=gin.REQUIRED,
    save_interval=gin.REQUIRED,
    log_image_interval=gin.REQUIRED,
    grad_clip_norm=gin.REQUIRED,
    image_l1_loss_weight=0.0,
    lpips_loss_weight=0.0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
):
    loss_config = gaussian_attribute_loss_config()
    many = FLAGS.scene_mode == "many"
    prepared_scenes = []
    baseline_scene_metrics = {}
    loss_scene_statistics = {}
    for scene_idx in scene_indices:
        scene_name = dataset.folders[scene_idx]["scene_name"]
        scene_dir = os.path.join(output_dir, "scenes", scene_name) if many else output_dir
        try:
            scene = dataset.load_scene(scene_idx, fit_alignment=FLAGS.alignment)
            prepared = prepare_overfit_scene(dataset, scene, scene_dir, logger, device, gs_statistics_path, loss_config)
        except Exception as error:
            raise RuntimeError(f"Failed to prepare scene {scene_name!r} (index {scene_idx})") from error
        if not prepared["target_images"]:
            raise ValueError(f"Scene {scene_name!r} has no target views")
        baseline_scene_metrics[scene_name] = prepared.pop("baseline_metrics")
        loss_scene_statistics[scene_name] = prepared.pop("loss_statistics")
        prepared_scenes.append(gpu_utils.to_cpu(prepared) if many else prepared)

    baseline_mean = {
        gs_key: {
            metric: sum(values[gs_key][metric] for values in baseline_scene_metrics.values()) / len(baseline_scene_metrics)
            for metric in ("psnr", "ssim", "lpips")
        }
        for gs_key in ("source_gs", "target_gs", "matching_target_gs")
    }
    baseline_report = {
        "source_resolution": dataset.src_resolution,
        "target_resolution": dataset.tgt_resolution,
        "evaluation_resolution": dataset.tgt_resolution,
        "alignment": FLAGS.alignment,
        "scenes": baseline_scene_metrics,
        "mean": baseline_mean,
    }
    with open(os.path.join(output_dir, "baseline_render_metrics.json"), "w") as baseline_file:
        json.dump(baseline_report, baseline_file, indent=2)
        baseline_file.write("\n")
    loss_report = {
        "loss_type": "attribute_mse_plus_render",
        "post_activate_loss": FLAGS.post_activate_loss,
        "quat_direct_mse": loss_config.quat_direct_mse,
        "means_loss_reduction": loss_config.means_loss_reduction,
        "gs_statistics_path": gs_statistics_path,
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "scenes": loss_scene_statistics,
    }
    with open(os.path.join(output_dir, "mse_loss_statistics.json"), "w") as statistics_file:
        json.dump(loss_report, statistics_file, indent=2)
        statistics_file.write("\n")

    model = FeaturePredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
        logger.info(f"Loaded model checkpoint from {model.resume_ckpt}")
    model.train()
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)
    with open(os.path.join(output_dir, "config.gin"), "w") as handle:
        handle.write(gin.operative_config_str())
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    batch_size = FLAGS.batch_size
    grad_accum_steps = FLAGS.grad_accum_steps
    quotient, remainder = divmod(batch_size, grad_accum_steps)
    microbatch_sizes = [quotient + (index < remainder) for index in range(grad_accum_steps)]
    render_enabled = image_l1_loss_weight > 0 or lpips_loss_weight > 0
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if lpips_loss_weight > 0 else None
    logger.info(
        f"SR-MSE scenes={[(scene['scene_name'], scene['scene_idx']) for scene in prepared_scenes]} "
        f"batch_size={batch_size} grad_accum_steps={grad_accum_steps} microbatch_sizes={microbatch_sizes} "
        f"render_enabled={render_enabled} image_l1_loss_weight={image_l1_loss_weight} lpips_loss_weight={lpips_loss_weight}"
    )

    scene_order = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps), desc="SR-MSE")
    for step in pbar:
        batch_scenes = []
        for _ in range(batch_size):
            if many:
                if not scene_order:
                    scene_order = np.random.permutation(len(prepared_scenes)).tolist()
                batch_scenes.append(prepared_scenes[scene_order.pop()])
            else:
                batch_scenes.append(prepared_scenes[0])

        batch_statistics = {}
        offset = 0
        for microbatch_size in microbatch_sizes:
            microbatch_loss, statistics = compute_microbatch_loss(
                model,
                batch_scenes[offset:offset + microbatch_size],
                device,
                loss_config,
                image_l1_loss_weight,
                lpips_loss_weight,
                lpips_loss_func,
                enable_amp,
            )
            microbatch_loss = microbatch_loss / batch_size
            if enable_amp:
                scaler.scale(microbatch_loss).backward()
            else:
                microbatch_loss.backward()
            for key, value in statistics.items():
                batch_statistics[key] = batch_statistics.get(key, 0.0) + value / batch_size
            offset += microbatch_size

        optimizer_stepped = True
        if enable_amp:
            previous_scale = scaler.get_scale()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        else:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_stepped:
            scheduler.step()

        pbar.set_postfix(loss=f"{batch_statistics['total']:.3e}", attr=f"{batch_statistics['attribute_loss']:.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()
        if step % log_interval == 0:
            identities = [(scene["scene_name"], scene["scene_idx"]) for scene in batch_scenes]
            values = " ".join(f"{key}={value:.6f}" for key, value in batch_statistics.items())
            logger.info(
                f"step={step} scenes={identities} batch_size={batch_size} grad_accum_steps={grad_accum_steps} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {values}"
            )

        if step % log_image_interval == 0:
            active = batch_scenes[0]
            model.eval()
            with torch.no_grad():
                source = gpu_utils.move_to_device(active["source_gs"], device)
                preview_gs = model(batch_normalized_gs=[source], batch_scene_idx=[active["scene_idx"]])[0]
                preview_cameras = dict(active["target_cameras"])
                preview_cameras["camera_to_worlds"] = preview_cameras["camera_to_worlds"][:9]
                preview_cameras = gpu_utils.move_to_device(preview_cameras, device)
                pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(preview_gs, preview_cameras)
                pred_images = [(image * 255).detach().cpu().numpy().astype(np.uint8) for image in pred_images]
                if pred_images:
                    cv2.imwrite(os.path.join(active["train_dir"], f"{step:08d}_pred.png"), cv2.cvtColor(make_grid(pred_images), cv2.COLOR_RGB2BGR))
            model.train()

        is_final_step = step == total_steps - 1
        if step % eval_interval == 0 or is_final_step:
            eval_dir = os.path.join(output_dir, FLAGS.eval_subdir) if is_final_step else os.path.join(output_dir, "eval", f"{step:08d}")
            scene_metrics = {}
            scene_input_metrics = {}
            for evaluated in prepared_scenes:
                scene_name = evaluated["scene_name"]
                scene_eval_dir = os.path.join(eval_dir, "scenes", scene_name) if many else eval_dir
                metrics, metrics_input = evaluate_single_scene(
                    model=model,
                    input_gs=gpu_utils.move_to_device(evaluated["source_gs"], device),
                    gt_gs=gpu_utils.move_to_device(evaluated["target_gs"], device),
                    scene_idx=evaluated["scene_idx"],
                    scene_name=scene_name,
                    eval_images=evaluated["target_images"],
                    eval_cameras=evaluated["target_cameras"],
                    image_names=evaluated["target_image_names"],
                    output_dir=scene_eval_dir,
                    eval_chunk_size=evaluated["eval_chunk_size"],
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    output_gt=step == 0 or is_final_step,
                )
                scene_metrics[scene_name] = metrics
                scene_input_metrics[scene_name] = metrics_input
                logger.info(f"Eval step={step} scene={scene_name}: {metrics}")
            if many:
                os.makedirs(eval_dir, exist_ok=True)
                mean = write_aggregate_metrics(eval_dir, "metrics.json", scene_metrics)
                logger.info(f"Eval step={step} scene_mean={mean}")
                if FLAGS.compare_with_input:
                    input_mean = write_aggregate_metrics(eval_dir, "metrics_input.json", scene_input_metrics)
                    logger.info(f"Eval input step={step} scene_mean={input_mean}")
            if is_final_step:
                print(f"Final eval: {scene_metrics if many else next(iter(scene_metrics.values()))}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "model_last.pth"))


def main(argv):
    del argv
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    if FLAGS.gs_statistics_path is not None and FLAGS.post_activate_loss:
        raise ValueError("--gs_statistics_path cannot be used with --post_activate_loss")
    seed = set_seed()
    logger = ProcessSafeLogger(os.path.join(output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = SplatFactoSRDataset.from_gin_scope("test_dataset")
    if not all((dataset.load_src_gs, dataset.load_tgt_gs, dataset.load_src_images, dataset.load_tgt_images)):
        raise ValueError("SR dev overfitting requires every source and target payload")
    scene_indices = select_scene_indices(dataset, FLAGS.scene_mode, FLAGS.scene_name, FLAGS.scene_count, seed)
    selection = {
        "scene_mode": FLAGS.scene_mode,
        "seed": seed,
        "scenes": [{"scene_name": dataset.folders[index]["scene_name"], "scene_idx": index} for index in scene_indices],
    }
    with open(os.path.join(output_dir, "selected_scenes.json"), "w") as selection_file:
        json.dump(selection, selection_file, indent=2)
        selection_file.write("\n")
    logger.info(f"Selected scenes: {selection}")
    training(
        dataset=dataset,
        scene_indices=scene_indices,
        output_dir=output_dir,
        gs_statistics_path=FLAGS.gs_statistics_path,
        logger=logger,
        device=device,
    )


if __name__ == "__main__":
    app.run(main)

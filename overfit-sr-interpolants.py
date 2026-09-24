import json
import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_SR import SplatFactoSRDataset
from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy Gin keys.
from sr import flow, interpolants
from utils.gs_normalization import GaussianStandardizer
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer, render_gs_average_metrics
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit_interpolants", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit in one mode")
flags.DEFINE_enum("predictor", "ptv3", ["ptv3", "dipt"], "Gaussian velocity predictor")
flags.DEFINE_enum("scene_mode", "one", ["one", "many"], "Overfit one scene or a fixed random scene set")
flags.DEFINE_integer("scene_count", 1, "Number of scenes selected in many mode")
flags.DEFINE_integer("batch_size", 1, "Total scene samples per optimizer update")
flags.DEFINE_integer("grad_accum_steps", 1, "Forward/backward passes per batch; splits batch_size into smaller microbatches")
flags.register_validator("batch_size", lambda value: value >= 1, message="batch_size must be at least 1")
flags.register_multi_flags_validator(
    ["batch_size", "grad_accum_steps"],
    lambda values: 1 <= values["grad_accum_steps"] <= values["batch_size"],
    message="grad_accum_steps must be between 1 and batch_size",
)
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_enum("alignment", "emd", ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"], "Matching modes")
flags.DEFINE_enum("attribute_init", "aligned", ["aligned", "3dgs"], "attribute initialization for emd and random.",)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
PREDICTORS = {"ptv3": GSFlowPredictor, "dipt": DiffusionGaussianPredictor}
EVAL_FLOW_STEPS = [50]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.


@gin.configurable
def set_seed(seed):
    seed_everything(seed)
    return seed


@gin.configurable
def flow_matching(flow_steps=10, flow_noise_std=1.0, flow_t_eps=1e-4, loss_type="velocity",
                  interpolant_type="linear", loss_rollout_steps=10, eval_noise_seed=0,
                  fixed_train_noise=False, train_noise_seed=0,
                  normalization_variance_floor=1e-8,
                  gs_statistics_path="/project/ricky/splatformer-sr-data-scaled/gs_statistics.json"):
    if loss_type not in ("velocity", "x1"):
        raise ValueError(f"Unsupported flow_matching.loss_type={loss_type!r}")
    interpolants.validate_settings(interpolant_type, flow_steps, flow_t_eps)
    interpolants.validate_settings(interpolant_type, loss_rollout_steps, flow_t_eps)
    return {"interpolant_type": interpolant_type, "loss_rollout_steps": loss_rollout_steps,
            "eval_noise_seed": eval_noise_seed, "fixed_train_noise": fixed_train_noise,
            "train_noise_seed": train_noise_seed, "flow_steps": flow_steps, "flow_noise_std": flow_noise_std, "flow_t_eps": flow_t_eps,
            "loss_type": loss_type, "normalization_variance_floor": normalization_variance_floor,
            "gs_statistics_path": gs_statistics_path}


@gin.configurable
def loss_mixing(schedule="linear"):
    valid_schedules = ("linear", "free-range-gs", "fm-only")
    if schedule not in valid_schedules:
        raise ValueError(
            f"Unsupported loss_mixing.schedule={schedule!r}; "
            f"expected one of {valid_schedules}"
        )
    return {"schedule": schedule}


def evaluate_single_scene(
    model,
    input_gs,
    source_flow_gs,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    flow_steps,
    standardizer,
    flow_cfg,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    output_gt=True,
):
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
        out_gs = interpolants.sample_flow_model(model, source_flow_gs, scene_idx, flow_steps, standardizer,
                                               mode=flow_cfg["interpolant_type"], target_means=gt_gs["means"],
                                               t_eps=flow_cfg["flow_t_eps"], noise_seed=flow_cfg["eval_noise_seed"])
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
                pred_preview.extend([im.cpu().numpy().astype(np.uint8) for im in pred_imgs[:preview_slots]])
                if output_gt:
                    gt_preview.extend([im.cpu().numpy().astype(np.uint8) for im in gt_imgs[:preview_slots]])

            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)

            for name, pred_img in zip(image_names[start:end], pred_imgs):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(pred_single_dir, name), pred_img[:, :, ::-1])

            if compare_with_input:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
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
                    cv2.imwrite(os.path.join(compare_dir, f"{global_idx:04d}.png"), cmp_img[:, :, ::-1])

        if len(pred_preview) > 0:
            pred_grid = cv2.cvtColor(make_grid(pred_preview), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid = cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, model.sh_degree)
            gs_utils.export_ply_forviewer(input_gs, os.path.join(viewerdir, "point_cloud/input.ply"))
            gs_utils.export_ply_forviewer(out_gs, os.path.join(viewerdir, "point_cloud/output.ply"))
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(gt_gs, os.path.join(viewerdir, "point_cloud/gt.ply"))

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))

    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}

    model.train()
    return metrics, metrics_input


def prepare_overfit_scene(dataset, scene, output_dir, logger, device, standardizer):
    """Prepare fixed alignment and normalization once for a scene."""
    os.makedirs(output_dir, exist_ok=True)
    input_resolution = dataset.src_resolution
    target_resolution = dataset.tgt_resolution
    if scene["coordinate_frame"] != "input_resolution":
        raise ValueError("SR dev overfitting requires input_resolution coordinates")
    input_resolution_entry = scene["data"][input_resolution]
    target_resolution_entry = scene["data"][target_resolution]

    target_images = target_resolution_entry["images"]
    target_image_names = target_resolution_entry["images_name"]
    target_cameras = target_resolution_entry["cameras"]

    target_gs = gpu_utils.move_to_device(target_resolution_entry["gs_params"], device)
    eval_chunk_size = dataset.image_per_scene or len(target_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(target_images)

    if FLAGS.alignment == "fit_lr_to_hr":
        source_gs = gpu_utils.move_to_device(input_resolution_entry["gs_params"], device)
        matching_target_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
        alignment_info = {"status": "dataset_preloaded", "direction": FLAGS.alignment}
    elif FLAGS.alignment == "fit_hr_to_lr":
        source_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
        matching_target_gs = target_gs
        alignment_info = {"status": "dataset_preloaded", "direction": FLAGS.alignment}
    else:
        source_gs, matching_target_gs, alignment_info = prepare_alignment(
            dataset=dataset,
            scene=scene,
            input_resolution_entry=input_resolution_entry,
            target_resolution_entry=target_resolution_entry,
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

    loss_target_gs = matching_target_gs
    source_flow_gs = standardizer.encode(source_gs)
    target_flow_gs = standardizer.encode(loss_target_gs)
    logger.info(
        f"Scene={scene['scene_name']} idx={scene['scene_idx']} "
        f"input_resolution={input_resolution} target_resolution={target_resolution} "
        f"input_gaussians={input_resolution_entry['gs_params']['means'].shape[0]} "
        f"matching_target_gaussians={matching_target_gs['means'].shape[0]} "
        f"coordinate_frame={scene['coordinate_frame']} "
        f"coordinate_frame_version={scene['coordinate_frame_version']} "
        f"coordinate_resolution={scene['coordinate_resolution']} "
        f"alignment_info={alignment_info} attribute_init={FLAGS.attribute_init} "
        f"fit_lr_to_hr_root={dataset.fit_lr_to_hr_root} fit_hr_to_lr_root={dataset.fit_hr_to_lr_root}"
    )
    # Compare fixed Gaussian baselines against the same HR views without changing training RNG state.
    metric_devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=metric_devices):
        original_source_gs = gpu_utils.move_to_device(input_resolution_entry["gs_params"], device)
        source_metrics = render_gs_average_metrics(original_source_gs, target_images, target_cameras, eval_chunk_size, device)
        target_metrics = render_gs_average_metrics(target_gs, target_images, target_cameras, eval_chunk_size, device)
        matching_metrics = target_metrics if matching_target_gs is target_gs else render_gs_average_metrics(
            matching_target_gs, target_images, target_cameras, eval_chunk_size, device
        )
    baseline_metrics = {
        "scene_idx": scene["scene_idx"], "view_count": len(target_images),
        "source_gs": source_metrics, "target_gs": target_metrics, "matching_target_gs": matching_metrics,
    }
    logger.info(f"Baseline render metrics scene={scene['scene_name']}: {baseline_metrics}")
    return {
        "baseline_metrics": baseline_metrics,
        "scene_idx": scene["scene_idx"], "scene_name": scene["scene_name"],
        "source_gs": source_gs, "source_flow_gs": source_flow_gs,
        "target_flow_gs": target_flow_gs, "target_gs": loss_target_gs,
        "target_images": target_images, "target_cameras": target_cameras,
        "target_image_names": target_image_names, "eval_chunk_size": eval_chunk_size,
    }


def compute_microbatch_loss(model, scenes, device, flow_cfg, mix_cfg, standardizer,
                            image_l1_loss_weight, lpips_loss_weight, lpips_loss_func, enable_amp):
    """Return summed sample losses; keep temporary GPU state scoped to one microbatch."""
    is_fm_only = mix_cfg["schedule"] == "fm-only"
    render_enabled = not is_fm_only and (image_l1_loss_weight != 0 or (lpips_loss_weight != 0 and lpips_loss_func is not None))
    mode = flow_cfg["interpolant_type"]
    samples = []
    for active in scenes:
        source = None if mode == "one_sided" else gpu_utils.move_to_device(active["source_flow_gs"], device)
        target = gpu_utils.move_to_device(active["target_flow_gs"], device)
        t = torch.empty(1, device=device).uniform_(float(flow_cfg["flow_t_eps"]), 1.0 - float(flow_cfg["flow_t_eps"]))
        train_images, train_cameras = None, None
        if render_enabled:
            camera_indices = np.random.permutation(len(active["target_images"]))[:active["render_view_count"]]
            train_images = gpu_utils.move_to_device([active["target_images"][index] for index in camera_indices], device)
            train_cameras = dict(active["target_cameras"])
            train_cameras["camera_to_worlds"] = train_cameras["camera_to_worlds"][camera_indices]
            train_cameras = gpu_utils.move_to_device(train_cameras, device)
        source_means = None if mode == "one_sided" else active["source_gs"]["means"].to(device)
        target_means = active["target_gs"]["means"].to(device)
        noise = gpu_utils.move_to_device(active["fixed_train_noise"], device) if flow_cfg.get("fixed_train_noise", False) else None
        path = interpolants.construct_path(source, target, t, mode, source_means, target_means,
                                           float(flow_cfg["flow_noise_std"]), noise=noise)
        samples.append({**path, "source": source, "target": target, "t": t,
                        "source_means": source_means, "target_means": target_means, "scene_idx": active["scene_idx"],
                        "images": train_images, "cameras": train_cameras})

    summed_loss = None
    statistics = {}
    with torch.cuda.amp.autocast(enabled=enable_amp):
        predictions = model(
            batch_flow_gs=[sample["query"] for sample in samples],
            batch_scene_idx=[scene["scene_idx"] for scene in scenes],
            batch_reference_means=[sample["reference_means"] for sample in samples],
            t=torch.cat([sample["t"] for sample in samples]),
        )
        for sample, pred_vel in zip(samples, predictions):
            t, gamma, gamma_dot = sample["t"], sample["gamma"], sample["gamma_dot"]
            train_images, train_cameras = sample["images"], sample["cameras"]
            if flow_cfg["loss_type"] == "x1" or render_enabled:
                x1_pred_flow_gs = interpolants.rollout(
                    model, sample["source"], sample["scene_idx"], flow_cfg["loss_rollout_steps"], mode,
                    sample["source_means"], sample["target_means"], sample["noise"], flow_cfg["flow_t_eps"])
            if flow_cfg["loss_type"] == "velocity":
                fm_loss, attr_losses, weighted_attr_losses = interpolants.attribute_mse(pred_vel, sample["velocity"])
            else:
                fm_loss, attr_losses, weighted_attr_losses = interpolants.attribute_mse(x1_pred_flow_gs, sample["target"])
            if not render_enabled:
                render_l1 = fm_loss.new_zeros(())
                render_lpips = render_l1
                weighted_render_l1 = render_l1
                weighted_render_lpips = render_l1
                render_loss = render_l1
            else:
                render_gs = standardizer.decode(x1_pred_flow_gs)
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(render_gs, train_cameras)
                render_l1 = sum(
                    (pred_img - gt_img[..., :3]).abs().mean()
                    for pred_img, gt_img in zip(pred_imgs, train_images)
                ) / len(pred_imgs)
                render_lpips = 0.0
                if lpips_loss_func is not None:
                    render_lpips = sum(
                        lpips_loss_func(pred_img.unsqueeze(0), gt_img[..., :3].unsqueeze(0)).mean()
                        for pred_img, gt_img in zip(pred_imgs, train_images)
                    ) / len(pred_imgs)
                weighted_render_l1 = image_l1_loss_weight * render_l1
                weighted_render_lpips = (
                    lpips_loss_weight * render_lpips if lpips_loss_func is not None else 0.0
                )
                render_loss = weighted_render_l1 + weighted_render_lpips
            fm_mix_weight, render_mix_weight = flow.loss_mix_weights(t, mix_cfg["schedule"])
            total_loss = fm_mix_weight * fm_loss + render_mix_weight * render_loss

            summed_loss = total_loss if summed_loss is None else summed_loss + total_loss
            values = {
                "total": total_loss, "fm_loss": fm_loss, "render_loss": render_loss,
                "fm_weight": fm_mix_weight, "render_weight": render_mix_weight,
                "render_l1": render_l1, "weighted_render_l1": weighted_render_l1,
                "render_lpips": render_lpips, "weighted_render_lpips": weighted_render_lpips,
                "t": t, "gamma": gamma, "gamma_dot": gamma_dot,
            }
            values.update({f"{key}_loss": value for key, value in attr_losses.items()})
            values.update({f"{key}_weighted": value for key, value in weighted_attr_losses.items()})
            for key, value in values.items():
                scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                statistics[key] = statistics.get(key, 0.0) + scalar
    return summed_loss, statistics


@gin.configurable
def training(
    dataset,
    scene_indices,
    output_dir,
    logger,
    device,
    total_steps=gin.REQUIRED,
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
    flow_cfg = flow_matching()
    if not (0.0 < float(flow_cfg["flow_t_eps"]) < 0.5):
        raise ValueError(f"flow_t_eps must be in (0, 0.5), got {flow_cfg['flow_t_eps']}")
    mix_cfg = loss_mixing()
    # Load fixed target statistics before scene preparation or rendering.
    standardizer = GaussianStandardizer(flow_cfg["gs_statistics_path"], flow_cfg["normalization_variance_floor"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "normalization_statistics.json"), "w") as handle:
        json.dump(standardizer.report(), handle, indent=2)
        handle.write("\n")

    is_fm_only = mix_cfg["schedule"] == "fm-only"
    many = FLAGS.scene_mode == "many"
    prepared_scenes = []
    baseline_scene_metrics = {}
    # Cache fixed scene data on CPU; only the active scene needs GPU storage.
    for scene_idx in scene_indices:
        scene_name = dataset.folders[scene_idx]["scene_name"]
        scene_dir = os.path.join(output_dir, "scenes", scene_name) if many else output_dir
        try:
            scene = dataset.load_scene(scene_idx, fit_alignment=FLAGS.alignment)
            prepared = prepare_overfit_scene(dataset, scene, scene_dir, logger, device, standardizer)
        except Exception as error:
            raise RuntimeError(f"Failed to prepare scene {scene_name!r} (index {scene_idx})") from error
        view_count = len(prepared["target_images"])
        prepared["render_view_count"] = 0 if is_fm_only else min(dataset.image_per_scene or view_count, view_count)
        if flow_cfg["fixed_train_noise"]:
            prepared["fixed_train_noise"] = interpolants.seeded_noise_like(
                prepared["target_flow_gs"], int(flow_cfg["train_noise_seed"]) + int(prepared["scene_idx"])
            )
        if not prepared["target_images"]:
            raise ValueError(f"Scene {scene_name!r} has no target views")
        if not is_fm_only and prepared["render_view_count"] <= 0:
            raise ValueError(f"Scene {scene_name!r} has no render-loss views")
        train_dir = os.path.join(output_dir, "train", scene_name) if many else os.path.join(output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        prepared["train_dir"] = train_dir
        gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in prepared["target_images"]]
        gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(train_dir, "00000000_gt.png"), gt_grid)
        logger.info(
            f"Interpolants scene={scene_name} idx={scene_idx} target_views={view_count} "
            f"render_views_per_step={prepared['render_view_count']} alignment={FLAGS.alignment} "
            f"flow_config={flow_cfg} loss_mix_schedule={mix_cfg['schedule']} "
            f"image_l1_loss_weight={image_l1_loss_weight} lpips_loss_weight={lpips_loss_weight} "
            "loss=standardized_attribute_mse attribute_weights=unit"
        )
        baseline_scene_metrics[scene_name] = prepared.pop("baseline_metrics")
        prepared_scenes.append(gpu_utils.to_cpu(prepared) if many else prepared)
        del scene, prepared
    # Give every selected scene equal weight, regardless of its view count.
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
        "scenes": baseline_scene_metrics, "mean": baseline_mean,
    }
    baseline_path = os.path.join(output_dir, "baseline_render_metrics.json")
    with open(baseline_path, "w") as baseline_file:
        json.dump(baseline_report, baseline_file, indent=2)
        baseline_file.write("\n")
    logger.info(f"Baseline render metrics saved to {baseline_path}: mean={baseline_mean}")
    # exit()
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    predictor_class = PREDICTORS[FLAGS.predictor]
    model = predictor_class().to(device)
    logger.info(f"Predictor={FLAGS.predictor} class={predictor_class.__name__}")
    missing_outputs = sorted(set(SUPPORTED_GS_KEYS) - set(model.output_features))
    if missing_outputs:
        raise ValueError(
            f"{predictor_class.__name__}.output_features must include all Gaussian "
            f"attributes; missing {missing_outputs}"
        )
    if model.resume_ckpt is not None:
        raise ValueError(
            "GS_SR overfitting does not support resume_ckpt; start from scratch"
        )
    if model.quat_residual_mode != "add" or any(not isinstance(model.res_feature_activation[key], torch.nn.Identity) for key in SUPPORTED_GS_KEYS):
        raise ValueError("Standardized flow requires identity heads and additive quaternion updates")
    model.train()

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    with open(os.path.join(output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if not is_fm_only and lpips_loss_weight > 0 else None
    scene_order = []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    flow_t_eps = float(flow_cfg["flow_t_eps"])
    # Split the effective batch without dropping samples or overweighting a short microbatch.
    batch_size = FLAGS.batch_size
    grad_accum_steps = FLAGS.grad_accum_steps
    quotient, remainder = divmod(batch_size, grad_accum_steps)
    microbatch_sizes = [quotient + (index < remainder) for index in range(grad_accum_steps)]
    logger.info(f"batch_size={batch_size} grad_accum_steps={grad_accum_steps} microbatch_sizes={microbatch_sizes}")
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
                model, batch_scenes[offset:offset + microbatch_size], device, flow_cfg, mix_cfg,
                standardizer, image_l1_loss_weight, lpips_loss_weight,
                lpips_loss_func, enable_amp,
            )
            microbatch_loss = microbatch_loss / batch_size
            if enable_amp:
                scaler.scale(microbatch_loss).backward()
            else:
                microbatch_loss.backward()
            del microbatch_loss
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

        pbar.set_postfix({
            "loss": f"{batch_statistics['total']:.4f}",
            "fm": f"{batch_statistics['fm_loss']:.4f}",
            "render": f"{batch_statistics['render_loss']:.4f}",
            "t": f"{batch_statistics['t']:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            values = " ".join(f"{key}={value:.6f}" for key, value in batch_statistics.items())
            identities = [(scene["scene_name"], scene["scene_idx"]) for scene in batch_scenes]
            sampled_views = sum(scene["render_view_count"] for scene in batch_scenes)
            logger.info(
                f"step={step} scenes={identities} batch_size={batch_size} grad_accum_steps={grad_accum_steps} "
                f"mix_schedule={mix_cfg['schedule']} loss_type={flow_cfg['loss_type']} "
                f"sampled_views={sampled_views} t_eps={flow_t_eps:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {values}"
            )

        if step % log_image_interval == 0:
            active = batch_scenes[0]
            with torch.no_grad():
                source_flow_gs = gpu_utils.move_to_device(active["source_gs"], device)
                train_out_gs = interpolants.sample_flow_model(
                    model, source_flow_gs, active["scene_idx"], int(flow_cfg["flow_steps"]), standardizer,
                    mode=flow_cfg["interpolant_type"], target_means=active["target_gs"]["means"].to(device),
                    t_eps=flow_cfg["flow_t_eps"], noise_seed=flow_cfg["eval_noise_seed"])
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(train_out_gs, gpu_utils.move_to_device(active["target_cameras"], device))
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs[:9]]
                if len(pred_imgs_uint8) > 0:
                    pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(active["train_dir"], f"{step:08d}_pred.png"), pred_grid)
            model.train()
            del source_flow_gs, train_out_gs, pred_imgs, _

        is_final_step = step == total_steps - 1
        if step % eval_interval == 0 or is_final_step:
            eval_base_dir = (
                os.path.join(output_dir, FLAGS.eval_subdir)
                if is_final_step
                else os.path.join(output_dir, "eval", f"{step:08d}")
            )
            eval_label = "Final eval" if is_final_step else f"Eval step {step}"
            for eval_flow_steps in EVAL_FLOW_STEPS:
                eval_dir = os.path.join(eval_base_dir, f"flow_steps_{eval_flow_steps:02d}")
                scene_metrics = {}
                input_metrics = {}
                for evaluated in prepared_scenes:
                    scene_name = evaluated["scene_name"]
                    scene_eval_dir = os.path.join(eval_dir, "scenes", scene_name) if many else eval_dir
                    metrics, metrics_input = evaluate_single_scene(
                        model=model,
                        input_gs=gpu_utils.move_to_device(evaluated["source_gs"], device),
                        source_flow_gs=gpu_utils.move_to_device(evaluated["source_gs"], device),
                        standardizer=standardizer,
                        flow_cfg=flow_cfg,
                        gt_gs=gpu_utils.move_to_device(evaluated["target_gs"], device),
                        scene_idx=evaluated["scene_idx"],
                        scene_name=scene_name,
                        eval_images=evaluated["target_images"],
                        eval_cameras=evaluated["target_cameras"],
                        image_names=evaluated["target_image_names"],
                        output_dir=scene_eval_dir,
                        flow_steps=int(eval_flow_steps),
                        eval_chunk_size=evaluated["eval_chunk_size"],
                        compare_with_input=FLAGS.compare_with_input,
                        save_viewer=FLAGS.save_viewer,
                        output_gt=((step == 0 or is_final_step) and eval_flow_steps == EVAL_FLOW_STEPS[0]),
                    )
                    scene_metrics[scene_name] = metrics
                    input_metrics[scene_name] = metrics_input
                    message = f"{eval_label} scene={scene_name} flow_steps={eval_flow_steps}: {metrics}"
                    logger.info(message)
                    print(message)
                    if FLAGS.compare_with_input:
                        logger.info(f"{eval_label} input scene={scene_name} flow_steps={eval_flow_steps}: {metrics_input}")
                if many:
                    reports = [("metrics.json", scene_metrics)]
                    if FLAGS.compare_with_input:
                        reports.append(("metrics_input.json", input_metrics))
                    for filename, per_scene in reports:
                        mean = {
                            key: sum(values[key] for values in per_scene.values()) / len(per_scene)
                            for key in next(iter(per_scene.values()))
                        }
                        with open(os.path.join(eval_dir, filename), "w") as metric_file:
                            json.dump({"mean": mean, "scenes": per_scene}, metric_file, indent=2)
                        message = f"{eval_label} flow_steps={eval_flow_steps} {filename} scene_mean={mean}"
                        logger.info(message)
                        print(message)

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "model_last.pth"))


def main(argv):
    del argv
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    seed = set_seed()
    logger = ProcessSafeLogger(os.path.join(output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = SplatFactoSRDataset.from_gin_scope("test_dataset")
    # if not all((
    #     dataset.load_src_gs,
    #     dataset.load_tgt_gs,
    #     dataset.load_src_images,
    #     dataset.load_tgt_images,
    # )):
    #     raise ValueError("SR dev overfitting requires every source and target payload")
    if FLAGS.scene_mode == "one":
        scene_indices = [dataset.scene_index(FLAGS.scene_name)]
    else:
        # Deduplicate scene names while retaining their original dataset indices.
        unique_scenes = {}
        for index, entry in enumerate(dataset.folders):
            unique_scenes.setdefault(entry["scene_name"], index)
        if not 1 <= FLAGS.scene_count <= len(unique_scenes):
            raise ValueError(f"scene_count must be between 1 and {len(unique_scenes)}, got {FLAGS.scene_count}")
        scene_indices = np.random.default_rng(seed).choice(list(unique_scenes.values()), size=FLAGS.scene_count, replace=False).tolist()
    selection = {
        "scene_mode": FLAGS.scene_mode, "seed": seed,
        "scenes": [{"scene_name": dataset.folders[index]["scene_name"], "scene_idx": index} for index in scene_indices],
    }
    with open(os.path.join(output_dir, "selected_scenes.json"), "w") as selection_file:
        json.dump(selection, selection_file, indent=2)
    logger.info(f"Selected scenes: {selection}")
    training(
        dataset=dataset,
        scene_indices=scene_indices,
        output_dir=output_dir,
        logger=logger,
        device=device,
    )


if __name__ == "__main__":
    app.run(main)

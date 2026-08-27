import json
import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_SR_dev import SplatFactoSRDevDataset
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy Gin keys.
from sr import flow
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit_gsfm_noemd", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_enum("alignment", "emd", ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"], "Matching modes")
flags.DEFINE_enum("attribute_init", "aligned", ["aligned", "3dgs"], "attribute initialization for emd and random.",)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
EVAL_FLOW_STEPS = [1, 5, 20]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.


@gin.configurable
def set_seed(seed):
    seed_everything(seed)


@gin.configurable
def flow_matching(
    flow_steps=5,
    flow_noise_std=0.0,
    flow_t_eps=1e-4,
    loss_type="velocity",
    velocity_variance_floor=1e-8,
):
    if loss_type not in ("velocity", "x1"):
        raise ValueError(
            f"Unsupported flow_matching.loss_type={loss_type!r}; expected 'velocity' or 'x1'"
        )
    if float(velocity_variance_floor) <= 0.0:
        raise ValueError(
            "flow_matching.velocity_variance_floor must be positive, "
            f"got {velocity_variance_floor}"
        )
    return {
        "flow_steps": flow_steps,
        "flow_noise_std": flow_noise_std,
        "flow_t_eps": flow_t_eps,
        "loss_type": loss_type,
        "velocity_variance_floor": float(velocity_variance_floor),
    }

@gin.configurable
def loss_mixing(schedule="linear"):
    valid_schedules = ("linear", "free-range-gs", "fm-only")
    if schedule not in valid_schedules:
        raise ValueError(
            f"Unsupported loss_mixing.schedule={schedule!r}; "
            f"expected one of {valid_schedules}"
        )
    return {"schedule": schedule}


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
    default_weights.update(
        {key: float(value) for key, value in loss_weights.items()}
    )
    return {
        "loss_weights": default_weights,
        "quat_direct_mse": quat_direct_mse,
    }


def compute_all_feature_mse_loss(
    out_gs,
    target_gs,
    loss_weights,
    quat_direct_mse,
):
    missing_output = [
        key for key in SUPPORTED_GS_KEYS if key not in out_gs
    ]
    missing_target = [
        key for key in SUPPORTED_GS_KEYS if key not in target_gs
    ]
    if missing_output:
        raise ValueError(
            f"GSFlowPredictor must output every Gaussian attribute; "
            f"missing {missing_output}"
        )
    if missing_target:
        raise ValueError(
            f"All-attribute flow target is missing {missing_target}"
        )

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
        loss = loss_utils.feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=True,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=MEANS_LOSS_REDUCTION,
        )
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    if not total_loss.requires_grad:
        raise ValueError(
            "GSFlowPredictor outputs do not receive gradients for the "
            "all-attribute x1 loss"
        )
    return total_loss, losses, weighted_losses


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
        out_gs = flow.sample_flow_model(model, source_flow_gs, scene_idx, flow_steps)
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


@gin.configurable
def training(
    dataset,
    scene,
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
    mse_loss_cfg = feature_mse_loss()
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

    model = GSFlowPredictor().to(device)
    missing_outputs = sorted(set(SUPPORTED_GS_KEYS) - set(model.output_features))
    if missing_outputs:
        raise ValueError(
            f"GSFlowPredictor.output_features must include all Gaussian "
            f"attributes; missing {missing_outputs}"
        )
    if model.resume_ckpt is not None:
        raise ValueError(
            "GS_SR_dev overfitting does not support resume_ckpt; start from scratch"
        )
    model.train()

    loss_target_gs = matching_target_gs
    source_flow_gs = gs_utils.clone_gaussians(source_gs)
    target_flow_gs = gs_utils.clone_gaussians(loss_target_gs)
    raw_velocity_variances, velocity_variances = (
        flow.compute_matching_velocity_variances(
            source_flow_gs,
            target_flow_gs,
            flow_cfg["velocity_variance_floor"],
        )
    )
    variance_report = {
        key: {
            "variance": raw_velocity_variances[key].detach().cpu().tolist(),
            "effective_variance": velocity_variances[key].detach().cpu().tolist(),
        }
        for key in SUPPORTED_GS_KEYS
    }
    variance_message = (
        "GT matching velocity variances: "
        + json.dumps(variance_report, sort_keys=True)
    )
    print(variance_message)
    logger.info(variance_message)
    if flow_cfg["loss_type"] == "velocity":
        flow_loss_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
    else:
        flow_loss_weights = mse_loss_cfg["loss_weights"]
    batch_scene_idx = [scene["scene_idx"]]

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    with open(os.path.join(output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    is_fm_only = mix_cfg["schedule"] == "fm-only"
    render_view_count = 0
    if not is_fm_only:
        render_view_count = min(dataset.image_per_scene or len(target_images), len(target_images))
        if render_view_count <= 0:
            raise ValueError("No target views available for render loss")

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if not is_fm_only and lpips_loss_weight > 0 else None

    training_brief = (
        f"GSFM scene={scene['scene_name']} idx={scene['scene_idx']} "
        f"target_views={len(target_images)} "
        f"input_gaussians={input_resolution_entry['gs_params']['means'].shape[0]} "
        f"matching_target_gaussians={matching_target_gs['means'].shape[0]} "
        f"input_resolution={input_resolution} target_resolution={target_resolution} "
        f"alignment={FLAGS.alignment} "
        f"attribute_init={FLAGS.attribute_init if FLAGS.alignment in {'emd', 'random'} else 'inactive'} "
        f"alignment_info={alignment_info} "
        f"attribute_keys={','.join(SUPPORTED_GS_KEYS)} "
        f"flow_steps={flow_cfg['flow_steps']} "
        f"flow_noise_std={flow_cfg['flow_noise_std']} "
        f"flow_t_eps={flow_cfg['flow_t_eps']} "
        f"flow_loss_type={flow_cfg['loss_type']} "
        f"velocity_variance_floor={flow_cfg['velocity_variance_floor']} "
        f"loss_mix_schedule={mix_cfg['schedule']} "
        f"image_l1_loss_weight={image_l1_loss_weight} "
        f"lpips_loss_weight={lpips_loss_weight} "
        f"render_views_per_step={render_view_count} "
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} "
        f"loss_weights={flow_loss_weights}\n"
        f"dataset_class={type(dataset).__name__} "
        f"coordinate_frame={scene['coordinate_frame']} "
        f"coordinate_frame_version={scene['coordinate_frame_version']} "
        f"coordinate_resolution={scene['coordinate_resolution']}\n"
        f"fit_lr_to_hr_root={dataset.fit_lr_to_hr_root}\n"
        f"fit_hr_to_lr_root={dataset.fit_hr_to_lr_root}"
    )
    print(training_brief)
    logger.info(training_brief)

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    train_images_device = gpu_utils.move_to_device(target_images, device)
    train_cameras_device = gpu_utils.move_to_device(target_cameras, device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in train_images_device]
    if len(gt_imgs_uint8) > 0:
        gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    flow_t_eps = float(flow_cfg["flow_t_eps"])
    flow_noise_std = float(flow_cfg["flow_noise_std"])
    for step in pbar:
        t = torch.empty(1, device=device).uniform_(flow_t_eps, 1.0 - flow_t_eps)
        if not is_fm_only:
            camera_indices = np.random.permutation(len(target_images))[:render_view_count]
            train_images = [target_images[index] for index in camera_indices]
            train_cameras = dict(target_cameras)
            train_cameras["camera_to_worlds"] = target_cameras["camera_to_worlds"][camera_indices]
            train_images = gpu_utils.move_to_device(train_images, device)
            train_cameras = gpu_utils.move_to_device(train_cameras, device)
        query_flow_gs, flow_noise, gamma, gamma_dot = flow.sample_stochastic_interpolant(
            source_flow_gs, target_flow_gs, t, flow_noise_std
        )
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_vel = model(
                batch_flow_gs=[query_flow_gs],
                batch_scene_idx=batch_scene_idx,
                batch_reference_means=[source_flow_gs["means"]],
                t=t,
            )[0]
            x1_pred_flow_gs = flow.predict_x1_from_velocity(
                model, source_flow_gs, query_flow_gs, pred_vel, flow_noise, gamma, gamma_dot, t
            )
            if flow_cfg["loss_type"] == "velocity":
                fm_loss, attr_losses, weighted_attr_losses = (
                    flow.compute_variance_normalized_velocity_loss(
                        pred_vel=pred_vel,
                        source_flow_gs=source_flow_gs,
                        target_flow_gs=target_flow_gs,
                        flow_noise=flow_noise,
                        gamma_dot=gamma_dot,
                        velocity_variances=velocity_variances,
                        loss_weights=flow_loss_weights,
                    )
                )
            else:
                fm_loss, attr_losses, weighted_attr_losses = (
                    compute_all_feature_mse_loss(
                        out_gs=x1_pred_flow_gs,
                        target_gs=loss_target_gs,
                        loss_weights=flow_loss_weights,
                        quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                    )
                )
            if is_fm_only:
                render_l1 = fm_loss.new_zeros(())
                render_lpips = render_l1
                weighted_render_l1 = render_l1
                weighted_render_lpips = render_l1
                render_loss = render_l1
            else:
                render_gs = x1_pred_flow_gs
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

        optimizer_stepped = True
        if enable_amp:
            previous_scale = scaler.get_scale()
            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        else:
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if optimizer_stepped:
            scheduler.step()

        postfix = {
            "loss": f"{total_loss.item():.4f}",
            "fm": f"{fm_loss.item():.4f}",
            "render": f"{render_loss.item():.4f}",
            "t": f"{t.item():.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        if flow_noise_std > 0.0:
            postfix["gamma"] = f"{gamma.item():.3e}"
            postfix["gdot"] = f"{gamma_dot.item():.3e}"
        pbar.set_postfix(postfix)

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            attr_str = " ".join(
                [
                    f"{key}_loss={attr_losses[key].item():.6f} "
                    f"{key}_weighted={weighted_attr_losses[key].item():.6f}"
                    for key in attr_losses.keys()
                ]
            )
            logger.info(
                f"step={step} total={total_loss.item():.6f} "
                f"mix_schedule={mix_cfg['schedule']} "
                f"fm_loss={fm_loss.item():.6f} fm_weight={fm_mix_weight.item():.6f} "
                f"render_loss={render_loss.item():.6f} render_weight={render_mix_weight.item():.6f} "
                f"render_l1={render_l1.item():.6f} weighted_render_l1={weighted_render_l1.item():.6f} "
                f"render_lpips={render_lpips.item() if lpips_loss_func is not None else 0.0:.6f} "
                f"weighted_render_lpips={weighted_render_lpips.item() if lpips_loss_func is not None else 0.0:.6f} "
                f"sampled_views={render_view_count} loss_type={flow_cfg['loss_type']} "
                f"attribute_keys={','.join(SUPPORTED_GS_KEYS)} t={t.item():.6f} "
                f"t_eps={flow_t_eps:.6f} gamma={gamma.item():.8f} gamma_dot={gamma_dot.item():.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {attr_str}"
            )

        if step % log_image_interval == 0:
            with torch.no_grad():
                train_out_gs = flow.sample_flow_model(
                    model, source_flow_gs, scene["scene_idx"], int(flow_cfg["flow_steps"])
                )
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(train_out_gs, train_cameras_device)
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs[:9]]
                if len(pred_imgs_uint8) > 0:
                    pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

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
                metrics, metrics_input = evaluate_single_scene(
                    model=model,
                    input_gs=source_gs,
                    source_flow_gs=source_flow_gs,
                    gt_gs=matching_target_gs,
                    scene_idx=scene["scene_idx"],
                    scene_name=scene["scene_name"],
                    eval_images=target_images,
                    eval_cameras=target_cameras,
                    image_names=target_image_names,
                    output_dir=eval_dir,
                    flow_steps=int(eval_flow_steps),
                    eval_chunk_size=eval_chunk_size,
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    output_gt=((step == 0 or is_final_step) and eval_flow_steps == EVAL_FLOW_STEPS[0]),
                )
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"{eval_label} flow_steps={eval_flow_steps}: {metric_str}")
                print(f"{eval_label} flow_steps={eval_flow_steps}: {metric_str}")
                if FLAGS.compare_with_input:
                    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                    if is_final_step:
                        logger.info(f"{eval_label} input flow_steps={eval_flow_steps}: {metric_str}")
                    print(f"{eval_label} input flow_steps={eval_flow_steps}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "model_last.pth"))


def main(argv):
    del argv
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    set_seed()
    logger = ProcessSafeLogger(os.path.join(output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = SplatFactoSRDevDataset.from_gin_scope("test_dataset")
    if not dataset.load_gs or not dataset.load_images:
        raise ValueError("SR dev overfitting requires load_gs=True and load_images=True")
    scene_idx = dataset.scene_index(FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx, fit_alignment=FLAGS.alignment)
    training(
        dataset=dataset,
        scene=scene,
        output_dir=output_dir,
        logger=logger,
        device=device,
    )


if __name__ == "__main__":
    app.run(main)

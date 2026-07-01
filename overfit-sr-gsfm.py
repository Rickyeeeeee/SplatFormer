import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy gin keys used by GS_multi.
from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import (
    copy_gt_attributes,
    make_grid,
    unscale_means_origin,
    scale_means_origin,
)
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import (
    SUPPORTED_GS_KEYS,
    feature_mse_loss as compute_feature_mse_loss,
    fixed_attribute_keys as select_fixed_attribute_keys,
    parse_loss_features,
)
from utils.metrics import MetricComputer, write_densify_stage_render_metrics
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_dataset_utils import build_dataset, find_scene_index
from utils.sr_densify_utils import build_densified_input_gs, save_densify_stage_plys


flags.DEFINE_string("output_dir", "output_overfit_gsfm", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as densification source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as flow target")
flags.DEFINE_enum("alignment", "emd", ["emd", "nearest"], "Interpolated-to-target alignment method")
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_integer("flow_steps", None, "Euler sampling steps; overrides gin flow_matching.flow_steps")
flags.DEFINE_float("flow_noise_std", None, "Stochastic-interpolant noise scale multiplying sqrt(2t(1-t))")
flags.DEFINE_boolean(
    "post_activate_loss",
    False,
    "Use MSE-style feature loss transforms: sigmoid opacities and normalized quaternion loss.",
)
flags.DEFINE_string(
    "loss_features",
    "",
    "Comma-separated Gaussian attributes to include in point-wise flow loss. "
    "Defaults to GSFlowPredictor.output_features.",
)
flags.DEFINE_float(
    "means_origin_scale",
    1.0,
    "Training-only origin scale for target Gaussian means when computing means flow loss. "
    "Predicted means are divided by this value before render/export.",
)
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
FLOW_KEYS = SUPPORTED_GS_KEYS
EVAL_FLOW_STEPS = [1, 2, 5]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.
EPS = 1e-6


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
        "resume_from_step": resume_from_step,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


@gin.configurable
def flow_matching(
    flow_steps=5,
    flow_noise_std=0.0,
    flow_t_eps=1e-4,
):
    return {
        "flow_steps": flow_steps,
        "flow_noise_std": flow_noise_std,
        "flow_t_eps": flow_t_eps,
    }


@gin.configurable
def feature_mse_loss(loss_weights=None, quat_direct_mse=False):
    default_weights = {key: 1.0 for key in FLOW_KEYS}
    if loss_weights is None:
        loss_weights = {}
    unknown_keys = sorted(set(loss_weights) - set(FLOW_KEYS))
    if unknown_keys:
        raise ValueError(
            f"Unsupported feature_mse_loss.loss_weights keys: {unknown_keys}. "
            f"Supported keys are: {FLOW_KEYS}"
        )
    default_weights.update({key: float(value) for key, value in loss_weights.items()})
    return {"loss_weights": default_weights, "quat_direct_mse": quat_direct_mse}



def _apply_fixed_flow_attributes(flow_gs, fixed_flow_gs, attribute_keys):
    if fixed_flow_gs is None:
        return flow_gs
    for key in attribute_keys:
        if key in flow_gs and key in fixed_flow_gs:
            flow_gs[key] = fixed_flow_gs[key].to(device=flow_gs[key].device, dtype=flow_gs[key].dtype).clone()
    return flow_gs


def _clone_gs(gs):
    return {key: value.clone() for key, value in gs.items()}



def sample_stochastic_interpolant(source_flow_gs, target_flow_gs, t, noise_scale):
    alpha = t.view(1, 1)
    gamma_base = torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=EPS))
    gamma = (float(noise_scale) * gamma_base).view(1, 1)
    gamma_dot = (float(noise_scale) * (1.0 - 2.0 * t) / gamma_base).view(1, 1)

    query_flow_gs = {}
    flow_noise = {}
    for key, source_value in source_flow_gs.items():
        if key not in target_flow_gs:
            continue
        target_value = target_flow_gs[key]
        if float(noise_scale) > 0.0:
            z = torch.randn_like(source_value)
        else:
            z = torch.zeros_like(source_value)
        flow_noise[key] = z
        query_flow_gs[key] = (1.0 - alpha) * source_value + alpha * target_value + gamma * z

    return query_flow_gs, flow_noise, gamma, gamma_dot


def subtract_stochastic_velocity(pred_vel, flow_noise, gamma_dot):
    return {
        key: value - gamma_dot * flow_noise[key] if key in flow_noise else value
        for key, value in pred_vel.items()
    }


def _apply_model_flow_update(model, feature, value, update, step_scale=1.0):
    if model is not None and hasattr(model, "apply_feature_update"):
        return model.apply_feature_update(feature, value, update, step_scale)
    if torch.is_tensor(step_scale):
        step_scale = step_scale.to(device=update.device, dtype=update.dtype)
    else:
        step_scale = float(step_scale)
    return value + step_scale * update


def apply_flow_velocity(source_flow_gs, pred_vel, model=None):
    return {
        key: _apply_model_flow_update(model, key, source_value, pred_vel[key])
        if key in pred_vel
        else source_value.clone()
        for key, source_value in source_flow_gs.items()
    }


def predict_x1_from_velocity(model, query_flow_gs, pred_vel, flow_noise, gamma, gamma_dot, t):
    one_minus_t = (1.0 - t).view(1, 1)
    x1_pred = {}
    for key, query_value in query_flow_gs.items():
        clean_xt = query_value - gamma * flow_noise.get(key, torch.zeros_like(query_value))
        if key in pred_vel:
            clean_vel = pred_vel[key] - gamma_dot * flow_noise.get(key, torch.zeros_like(pred_vel[key]))
            x1_pred[key] = _apply_model_flow_update(model, key, clean_xt, clean_vel, one_minus_t)
        else:
            x1_pred[key] = clean_xt.clone()
    return x1_pred



def sample_flow_model(model, source_flow_gs, scene_idx, flow_steps, fixed_flow_gs=None, fixed_attribute_keys=None):
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    model.eval()
    if fixed_attribute_keys is None:
        fixed_attribute_keys = []
    state = _clone_gs(source_flow_gs)
    state = _apply_fixed_flow_attributes(state, fixed_flow_gs, fixed_attribute_keys)
    device = state["means"].device
    with torch.no_grad():
        for step in range(flow_steps):
            t_value = torch.full((1,), float(step) / float(flow_steps), device=device)
            pred_vel = model(batch_flow_gs=[state], batch_scene_idx=[scene_idx], t=t_value)[0]
            for key in state.keys():
                if key in pred_vel:
                    state[key] = _apply_model_flow_update(
                        model, key, state[key], pred_vel[key], 1.0 / float(flow_steps)
                    )
            state = _apply_fixed_flow_attributes(state, fixed_flow_gs, fixed_attribute_keys)
    return _clone_gs(state)



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
    fixed_raw_gs=None,
    fixed_flow_gs=None,
    fixed_attribute_keys=None,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
    means_origin_scale=1.0,
):
    if fixed_attribute_keys is None:
        fixed_attribute_keys = []
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    del save_residuals

    os.makedirs(output_dir, exist_ok=True)

    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input:
        compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = sample_flow_model(
            model,
            source_flow_gs,
            scene_idx,
            flow_steps,
            fixed_flow_gs=fixed_flow_gs,
            fixed_attribute_keys=fixed_attribute_keys,
        )
        if fixed_raw_gs is not None:
            out_gs = copy_gt_attributes(out_gs, fixed_raw_gs, fixed_attribute_keys)
        out_gs = unscale_means_origin(out_gs, means_origin_scale)
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


def _resolve_flow_cfg():
    cfg = flow_matching()
    if FLAGS.flow_steps is not None:
        cfg["flow_steps"] = FLAGS.flow_steps
    if FLAGS.flow_noise_std is not None:
        cfg["flow_noise_std"] = FLAGS.flow_noise_std
    if not (0.0 < float(cfg["flow_t_eps"]) < 0.5):
        raise ValueError(f"flow_t_eps must be in (0, 0.5), got {cfg['flow_t_eps']}")
    return cfg


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    flow_cfg = _resolve_flow_cfg()
    mse_loss_cfg = feature_mse_loss()
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = build_dataset()
    scene_idx = find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_dict = scene["factor_data"][FLAGS.input_factor]
    target_factor_dict = scene["factor_data"][FLAGS.target_factor]

    target_images, target_image_names, target_cameras = dataset.load_factor_views(target_factor_dict)

    model = GSFlowPredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    target_gs_raw = gpu_utils.move_to_device(target_factor_dict["gs_params"], device)
    loss_features = parse_loss_features(FLAGS.loss_features, model, target_gs_raw, no_features_message="No Gaussian attributes selected for flow loss")
    fixed_attribute_keys = select_fixed_attribute_keys(loss_features, target_gs_raw)
    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = requested_means_origin_scale if "means" in loss_features else 1.0

    input_gs_raw, densify_stage_gs = build_densified_input_gs(
        input_factor_dict=input_factor_dict,
        target_factor_dict=target_factor_dict,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        return_stages=True,
    )
    densify_stage_gs["03_input_high_res_gs.ply"] = input_gs_raw
    save_densify_stage_plys(
        output_dir=FLAGS.output_dir,
        low_res_gs=densify_stage_gs["00_low_res_gs.ply"],
        interpolated_gs=densify_stage_gs["01_interpolated_high_res_gs.ply"],
        gt_high_res_gs=densify_stage_gs["02_gt_high_res_gs.ply"],
        input_high_res_gs=densify_stage_gs["03_input_high_res_gs.ply"],
    )
    loss_target_gs = scale_means_origin(target_gs_raw, means_origin_scale)
    source_flow_gs = _clone_gs(input_gs_raw)
    target_flow_gs = _clone_gs(loss_target_gs)
    batch_scene_idx = [scene["idx"]]

    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(target_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(target_images)

    densify_metrics = write_densify_stage_render_metrics(
        output_dir=FLAGS.output_dir,
        stage_gs=densify_stage_gs,
        images=target_images,
        cameras=target_cameras,
        chunk_size=eval_chunk_size,
        device=device,
    )
    for ply_name, metrics in densify_metrics.items():
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Densify init {ply_name}: {metric_str}")

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

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    print(
        f"GSFM scene={scene['scene_name']} idx={scene['idx']} "
        f"target_views={len(target_images)} "
        f"input_gaussians={input_factor_dict['gs_params']['means'].shape[0]} "
        f"densified_gaussians={input_gs_raw['means'].shape[0]} "
        f"target_gaussians={target_gs_raw['means'].shape[0]} "
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} "
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} "
        f"loss_features={','.join(loss_features)} "
        f"eval_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'} "
        f"flow_steps={flow_cfg['flow_steps']} "
        f"flow_noise_std={flow_cfg['flow_noise_std']} "
        f"flow_t_eps={flow_cfg['flow_t_eps']} "
        f"post_activate_loss={FLAGS.post_activate_loss} "
        f"means_origin_scale={requested_means_origin_scale} "
        f"effective_means_origin_scale={means_origin_scale} "
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} "
        f"loss_weights={mse_loss_cfg['loss_weights']}"
    )
    logger.info(
        f"means_origin_scale={requested_means_origin_scale} "
        f"effective_means_origin_scale={means_origin_scale}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    train_images_device = gpu_utils.move_to_device(target_images, device)
    train_cameras_device = gpu_utils.move_to_device(target_cameras, device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in train_images_device]
    if len(gt_imgs_uint8) > 0:
        gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    flow_t_eps = float(flow_cfg["flow_t_eps"])
    flow_noise_std = float(flow_cfg["flow_noise_std"])
    for step in pbar:
        t = torch.empty(1, device=device).uniform_(flow_t_eps, 1.0 - flow_t_eps)
        query_flow_gs, flow_noise, gamma, gamma_dot = sample_stochastic_interpolant(
            source_flow_gs, target_flow_gs, t, flow_noise_std
        )
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_vel = model(batch_flow_gs=[query_flow_gs], batch_scene_idx=batch_scene_idx, t=t)[0]
            x1_pred_flow_gs = predict_x1_from_velocity(
                model, query_flow_gs, pred_vel, flow_noise, gamma, gamma_dot, t
            )
            x1_pred_raw_gs = x1_pred_flow_gs
            total_loss, attr_losses, weighted_attr_losses = compute_feature_mse_loss(
                x1_pred_raw_gs,
                loss_target_gs,
                loss_features,
                post_activate_loss=FLAGS.post_activate_loss,
                loss_weights=mse_loss_cfg["loss_weights"],
                quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                means_loss_reduction=MEANS_LOSS_REDUCTION,
                output_label="predicted x1",
                total_loss_error="No x1_pred MSE losses were computed",
                grad_error=(
                    "Selected loss features do not receive gradients. "
                    "Make sure GSFlowPredictor.output_features includes at least one selected loss feature."
                ),
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

        postfix = {
            "loss": f"{total_loss.item():.4f}",
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
                    f"{key}_mse={attr_losses[key].item():.6f} "
                    f"{key}_weighted={weighted_attr_losses[key].item():.6f}"
                    for key in attr_losses.keys()
                ]
            )
            logger.info(
                f"step={step} total={total_loss.item():.6f} loss_type=x1_pred_mse "
                f"loss_features={','.join(loss_features)} t={t.item():.6f} "
                f"t_eps={flow_t_eps:.6f} gamma={gamma.item():.8f} gamma_dot={gamma_dot.item():.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {attr_str}"
            )

        if step % log_image_interval == 0:
            with torch.no_grad():
                train_out_gs = sample_flow_model(
                    model,
                    source_flow_gs,
                    scene["idx"],
                    int(flow_cfg["flow_steps"]),
                    fixed_flow_gs=target_flow_gs,
                    fixed_attribute_keys=fixed_attribute_keys,
                )
                train_out_gs = copy_gt_attributes(train_out_gs, loss_target_gs, fixed_attribute_keys)
                train_out_gs = unscale_means_origin(train_out_gs, means_origin_scale)
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(train_out_gs, train_cameras_device)
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs[:9]]
                if len(pred_imgs_uint8) > 0:
                    pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_base_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            for eval_flow_steps in EVAL_FLOW_STEPS:
                eval_dir = os.path.join(eval_base_dir, f"flow_steps_{eval_flow_steps:02d}")
                metrics, metrics_input = evaluate_single_scene(
                    model=model,
                    input_gs=input_gs_raw,
                    source_flow_gs=source_flow_gs,
                    gt_gs=target_gs_raw,
                    scene_idx=scene["idx"],
                    scene_name=scene["scene_name"],
                    eval_images=target_images,
                    eval_cameras=target_cameras,
                    image_names=target_image_names,
                    output_dir=eval_dir,
                    flow_steps=int(eval_flow_steps),
                    fixed_raw_gs=loss_target_gs,
                    fixed_flow_gs=target_flow_gs,
                    fixed_attribute_keys=fixed_attribute_keys,
                    eval_chunk_size=eval_chunk_size,
                    means_origin_scale=means_origin_scale,
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    save_residuals=FLAGS.save_residuals,
                    output_gt=(step == 0 and eval_flow_steps == EVAL_FLOW_STEPS[0]),
                )
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Eval step {step} flow_steps={eval_flow_steps}: {metric_str}")
                if FLAGS.compare_with_input:
                    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                    logger.info(f"Eval input step {step} flow_steps={eval_flow_steps}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))

    final_eval_base_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    for eval_flow_steps in EVAL_FLOW_STEPS:
        final_eval_dir = os.path.join(final_eval_base_dir, f"flow_steps_{eval_flow_steps:02d}")
        metrics, metrics_input = evaluate_single_scene(
            model=model,
            input_gs=input_gs_raw,
            source_flow_gs=source_flow_gs,
            gt_gs=target_gs_raw,
            scene_idx=scene["idx"],
            scene_name=scene["scene_name"],
            eval_images=target_images,
            eval_cameras=target_cameras,
            image_names=target_image_names,
            output_dir=final_eval_dir,
            flow_steps=int(eval_flow_steps),
            fixed_raw_gs=loss_target_gs,
            fixed_flow_gs=target_flow_gs,
            fixed_attribute_keys=fixed_attribute_keys,
            eval_chunk_size=eval_chunk_size,
            means_origin_scale=means_origin_scale,
            compare_with_input=FLAGS.compare_with_input,
            save_viewer=FLAGS.save_viewer,
            save_residuals=FLAGS.save_residuals,
            output_gt=(eval_flow_steps == EVAL_FLOW_STEPS[0]),
        )

        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Final eval flow_steps={eval_flow_steps}: {metric_str}")
        if FLAGS.compare_with_input:
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
            logger.info(f"Final eval input flow_steps={eval_flow_steps}: {metric_str}")


if __name__ == "__main__":
    app.run(main)

import json
import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy gin keys used by GS_multi.
from utils import gpu_utils, gs_utils, loss_utils
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
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_dataset_utils import build_dataset, find_scene_index
from utils.sr_matching_utils import (
    build_matching_source,
    get_or_fit_matching_target,
    matching_fit,
    save_matching_artifacts,
)


flags.DEFINE_string("output_dir", "output_overfit_gsfm_noemd", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as flow source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as flow target")
flags.DEFINE_string(
    "pre_matching_root",
    "/project2/ricky/splatformer-data-to-4x",
    "Root directory for persistent scene-level pre-matching targets.",
)
flags.DEFINE_boolean(
    "force_pre_matching",
    False,
    "Ignore a compatible cached pre-matching target and fit a replacement.",
)
flags.DEFINE_integer("flow_steps", None, "Euler sampling steps; overrides gin flow_matching.flow_steps")
flags.DEFINE_float("flow_noise_std", None, "Stochastic-interpolant noise scale multiplying sqrt(2t(1-t))")
flags.DEFINE_enum(
    "flow_loss_type",
    None,
    ["velocity", "x1"],
    "Flow objective; overrides gin flow_matching.loss_type.",
)
flags.DEFINE_string(
    "loss_features",
    "",
    "Comma-separated Gaussian attributes to include in point-wise flow loss. "
    "Defaults to GSFlowPredictor.output_features.",
)
flags.DEFINE_boolean(
    "model_features_from_loss",
    False,
    "Override GSFlowPredictor.input_features and GSFlowPredictor.output_features "
    "to exactly match the selected loss_features.",
)
flags.DEFINE_float(
    "means_origin_scale",
    1.05,
    "Training-only origin scale for target Gaussian means when computing means flow loss. "
    "Predicted means are divided by this value before render/export.",
)
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
FLOW_KEYS = SUPPORTED_GS_KEYS
EVAL_FLOW_STEPS = [1, 5, 20]
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


def get_loss_mix_weights(t, schedule):
    """Return FM and render weights for the sampled GSFM flow time."""
    if schedule == "linear":
        return 1.0 - t, t
    if schedule == "free-range-gs":
        render_weight = 50.0 * torch.clamp(t / 0.9, max=1.0).pow(5)
        return torch.ones_like(t), render_weight
    if schedule == "fm-only":
        return torch.ones_like(t), torch.zeros_like(t)
    raise ValueError(
        f"Unsupported loss_mixing.schedule={schedule!r}; "
        "expected one of ('linear', 'free-range-gs', 'fm-only')"
    )



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


class _FeatureSpec:
    def __init__(self, output_features):
        self.output_features = output_features


def resolve_loss_features(raw_loss_features, target_gs):
    if raw_loss_features is None or raw_loss_features.strip() == "":
        configured_output_features = gin.query_parameter("GSFlowPredictor.output_features")
        return parse_loss_features(
            raw_loss_features,
            _FeatureSpec(configured_output_features),
            target_gs,
            no_features_message="No Gaussian attributes selected for flow loss",
        )
    return parse_loss_features(
        raw_loss_features,
        _FeatureSpec([]),
        target_gs,
        no_features_message="No Gaussian attributes selected for flow loss",
    )



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


def compute_matching_velocity_variances(
    source_flow_gs, target_flow_gs, loss_features, variance_floor
):
    raw_variances = {}
    effective_variances = {}
    for key in loss_features:
        if key not in source_flow_gs or key not in target_flow_gs:
            raise ValueError(f"Cannot compute velocity variance for missing feature {key!r}")
        if source_flow_gs[key].shape != target_flow_gs[key].shape:
            raise ValueError(
                f"Velocity variance shape mismatch for {key!r}: "
                f"source {tuple(source_flow_gs[key].shape)} vs "
                f"target {tuple(target_flow_gs[key].shape)}"
            )
        gt_velocity = (target_flow_gs[key] - source_flow_gs[key]).detach().float()
        variance = gt_velocity.var(dim=0, unbiased=False)
        raw_variances[key] = variance
        effective_variances[key] = variance.clamp_min(float(variance_floor))
    return raw_variances, effective_variances


def format_velocity_variances(raw_variances, effective_variances):
    report = {
        key: {
            "variance": raw_variances[key].detach().cpu().tolist(),
            "effective_variance": effective_variances[key].detach().cpu().tolist(),
        }
        for key in raw_variances
    }
    return json.dumps(report, sort_keys=True)


def compute_variance_normalized_velocity_loss(
    pred_vel,
    source_flow_gs,
    target_flow_gs,
    flow_noise,
    gamma_dot,
    loss_features,
    velocity_variances,
    loss_weights=None,
):
    if loss_weights is None:
        loss_weights = {}
    losses = {}
    weighted_losses = {}
    total_loss = None
    for key in loss_features:
        if key not in pred_vel:
            raise ValueError(f"Selected velocity feature {key!r} missing from model output")
        pred = pred_vel[key].float()
        target = (target_flow_gs[key] - source_flow_gs[key]).to(
            device=pred.device, dtype=pred.dtype
        )
        if key in flow_noise:
            noise = flow_noise[key].to(device=pred.device, dtype=pred.dtype)
            target = target + gamma_dot.to(device=pred.device, dtype=pred.dtype) * noise
        variance = velocity_variances[key].to(device=pred.device, dtype=pred.dtype)
        loss = ((pred - target).square() / variance).mean()
        weighted_loss = float(loss_weights.get(key, 1.0)) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

    if total_loss is None:
        raise ValueError("No variance-normalized velocity MSE losses were computed")
    if not total_loss.requires_grad:
        raise ValueError(
            "Selected velocity loss features do not receive gradients. "
            "Make sure GSFlowPredictor.output_features includes a selected feature."
        )
    return total_loss, losses, weighted_losses


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


def predict_x1_from_velocity(
    model,
    source_flow_gs,
    query_flow_gs,
    pred_vel,
    flow_noise,
    gamma,
    gamma_dot,
    t,
    source_anchored=True,
):
    one_minus_t = (1.0 - t).view(1, 1)
    x1_pred = {}
    for key, query_value in query_flow_gs.items():
        clean_xt = query_value - gamma * flow_noise.get(key, torch.zeros_like(query_value))
        base_value = source_flow_gs[key] if source_anchored else clean_xt
        step_scale = 1.0 if source_anchored else one_minus_t
        if key in pred_vel:
            clean_vel = pred_vel[key] - gamma_dot * flow_noise.get(key, torch.zeros_like(pred_vel[key]))
            x1_pred[key] = _apply_model_flow_update(model, key, base_value, clean_vel, step_scale)
        else:
            x1_pred[key] = base_value.clone()
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
    if FLAGS.flow_loss_type is not None:
        cfg["loss_type"] = FLAGS.flow_loss_type
    if not (0.0 < float(cfg["flow_t_eps"]) < 0.5):
        raise ValueError(f"flow_t_eps must be in (0, 0.5), got {cfg['flow_t_eps']}")
    return cfg


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    matching_cfg = matching_fit()
    flow_cfg = _resolve_flow_cfg()
    mix_cfg = loss_mixing()
    mse_loss_cfg = feature_mse_loss()
    set_seed()

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = build_dataset()
    scene_idx = find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_dict = scene["factor_data"][FLAGS.input_factor]
    target_factor_dict = scene["factor_data"][FLAGS.target_factor]

    target_images, target_image_names, target_cameras = dataset.load_factor_views(target_factor_dict)

    source_gs = build_matching_source(input_factor_dict, target_factor_dict, device)
    matching_target_gs, matching_cache = get_or_fit_matching_target(
        source_gs=source_gs,
        target_images=target_images,
        target_cameras=target_cameras,
        pre_matching_root=FLAGS.pre_matching_root,
        scene_name=scene["scene_name"],
        input_factor=FLAGS.input_factor,
        target_factor=FLAGS.target_factor,
        logger=logger,
        config=matching_cfg,
        force_pre_matching=FLAGS.force_pre_matching,
    )
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(target_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(target_images)
    matching_metrics = save_matching_artifacts(
        FLAGS.output_dir, source_gs, matching_target_gs, target_images, target_cameras, eval_chunk_size, device
    )
    for ply_name, metrics in matching_metrics.items():
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        logger.info(f"Matching init {ply_name}: {metric_str}")

    loss_features = resolve_loss_features(FLAGS.loss_features, matching_target_gs)
    if FLAGS.model_features_from_loss:
        with gin.unlock_config():
            gin.bind_parameter("GSFlowPredictor.input_features", loss_features)
            gin.bind_parameter("GSFlowPredictor.output_features", loss_features)
    model = GSFlowPredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    fixed_attribute_keys = select_fixed_attribute_keys(loss_features, matching_target_gs)
    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = requested_means_origin_scale if "means" in loss_features else 1.0
    loss_target_gs = scale_means_origin(matching_target_gs, means_origin_scale)
    source_flow_gs = _clone_gs(source_gs)
    target_flow_gs = _clone_gs(loss_target_gs)
    raw_velocity_variances, velocity_variances = compute_matching_velocity_variances(
        source_flow_gs,
        target_flow_gs,
        loss_features,
        flow_cfg["velocity_variance_floor"],
    )
    variance_message = (
        "GT matching velocity variances: "
        + format_velocity_variances(raw_velocity_variances, velocity_variances)
    )
    print(variance_message)
    logger.info(variance_message)
    if flow_cfg["loss_type"] == "velocity":
        flow_loss_weights = {key: 1.0 for key in loss_features}
    else:
        flow_loss_weights = mse_loss_cfg["loss_weights"]
    batch_scene_idx = [scene["idx"]]

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

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
    is_fm_only = mix_cfg["schedule"] == "fm-only"
    render_view_count = 0
    if not is_fm_only:
        render_view_count = min(dataset.image_per_scene or len(target_images), len(target_images))
        if render_view_count <= 0:
            raise ValueError("No target views available for render loss")

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if not is_fm_only and lpips_loss_weight > 0 else None

    print(
        f"GSFM scene={scene['scene_name']} idx={scene['idx']} "
        f"target_views={len(target_images)} "
        f"input_gaussians={input_factor_dict['gs_params']['means'].shape[0]} "
        f"matching_target_gaussians={matching_target_gs['means'].shape[0]} "
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} "
        f"matching_cache={matching_cache['status']} matching_cache_path={matching_cache['checkpoint_path']} "
        f"matching_steps={matching_cfg['total_steps']} matching_images_per_step={matching_cfg['image_per_step']} "
        f"loss_features={','.join(loss_features)} "
        f"model_features_from_loss={FLAGS.model_features_from_loss} "
        f"model_input_features={','.join(model.input_features)} "
        f"model_output_features={','.join(model.output_features)} "
        f"eval_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'} "
        f"flow_steps={flow_cfg['flow_steps']} "
        f"flow_noise_std={flow_cfg['flow_noise_std']} "
        f"flow_t_eps={flow_cfg['flow_t_eps']} "
        f"flow_loss_type={flow_cfg['loss_type']} "
        f"velocity_variance_floor={flow_cfg['velocity_variance_floor']} "
        f"loss_mix_schedule={mix_cfg['schedule']} "
        f"image_l1_loss_weight={image_l1_loss_weight} "
        f"lpips_loss_weight={lpips_loss_weight} "
        f"render_views_per_step={render_view_count} "
        f"means_origin_scale={requested_means_origin_scale} "
        f"effective_means_origin_scale={means_origin_scale} "
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} "
        f"loss_weights={flow_loss_weights}"
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
        if not is_fm_only:
            camera_indices = np.random.permutation(len(target_images))[:render_view_count]
            train_images, _, train_cameras = dataset.load_factor_views(target_factor_dict, cam_ids=camera_indices)
            train_images = gpu_utils.move_to_device(train_images, device)
            train_cameras = gpu_utils.move_to_device(train_cameras, device)
        query_flow_gs, flow_noise, gamma, gamma_dot = sample_stochastic_interpolant(
            source_flow_gs, target_flow_gs, t, flow_noise_std
        )
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_vel = model(batch_flow_gs=[query_flow_gs], batch_scene_idx=batch_scene_idx, t=t)[0]
            x1_pred_flow_gs = predict_x1_from_velocity(
                model, source_flow_gs, query_flow_gs, pred_vel, flow_noise, gamma, gamma_dot, t
            )
            if flow_cfg["loss_type"] == "velocity":
                fm_loss, attr_losses, weighted_attr_losses = (
                    compute_variance_normalized_velocity_loss(
                        pred_vel,
                        source_flow_gs,
                        target_flow_gs,
                        flow_noise,
                        gamma_dot,
                        loss_features,
                        velocity_variances,
                        loss_weights=flow_loss_weights,
                    )
                )
            else:
                fm_loss, attr_losses, weighted_attr_losses = compute_feature_mse_loss(
                    x1_pred_flow_gs,
                    loss_target_gs,
                    loss_features,
                    loss_weights=flow_loss_weights,
                    post_activate_loss=True,
                    quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                    means_loss_reduction=MEANS_LOSS_REDUCTION,
                    output_label="predicted x1",
                    total_loss_error="No x1_pred MSE losses were computed",
                    grad_error=(
                        "Selected loss features do not receive gradients. "
                        "Make sure GSFlowPredictor.output_features includes at least one selected loss feature."
                    ),
                )
            if is_fm_only:
                render_l1 = fm_loss.new_zeros(())
                render_lpips = render_l1
                weighted_render_l1 = render_l1
                weighted_render_lpips = render_l1
                render_loss = render_l1
            else:
                render_gs = unscale_means_origin(x1_pred_flow_gs, means_origin_scale)
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
            fm_mix_weight, render_mix_weight = get_loss_mix_weights(t, mix_cfg["schedule"])
            total_loss = fm_mix_weight * fm_loss + render_mix_weight * render_loss

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
                    input_gs=source_gs,
                    source_flow_gs=source_flow_gs,
                    gt_gs=matching_target_gs,
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
                print(f"Eval step {step} flow_steps={eval_flow_steps}: {metric_str}")
                if FLAGS.compare_with_input:
                    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                    print(f"Eval input step {step} flow_steps={eval_flow_steps}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))

    final_eval_base_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    for eval_flow_steps in EVAL_FLOW_STEPS:
        final_eval_dir = os.path.join(final_eval_base_dir, f"flow_steps_{eval_flow_steps:02d}")
        metrics, metrics_input = evaluate_single_scene(
            model=model,
            input_gs=source_gs,
            source_flow_gs=source_flow_gs,
            gt_gs=matching_target_gs,
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
        print(f"Final eval flow_steps={eval_flow_steps}: {metric_str}")
        if FLAGS.compare_with_input:
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
            logger.info(f"Final eval input flow_steps={eval_flow_steps}: {metric_str}")
            print(f"Final eval input flow_steps={eval_flow_steps}: {metric_str}")


if __name__ == "__main__":
    app.run(main)

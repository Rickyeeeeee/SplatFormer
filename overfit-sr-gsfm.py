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
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy gin keys used by GS_multi.
from utils import gpu_utils, gs_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_densify_utils import build_densified_input_gs


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
flags.DEFINE_string("flow_space", None, "Flow space. GSFM only supports raw, matching overfit-sr-mse.py.")
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
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
FLOW_SPACES = {"raw"}
FLOW_KEYS = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]
MEANS_LOSS_REDUCTION = "sum"  # Set to "mean" for per-coordinate averaging.
EPS = 1e-6


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
    flow_space="raw",
    flow_noise_std=0.0,
    flow_t_eps=1e-4,
):
    return {
        "flow_steps": flow_steps,
        "flow_space": flow_space,
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


def make_grid(imgs, nrow=3, ncols=3):
    img_h, img_w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        grid = np.zeros((img_h * nrow, img_w * ncols, 3), dtype=np.uint8)
    else:
        grid = np.zeros((img_h * nrow, img_w * ncols), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncols):
            if i * ncols + j >= len(imgs):
                break
            grid[i * img_h : (i + 1) * img_h, j * img_w : (j + 1) * img_w] = imgs[i * ncols + j]
    return grid


def _to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    return data


def _sanitize_for_filename(value):
    return str(value).replace("/", "_").replace("\\", "_")


def _unique_preserve_order(values):
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _parse_loss_features(raw_loss_features, model, target_gs):
    if raw_loss_features is None or raw_loss_features.strip() == "":
        loss_features = list(getattr(model, "output_features", []))
    else:
        loss_features = [feature.strip() for feature in raw_loss_features.split(",") if feature.strip()]

    loss_features = _unique_preserve_order(loss_features)
    if len(loss_features) == 0:
        raise ValueError("No Gaussian attributes selected for flow loss")

    unsupported = [feature for feature in loss_features if feature not in FLOW_KEYS]
    if unsupported:
        raise ValueError(f"Unsupported loss feature(s): {unsupported}. Supported features are: {FLOW_KEYS}")

    missing = [feature for feature in loss_features if feature not in target_gs]
    if missing:
        raise ValueError(f"Selected loss feature(s) missing from target GS: {missing}")

    return loss_features


def _fixed_attribute_keys(loss_features, target_gs):
    loss_set = set(loss_features)
    return [key for key in FLOW_KEYS if key in target_gs and key not in loss_set]


def _copy_gt_attributes(gs, target_gs, attribute_keys):
    for key in attribute_keys:
        if key in gs and key in target_gs:
            gs[key] = target_gs[key].to(device=gs[key].device, dtype=gs[key].dtype).clone()
    return gs


def _apply_fixed_flow_attributes(flow_gs, fixed_flow_gs, attribute_keys):
    if fixed_flow_gs is None:
        return flow_gs
    for key in attribute_keys:
        if key in flow_gs and key in fixed_flow_gs:
            flow_gs[key] = fixed_flow_gs[key].to(device=flow_gs[key].device, dtype=flow_gs[key].dtype).clone()
    return flow_gs


def _clone_gs(gs):
    return {key: value.clone() for key, value in gs.items()}


def _require_raw_flow_space(flow_space):
    if flow_space != "raw":
        raise ValueError(
            f"GSFM only supports flow_space='raw' to match overfit-sr-mse.py residual prediction; "
            f"got {flow_space!r}"
        )


def raw_to_flow_gs(gs, flow_space):
    _require_raw_flow_space(flow_space)
    return {key: value.clone() for key, value in gs.items()}


def flow_to_raw_gs(flow, flow_space):
    _require_raw_flow_space(flow_space)
    return {key: value.clone() for key, value in flow.items()}


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


def _feature_loss_value(key, pred, target, post_activate_loss=False, quat_direct_mse=False):
    if key == "means":
        squared_error = (pred - target).square()
        if MEANS_LOSS_REDUCTION == "mean":
            return squared_error.mean()
        if MEANS_LOSS_REDUCTION == "sum":
            return squared_error.sum()
        raise ValueError(
            f"Unsupported MEANS_LOSS_REDUCTION={MEANS_LOSS_REDUCTION}; expected 'mean' or 'sum'"
        )

    if not post_activate_loss:
        return F.mse_loss(pred, target)

    if key == "opacities":
        return F.mse_loss(torch.sigmoid(pred), torch.sigmoid(target))

    if key == "quats":
        if quat_direct_mse:
            return F.mse_loss(pred, target)
        pred_quat = F.normalize(pred, dim=-1)
        target_quat = F.normalize(target, dim=-1)
        cosine_sq = (pred_quat * target_quat).sum(dim=-1).square().clamp(max=1.0)
        return (1.0 - cosine_sq).mean()

    return F.mse_loss(pred, target)


def _feature_mse_loss(
    out_gs,
    target_gs,
    loss_features,
    post_activate_loss=False,
    loss_weights=None,
    quat_direct_mse=False,
):
    losses = {}
    weighted_losses = {}
    total = None
    if loss_weights is None:
        loss_weights = {key: 1.0 for key in loss_features}

    for key in loss_features:
        if key not in out_gs:
            raise ValueError(f"Selected loss feature '{key}' missing from predicted x1")
        if key not in target_gs:
            raise ValueError(f"Selected loss feature '{key}' missing from target GS")
        if out_gs[key].shape != target_gs[key].shape:
            raise ValueError(
                f"Shape mismatch for loss feature '{key}': "
                f"output {tuple(out_gs[key].shape)} vs target {tuple(target_gs[key].shape)}"
            )
        pred = out_gs[key]
        target = target_gs[key].to(device=pred.device, dtype=pred.dtype)
        loss = _feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=post_activate_loss,
            quat_direct_mse=quat_direct_mse,
        )
        weighted = float(loss_weights.get(key, 1.0)) * loss
        losses[key] = loss
        weighted_losses[key] = weighted
        total = weighted if total is None else total + weighted

    if total is None:
        raise ValueError("No x1_pred MSE losses were computed")
    if not total.requires_grad:
        raise ValueError(
            "Selected loss features do not receive gradients. "
            "Make sure GSFlowPredictor.output_features includes at least one selected loss feature."
        )
    return total, losses, weighted_losses


def sample_flow_model(model, source_flow_gs, scene_idx, flow_steps, flow_space, fixed_flow_gs=None, fixed_attribute_keys=None):
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
    return flow_to_raw_gs(state, flow_space)


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
    if split in ["train", "test"]:
        cam_ids = np.arange(total_num)
    else:
        raise ValueError(f"Unsupported split: {split}")

    images = [dataset.read_image(imgs_path[i], background=background) for i in cam_ids]
    images_name = [imgs_name[i] for i in cam_ids]
    camera_to_worlds = meta["camera_to_worlds"][cam_ids]

    cameras = {
        "camera_to_worlds": torch.as_tensor(camera_to_worlds).float(),
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
        "images_name": images_name,
        "cameras": cameras,
        "scene_idx": scene_idx,
        "scene_name": scene_name,
    }


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
    flow_space,
    fixed_raw_gs=None,
    fixed_flow_gs=None,
    fixed_attribute_keys=None,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
):
    if fixed_attribute_keys is None:
        fixed_attribute_keys = []
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
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
        out_gs = sample_flow_model(
            model,
            source_flow_gs,
            scene_idx,
            flow_steps,
            flow_space,
            fixed_flow_gs=fixed_flow_gs,
            fixed_attribute_keys=fixed_attribute_keys,
        )
        if fixed_raw_gs is not None:
            out_gs = _copy_gt_attributes(out_gs, fixed_raw_gs, fixed_attribute_keys)
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

        if save_residuals:
            residual_keys = sorted([key for key in out_gs.keys() if key in input_gs])
            residuals = {}
            residual_stats = {}
            for key in residual_keys:
                residual = out_gs[key] - input_gs[key]
                residuals[key] = residual
                residual_stats[key] = {
                    "mean": float(residual.mean().item()),
                    "abs_mean": float(residual.abs().mean().item()),
                }

            scene_stem = f"{int(scene_idx)}_{_sanitize_for_filename(scene_name)}"
            pt_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "residual_type": "sampled_out_minus_input",
                "flow_steps": int(flow_steps),
                "flow_space": flow_space,
                "residual_keys": residual_keys,
                "residuals": _to_cpu(residuals),
                "input_gs": _to_cpu(input_gs),
                "output_gs": _to_cpu(out_gs),
                "target_gs": _to_cpu(gt_gs) if gt_gs is not None else None,
                "cameras": _to_cpu(eval_cameras),
            }
            torch.save(pt_payload, os.path.join(residual_dir, f"{scene_stem}.pt"))

            stats_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "num_gaussians": int(input_gs["means"].shape[0]),
                "residual_type": "sampled_out_minus_input",
                "flow_steps": int(flow_steps),
                "flow_space": flow_space,
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


def _resolve_flow_cfg():
    cfg = flow_matching()
    if FLAGS.flow_steps is not None:
        cfg["flow_steps"] = FLAGS.flow_steps
    if FLAGS.flow_space is not None:
        cfg["flow_space"] = FLAGS.flow_space
    if FLAGS.flow_noise_std is not None:
        cfg["flow_noise_std"] = FLAGS.flow_noise_std
    if cfg["flow_space"] not in FLOW_SPACES:
        raise ValueError(f"Unsupported flow_space={cfg['flow_space']}; expected one of {sorted(FLOW_SPACES)}")
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

    dataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][FLAGS.input_factor]
    target_factor_entry = scene["factor_data"][FLAGS.target_factor]

    train_payload = _build_split_payload(dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train")
    eval_payload = _build_split_payload(dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test")

    model = GSFlowPredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    target_gs_raw = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
    loss_features = _parse_loss_features(FLAGS.loss_features, model, target_gs_raw)
    fixed_attribute_keys = _fixed_attribute_keys(loss_features, target_gs_raw)

    input_gs_raw = build_densified_input_gs(
        input_factor_entry=input_factor_entry,
        target_factor_entry=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        output_dir=FLAGS.output_dir,
        gt_attribute_keys=fixed_attribute_keys,
    )
    input_gs_raw = _copy_gt_attributes(input_gs_raw, target_gs_raw, fixed_attribute_keys)
    source_flow_gs = raw_to_flow_gs(input_gs_raw, flow_cfg["flow_space"])
    target_flow_gs = raw_to_flow_gs(target_gs_raw, flow_cfg["flow_space"])
    source_flow_gs = _apply_fixed_flow_attributes(source_flow_gs, target_flow_gs, fixed_attribute_keys)
    batch_scene_idx = [scene["idx"]]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

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
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"input_gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"densified_gaussians={input_gs_raw['means'].shape[0]} "
        f"target_gaussians={target_gs_raw['means'].shape[0]} "
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} "
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} "
        f"loss_features={','.join(loss_features)} "
        f"fixed_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'} "
        f"flow_space={flow_cfg['flow_space']} flow_steps={flow_cfg['flow_steps']} "
        f"flow_noise_std={flow_cfg['flow_noise_std']} "
        f"flow_t_eps={flow_cfg['flow_t_eps']} "
        f"post_activate_loss={FLAGS.post_activate_loss} "
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} "
        f"loss_weights={mse_loss_cfg['loss_weights']}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    train_images_device = gpu_utils.move_to_device(train_payload["images"], device)
    train_cameras_device = gpu_utils.move_to_device(train_payload["cameras"], device)
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
        query_flow_gs = _apply_fixed_flow_attributes(query_flow_gs, target_flow_gs, fixed_attribute_keys)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_vel = model(batch_flow_gs=[query_flow_gs], batch_scene_idx=batch_scene_idx, t=t)[0]
            x1_pred_flow_gs = predict_x1_from_velocity(
                model, query_flow_gs, pred_vel, flow_noise, gamma, gamma_dot, t
            )
            x1_pred_raw_gs = flow_to_raw_gs(x1_pred_flow_gs, flow_cfg["flow_space"])
            x1_pred_raw_gs = _copy_gt_attributes(x1_pred_raw_gs, target_gs_raw, fixed_attribute_keys)
            total_loss, attr_losses, weighted_attr_losses = _feature_mse_loss(
                x1_pred_raw_gs,
                target_gs_raw,
                loss_features,
                post_activate_loss=FLAGS.post_activate_loss,
                loss_weights=mse_loss_cfg["loss_weights"],
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
                    flow_cfg["flow_space"],
                    fixed_flow_gs=target_flow_gs,
                    fixed_attribute_keys=fixed_attribute_keys,
                )
                train_out_gs = _copy_gt_attributes(train_out_gs, target_gs_raw, fixed_attribute_keys)
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(train_out_gs, train_cameras_device)
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs[:9]]
                if len(pred_imgs_uint8) > 0:
                    pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=input_gs_raw,
                source_flow_gs=source_flow_gs,
                gt_gs=target_gs_raw,
                scene_idx=eval_payload["scene_idx"],
                scene_name=eval_payload["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=eval_payload["images_name"],
                output_dir=eval_dir,
                flow_steps=int(flow_cfg["flow_steps"]),
                flow_space=flow_cfg["flow_space"],
                fixed_raw_gs=target_gs_raw,
                fixed_flow_gs=target_flow_gs,
                fixed_attribute_keys=fixed_attribute_keys,
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                save_residuals=FLAGS.save_residuals,
                output_gt=(step == 0),
            )
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Eval step {step}: {metric_str}")
            if FLAGS.compare_with_input:
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                logger.info(f"Eval input step {step}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = evaluate_single_scene(
        model=model,
        input_gs=input_gs_raw,
        source_flow_gs=source_flow_gs,
        gt_gs=target_gs_raw,
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=final_eval_dir,
        flow_steps=int(flow_cfg["flow_steps"]),
        flow_space=flow_cfg["flow_space"],
        fixed_raw_gs=target_gs_raw,
        fixed_flow_gs=target_flow_gs,
        fixed_attribute_keys=fixed_attribute_keys,
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")


if __name__ == "__main__":
    app.run(main)

import json
import os
import random
import sys

import cv2
import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as densification source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as overfit target")
flags.DEFINE_enum("alignment", "emd", ["emd", "nearest"], "Interpolated-to-target alignment method")
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_string(
    "loss_features",
    "",
    "Comma-separated Gaussian attributes to include in point-wise MSE loss. "
    "Defaults to FeaturePredictor.output_features.",
)
flags.DEFINE_boolean(
    "post_activate_loss",
    False,
    "Use feature-specific loss transforms: log-space scales, sigmoid opacities, and geodesic quats.",
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

INPUT_FACTOR = 4
TARGET_FACTOR = 2
SUPPORTED_GS_KEYS = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.


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
    total_steps = gin.REQUIRED,
    pretrain_steps = gin.REQUIRED,
    eval_interval = gin.REQUIRED,
    log_interval = gin.REQUIRED,
    save_interval = gin.REQUIRED,
    log_image_interval = gin.REQUIRED,
    grad_clip_norm = gin.REQUIRED,
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
        raise ValueError("No Gaussian attributes selected for MSE loss")

    supported = set(SUPPORTED_GS_KEYS)
    unsupported = [feature for feature in loss_features if feature not in supported]
    if unsupported:
        raise ValueError(
            f"Unsupported loss feature(s): {unsupported}. "
            f"Supported features are: {SUPPORTED_GS_KEYS}"
        )

    missing = [feature for feature in loss_features if feature not in target_gs]
    if missing:
        raise ValueError(f"Selected loss feature(s) missing from target GS: {missing}")

    return loss_features


def _fixed_attribute_keys(loss_features, target_gs):
    loss_set = set(loss_features)
    return [key for key in SUPPORTED_GS_KEYS if key in target_gs and key not in loss_set]


def _copy_gt_attributes(gs, target_gs, attribute_keys):
    for key in attribute_keys:
        if key in gs and key in target_gs:
            gs[key] = target_gs[key].to(device=gs[key].device, dtype=gs[key].dtype).clone()
    return gs


def _scale_means_origin(gs, scale):
    scaled_gs = {key: value.clone() for key, value in gs.items()}
    if "means" in scaled_gs:
        scaled_gs["means"] = scaled_gs["means"] * float(scale)
    return scaled_gs


def _unscale_means_origin(gs, scale):
    unscaled_gs = {key: value.clone() for key, value in gs.items()}
    if "means" in unscaled_gs:
        unscaled_gs["means"] = unscaled_gs["means"] / float(scale)
    return unscaled_gs


def _feature_loss_value(key, pred, target, post_activate_loss=False, quat_direct_mse=False):
    if key == "means":
        # squared_error = (pred - target).square()
        squared_error = (pred - target).abs()
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

    # Means, SH/color features, and scales use their stored parameterization.
    # Scales are stored as log-scales, so this is log-space MSE rather than exp-space MSE.
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
    total_loss = None
    if loss_weights is None:
        loss_weights = {key: 1.0 for key in loss_features}

    for key in loss_features:
        if key not in out_gs:
            raise ValueError(f"Selected loss feature '{key}' missing from model output")
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
        weighted_loss = float(loss_weights.get(key, 1.0)) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

    if total_loss is None:
        raise ValueError("No MSE losses were computed")
    if not total_loss.requires_grad:
        raise ValueError(
            "Selected loss features do not receive gradients. "
            "Make sure FeaturePredictor.output_features includes at least one selected loss feature."
        )
    return total_loss, losses, weighted_losses

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



def _as_scaler_tensor(scaler, name, device, dtype):
    value = getattr(scaler, name)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def _as_scale_tensor(scaler, device, dtype):
    return _as_scaler_tensor(scaler, "scale_", device, dtype)


def _scaler_transform(scaler, value):
    device = value.device
    dtype = value.dtype
    scale = _as_scaler_tensor(scaler, "scale_", device, dtype)
    trans = _as_scaler_tensor(scaler, "trans_", device, dtype)
    return value * scale + trans


def _scaler_inverse_transform(scaler, value):
    device = value.device
    dtype = value.dtype
    scale = _as_scaler_tensor(scaler, "scale_", device, dtype)
    trans = _as_scaler_tensor(scaler, "trans_", device, dtype)
    return (value - trans) / scale


def _convert_gs_to_target_frame(input_gs, input_scaler, target_scaler):
    target_gs = {}
    device = input_gs["means"].device
    dtype = input_gs["means"].dtype

    raw_means = _scaler_inverse_transform(input_scaler, input_gs["means"])
    target_gs["means"] = _scaler_transform(target_scaler, raw_means)

    input_scale = _as_scale_tensor(input_scaler, device, dtype)
    target_scale = _as_scale_tensor(target_scaler, device, dtype)
    for key, value in input_gs.items():
        if key == "means":
            continue
        if key == "scales":
            target_gs[key] = value - torch.log(input_scale) + torch.log(target_scale)
        else:
            target_gs[key] = value.clone()
    return target_gs


def _chunked_knn_indices(points, centers, k, chunk_size=512):
    if points.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("Cannot query kNN on empty point sets")
    k = min(int(k), points.shape[0])
    idx_chunks = []
    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        dist = torch.cdist(centers[start:end].float(), points.float())
        idx_chunks.append(torch.topk(dist, k=k, dim=1, largest=False).indices)
    return torch.cat(idx_chunks, dim=0)


def _chunked_nearest_indices(source_means, target_means, chunk_size=512):
    return _chunked_knn_indices(source_means, target_means, 1, chunk_size=chunk_size).squeeze(1)


_POINTCEPT_POINTOPS = None


def _get_pointcept_pointops():
    global _POINTCEPT_POINTOPS
    if _POINTCEPT_POINTOPS is not None:
        return _POINTCEPT_POINTOPS

    pointcept_libs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pointcept", "libs")
    if pointcept_libs not in sys.path:
        sys.path.append(pointcept_libs)
    try:
        import pointops
    except Exception as exc:
        raise ImportError(
            "Could not import Pointcept pointops. Make sure Pointcept/libs/pointops "
            "is built or installed in the active environment."
        ) from exc
    missing = [name for name in ["knn_query", "farthest_point_sampling"] if not hasattr(pointops, name)]
    if missing:
        raise ImportError(f"Pointcept pointops is missing required API(s): {missing}")
    _POINTCEPT_POINTOPS = pointops
    return _POINTCEPT_POINTOPS


def _midpoint_interpolate_gs(input_gs, target_count):
    source_count = input_gs["means"].shape[0]
    if source_count <= 0:
        raise ValueError("Cannot densify an empty input GS")
    if target_count < source_count:
        raise ValueError(
            f"Cannot preserve {source_count} source Gaussians when target_count={target_count}"
        )

    new_count = target_count - source_count
    if new_count == 0:
        return {key: value.clone() for key, value in input_gs.items()}
    if source_count == 1:
        raise ValueError("Cannot create midpoint Gaussians from a single source Gaussian")

    means = input_gs["means"].contiguous()
    if not means.is_cuda:
        raise ValueError("Pointcept pointops midpoint interpolation requires CUDA tensors")
    pointops = _get_pointcept_pointops()

    up_rate = float(target_count) / float(source_count)
    k = min(source_count, int(2 * up_rate))
    if k < 2:
        raise ValueError(f"Need at least 2 neighbors for midpoint interpolation, got k={k}")

    offset = torch.tensor([source_count], device=means.device, dtype=torch.int32)
    nn_idx, _ = pointops.knn_query(k, means, offset, means, offset)
    nn_idx = nn_idx.long()
    src_idx = torch.arange(source_count, device=means.device).unsqueeze(1).expand(-1, k).reshape(-1)
    nbr_idx = nn_idx.reshape(-1)
    non_self = src_idx != nbr_idx
    src_idx = src_idx[non_self]
    nbr_idx = nbr_idx[non_self]

    candidate_count = src_idx.shape[0]
    if candidate_count < new_count:
        raise ValueError(
            f"PUFM midpoint interpolation produced {candidate_count} new candidates, "
            f"but target requires {new_count}. source_count={source_count}, k={k}"
        )

    candidate_means = ((means[src_idx] + means[nbr_idx]) * 0.5).contiguous()
    if candidate_count > new_count:
        candidate_offset = torch.tensor([candidate_count], device=means.device, dtype=torch.int32)
        new_offset = torch.tensor([new_count], device=means.device, dtype=torch.int32)
        keep_idx = pointops.farthest_point_sampling(candidate_means, candidate_offset, new_offset).long()
        src_idx = src_idx[keep_idx]
        nbr_idx = nbr_idx[keep_idx]

    interpolated = {}
    for key, value in input_gs.items():
        src_value = value[src_idx]
        nbr_value = value[nbr_idx]

        # Previous midpoint behavior, kept here for quick local toggling:
        # if key == "quats":
        #     sign = torch.sign((src_value * nbr_value).sum(dim=-1, keepdim=True))
        #     sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        #     nbr_value = nbr_value * sign
        #     midpoint_value = F.normalize((src_value + nbr_value) * 0.5, dim=-1)
        # else:
        #     midpoint_value = (src_value + nbr_value) * 0.5

        if key in ["means", "features_dc", "features_rest"]:
            midpoint_value = (src_value + nbr_value) * 0.5
        elif key == "quats":
            midpoint_value = torch.zeros_like(src_value)
            midpoint_value[:, 0] = 1
        elif key == "opacities":
            low_opacity = torch.full_like(src_value, 0.01)
            midpoint_value = torch.logit(low_opacity)
        elif key == "scales":
            pair_min_scale = torch.minimum(src_value, nbr_value).min(dim=-1, keepdim=True).values
            midpoint_value = pair_min_scale.repeat(1, src_value.shape[-1])
        else:
            midpoint_value = (src_value + nbr_value) * 0.5

        interpolated[key] = torch.cat([value.clone(), midpoint_value], dim=0)
    return interpolated


def _align_nearest_target_to_source(source_means, target_means):
    return _chunked_nearest_indices(source_means, target_means)


def _align_emd_target_to_source(source_means, target_means, eps, iters):
    try:
        from emd_assignment import emd_module
    except Exception as exc:
        raise ImportError(
            "Could not import local EMD package. Run "
            "`cd /home/ricky/SplatFormer/emd_assignment && python setup.py install`, "
            "or rerun with `--alignment=nearest`."
        ) from exc

    if source_means.shape[0] != target_means.shape[0]:
        raise ValueError(
            f"EMD alignment requires equal counts, got {source_means.shape[0]} and {target_means.shape[0]}"
        )

    count = source_means.shape[0]
    padded_count = int(np.ceil(count / 128.0) * 128)
    source_pad = source_means
    target_pad = target_means
    if padded_count != count:
        pad = padded_count - count
        source_pad = torch.cat([source_means, source_means[-1:].expand(pad, -1)], dim=0)
        target_pad = torch.cat([target_means, target_means[-1:].expand(pad, -1)], dim=0)

    aligner = emd_module.emdModule()
    with torch.no_grad():
        _, assignment = aligner(
            source_pad.unsqueeze(0).contiguous(),
            target_pad.unsqueeze(0).contiguous(),
            float(eps),
            int(iters),
        )
    assignment = assignment[0, :count].detach().long()
    assigned_target = assignment.cpu()
    source_cpu = torch.arange(count, dtype=torch.long)
    dist_cpu = ((source_means - target_pad[assignment.clamp(min=0)].to(source_means.device)) ** 2).sum(dim=1).detach().cpu()

    best_source = torch.full((count,), -1, dtype=torch.long)
    best_dist = torch.full((count,), float("inf"))
    valid = (assigned_target >= 0) & (assigned_target < count)
    for src_i, tgt_i, dist_i in zip(source_cpu[valid].tolist(), assigned_target[valid].tolist(), dist_cpu[valid].tolist()):
        if dist_i < best_dist[tgt_i].item():
            best_dist[tgt_i] = dist_i
            best_source[tgt_i] = src_i

    missing = best_source < 0
    if missing.any():
        missing_idx = missing.nonzero(as_tuple=False).squeeze(1).to(target_means.device)
        nearest = _align_nearest_target_to_source(source_means, target_means[missing_idx]).detach().cpu()
        best_source[missing] = nearest

    return best_source.to(source_means.device)


def _nearest_neighbor_dist2(means):
    count = means.shape[0]
    if count <= 1:
        return torch.full((count,), 1e-7, device=means.device, dtype=means.dtype)
    nn_idx = _chunked_knn_indices(means, means, 2)
    nearest = means[nn_idx[:, 1]]
    dist2 = ((means - nearest) ** 2).sum(dim=-1)
    return torch.clamp_min(dist2, 1e-7)


def _apply_3dgs_attribute_init(densified_gs):
    means = densified_gs["means"]
    count = means.shape[0]
    device = means.device
    dtype = means.dtype

    if "scales" in densified_gs:
        dist2 = _nearest_neighbor_dist2(means)
        densified_gs["scales"] = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    if "quats" in densified_gs:
        quats = torch.zeros((count, 4), device=device, dtype=dtype)
        quats[:, 0] = 1
        densified_gs["quats"] = quats

    if "opacities" in densified_gs:
        opacity = 0.1 * torch.ones((count, 1), device=device, dtype=dtype)
        densified_gs["opacities"] = torch.logit(opacity)

    return densified_gs


def _densify_stage_gs(low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs):
    return {
        "00_low_res_gs.ply": low_res_gs,
        "01_interpolated_high_res_gs.ply": interpolated_gs,
        "02_gt_high_res_gs.ply": gt_high_res_gs,
        "03_input_high_res_gs.ply": input_high_res_gs,
    }


def _save_densify_stage_plys(output_dir, low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs):
    stage_dir = os.path.join(output_dir, "densify_init")
    for ply_name, gs in _densify_stage_gs(
        low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs
    ).items():
        gs_utils.export_ply_forviewer(gs, os.path.join(stage_dir, ply_name))


def _render_gs_average_metrics(gs, images, cameras, chunk_size, device):
    metric_computer = MetricComputer()
    num_views = len(images)
    if num_views == 0:
        raise ValueError("Cannot compute render metrics with zero views")
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_views
    chunk_size = min(chunk_size, num_views)

    with torch.no_grad():
        for start in range(0, num_views, chunk_size):
            end = min(start + chunk_size, num_views)
            chunk_images = gpu_utils.move_to_device(images[start:end], device)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)

            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            gt_imgs = torch.stack(chunk_images, dim=0)

            if gt_imgs.shape[-1] == 4:
                masks = gt_imgs[..., 3].unsqueeze(-1)
                pred_imgs = (pred_imgs * masks * 255).to(torch.uint8)
                gt_imgs = (gt_imgs[..., :3] * 255).to(torch.uint8)
            else:
                pred_imgs = (pred_imgs * 255).to(torch.uint8)
                gt_imgs = (gt_imgs * 255).to(torch.uint8)

            metric_computer.update(pred_imgs, gt_imgs, name=f"{start:06d}_{end:06d}")

    return metric_computer.finalize()


def _write_densify_stage_render_metrics(output_dir, stage_gs, images, cameras, chunk_size, device):
    stage_dir = os.path.join(output_dir, "densify_init")
    os.makedirs(stage_dir, exist_ok=True)
    metrics_by_ply = {}
    for ply_name, gs in stage_gs.items():
        metrics_by_ply[ply_name] = _render_gs_average_metrics(gs, images, cameras, chunk_size, device)
    with open(os.path.join(stage_dir, "render_metrics.json"), "w") as f:
        json.dump(metrics_by_ply, f, indent=2)
    return metrics_by_ply


def build_densified_input_gs(
    input_factor_entry,
    target_factor_entry,
    alignment,
    attribute_init,
    emd_eps,
    emd_iters,
    device,
    return_stages=False,
):
    input_gs = gpu_utils.move_to_device(input_factor_entry["gs_params"], device)
    target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
    target_count = target_gs["means"].shape[0]

    input_in_target_frame = _convert_gs_to_target_frame(
        input_gs,
        input_factor_entry["scaler"],
        target_factor_entry["scaler"],
    )
    interpolated_gs = _midpoint_interpolate_gs(input_in_target_frame, target_count)

    if alignment == "emd":
        source_idx = _align_emd_target_to_source(
            interpolated_gs["means"], target_gs["means"], emd_eps, emd_iters
        )
    elif alignment == "nearest":
        source_idx = _align_nearest_target_to_source(interpolated_gs["means"], target_gs["means"])
    else:
        raise ValueError(f"Unsupported alignment method: {alignment}")

    densified_gs = {}
    for key, value in target_gs.items():
        if key in interpolated_gs:
            densified_gs[key] = interpolated_gs[key][source_idx].clone()
        else:
            densified_gs[key] = value.clone()

    if attribute_init == "3dgs":
        densified_gs = _apply_3dgs_attribute_init(densified_gs)
    elif attribute_init != "aligned":
        raise ValueError(f"Unsupported attribute initialization: {attribute_init}")

    if return_stages:
        return densified_gs, _densify_stage_gs(
            low_res_gs=input_in_target_frame,
            interpolated_gs=interpolated_gs,
            gt_high_res_gs=target_gs,
            input_high_res_gs=densified_gs,
        )
    return densified_gs


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
    save_residuals=True,
    output_gt=True,
    fixed_gt_gs=None,
    fixed_attribute_keys=None,
    means_origin_scale=1.0,
):
    model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    predicted_keys = list(getattr(model, "output_features", []))
    if fixed_attribute_keys is None:
        fixed_attribute_keys = []

    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")

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
        if fixed_gt_gs is not None:
            out_gs = _copy_gt_attributes(out_gs, fixed_gt_gs, fixed_attribute_keys)
        out_gs = _unscale_means_origin(out_gs, means_origin_scale)

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
            gs_utils.export_ply_forviewer(
                gs_params=input_gs,
                filename=os.path.join(viewerdir, "point_cloud/input.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(viewerdir, "point_cloud/output.ply"),
            )
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(
                    gs_params=gt_gs,
                    filename=os.path.join(viewerdir, "point_cloud/gt.ply"),
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


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    mse_loss_cfg = feature_mse_loss()
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset: SplatFactoMultiLevelDataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][FLAGS.input_factor]
    target_factor_entry = scene["factor_data"][FLAGS.target_factor]

    train_payload = _build_split_payload(
        dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
    )
    eval_payload = _build_split_payload(
        dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test"
    )
    if len(eval_payload["images"]) == 0:
        eval_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
        )

    target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)

    model = FeaturePredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    loss_features = _parse_loss_features(FLAGS.loss_features, model, target_gs)
    fixed_attribute_keys = _fixed_attribute_keys(loss_features, target_gs)
    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = requested_means_origin_scale if "means" in loss_features else 1.0

    densified_input_gs, densify_stage_gs = build_densified_input_gs(
        input_factor_entry=input_factor_entry,
        target_factor_entry=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        return_stages=True,
    )
    densified_input_gs = _copy_gt_attributes(densified_input_gs, target_gs, fixed_attribute_keys)
    loss_target_gs = _scale_means_origin(target_gs, means_origin_scale)
    densify_stage_gs["03_input_high_res_gs.ply"] = densified_input_gs
    _save_densify_stage_plys(
        output_dir=FLAGS.output_dir,
        low_res_gs=densify_stage_gs["00_low_res_gs.ply"],
        interpolated_gs=densify_stage_gs["01_interpolated_high_res_gs.ply"],
        gt_high_res_gs=densify_stage_gs["02_gt_high_res_gs.ply"],
        input_high_res_gs=densify_stage_gs["03_input_high_res_gs.ply"],
    )
    batch_gs = gpu_utils.move_to_device([densified_input_gs], device)
    batch_scene_idx = [scene["idx"]]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

    densify_metrics = _write_densify_stage_render_metrics(
        output_dir=FLAGS.output_dir,
        stage_gs=densify_stage_gs,
        images=eval_images,
        cameras=eval_cameras,
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
    # Keep render-loss training gin keys accepted for compatibility; this script optimizes GS MSE only.
    _ = train_cfg["image_l1_loss_weight"]
    _ = train_cfg["lpips_loss_weight"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    print(
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} "
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"input_gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"densified_gaussians={batch_gs[0]['means'].shape[0]} "
        f"target_gaussians={target_factor_entry['gs_params']['means'].shape[0]} "
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} "
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} "
        f"loss_features={','.join(loss_features)} "
        f"post_activate_loss={FLAGS.post_activate_loss} "
        f"means_origin_scale={requested_means_origin_scale} "
        f"effective_means_origin_scale={means_origin_scale} "
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} "
        f"loss_weights={mse_loss_cfg['loss_weights']} "
        f"fixed_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'} "
    )
    logger.info(
        f"means_origin_scale={requested_means_origin_scale} "
        f"effective_means_origin_scale={means_origin_scale}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([train_payload["images"]], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_batch_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)
            out_gs = _copy_gt_attributes(out_batch_gs[0], target_gs, fixed_attribute_keys)
            total_loss, feature_losses, weighted_feature_losses = _feature_mse_loss(
                out_gs,
                loss_target_gs,
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

        feature_loss_values = {key: loss.item() for key, loss in feature_losses.items()}
        weighted_loss_values = {key: loss.item() for key, loss in weighted_feature_losses.items()}
        postfix = {"loss": f"{total_loss.item():.3e}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        pbar.set_postfix(postfix)

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            feature_loss_str = " ".join(
                f"{key}_loss={value:.6f} {key}_weighted={weighted_loss_values[key]:.6f}"
                for key, value in feature_loss_values.items()
            )
            logger.info(
                f"step={step} total={total_loss.item():.6f} "
                f"{feature_loss_str} lr={optimizer.param_groups[0]['lr']:.8f}"
            )

        if step % log_image_interval == 0:
            with torch.no_grad():
                log_out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                log_out_gs = _copy_gt_attributes(log_out_gs, target_gs, fixed_attribute_keys)
                log_out_gs = _unscale_means_origin(log_out_gs, means_origin_scale)
                batch_cameras = gpu_utils.move_to_device(train_payload["cameras"], device)
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(log_out_gs, batch_cameras)
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=batch_gs[0],
                gt_gs=target_gs,
                scene_idx=eval_payload["scene_idx"],
                scene_name=eval_payload["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=eval_payload["images_name"],
                output_dir=eval_dir,
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                save_residuals=FLAGS.save_residuals,
                output_gt=(step == 0),
                fixed_gt_gs=target_gs,
                fixed_attribute_keys=fixed_attribute_keys,
                means_origin_scale=means_origin_scale,
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
        input_gs=batch_gs[0],
        gt_gs=target_gs,
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=final_eval_dir,
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
        fixed_gt_gs=target_gs,
        fixed_attribute_keys=fixed_attribute_keys,
        means_origin_scale=means_origin_scale,
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")



if __name__ == "__main__":
    app.run(main)

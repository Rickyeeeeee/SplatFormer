import json
import os
import pickle
import random
import zipfile
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

from dataset.GS_multi import SplatFactoMultiLevelDataset
# from gs_flow import GSFlowFeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_3DGSoptimizer, build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_sr", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("compare_with_input", True, "Compare predictions with input 3DGS")
flags.DEFINE_boolean("save_viewer", False, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", False, "Save residual tensors and stats")
flags.DEFINE_boolean("use_wandb", True, "Log training and evaluation metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_integer(
    "min_train_splats_per_factor",
    10000,
    "Minimum raw splat count required for each SR train factor. Set <=0 to disable.",
)
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

INPUT_FACTOR = 4
TARGET_FACTOR = 1
EXPECTED_IMAGE_COUNT = 128
FACTOR_TO_IMAGE_DIR = {
    1: "images",
    2: "images_2",
    4: "images_4",
}
CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"
GAUSSIAN_MEANS_KEY = "_model.gauss_params.means"
WANDB_EVAL_IMAGE_SCENE = "3e288ee8aced4a0797e66d53536112b1"


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
    flow_loss_weight=1.0,
    flow_loss_type="mse",
    flow_t_min=0.0,
    flow_t_max=1.0,
    flow_eval_steps=1,
    flow_trajectory_mode="linear",
    flow_optim_steps=0,
    flow_optim_snapshot_interval=300,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
):
    del pretrain_steps
    return {
        "output_dir": output_dir,
        "total_steps": total_steps,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "log_image_interval": log_image_interval,
        "grad_clip_norm": grad_clip_norm,
        "flow_loss_weight": flow_loss_weight,
        "flow_loss_type": flow_loss_type,
        "flow_t_min": flow_t_min,
        "flow_t_max": flow_t_max,
        "flow_eval_steps": flow_eval_steps,
        "flow_trajectory_mode": flow_trajectory_mode,
        "flow_optim_steps": flow_optim_steps,
        "flow_optim_snapshot_interval": flow_optim_snapshot_interval,
        "resume_from_step": resume_from_step,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


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


def _flow_float_keys(model, input_gs, target_gs):
    keys = []
    for key in getattr(model, "output_features", []):
        if key not in input_gs or key not in target_gs:
            continue
        if not torch.is_tensor(input_gs[key]) or not torch.is_tensor(target_gs[key]):
            continue
        if input_gs[key].shape != target_gs[key].shape:
            raise ValueError(
                f"Flow matching currently requires same GS shape for '{key}', "
                f"got {tuple(input_gs[key].shape)} and {tuple(target_gs[key].shape)}"
            )
        if input_gs[key].is_floating_point():
            keys.append(key)
    if len(keys) == 0:
        raise ValueError("No floating output features are shared by input and target GS")
    return keys


def _sample_flow_t(device, t_min=0.0, t_max=1.0):
    return torch.empty(1, device=device).uniform_(float(t_min), float(t_max))


def _interpolate_gs(input_gs, target_gs, t, flow_keys):
    t_value = t.reshape(()).to(input_gs["means"].device)
    out = {}
    flow_key_set = set(flow_keys)
    for key, value in input_gs.items():
        if key in flow_key_set:
            out[key] = value + t_value * (target_gs[key] - value)
        else:
            out[key] = value
    return out


def _flow_velocity_loss(pred_gs, xt_gs, input_gs, target_gs, flow_keys, loss_type="mse", time_delta=1.0):
    losses = {}
    total = None
    for key in flow_keys:
        pred_velocity = pred_gs[key] - xt_gs[key]
        target_velocity = (target_gs[key] - input_gs[key]) / time_delta
        if loss_type == "l1":
            loss = (pred_velocity - target_velocity).abs().mean()
        elif loss_type == "mse":
            loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
        else:
            raise ValueError(f"Unsupported flow_loss_type: {loss_type}")
        losses[key] = loss
        total = loss if total is None else total + loss
    return total / len(flow_keys), losses


def _detach_gs(gs):
    out = {}
    for key, value in gs.items():
        out[key] = value.detach().clone() if torch.is_tensor(value) else value
    return out


def _clone_trainable_gs(input_gs, flow_keys):
    state = {}
    trainable = {}
    flow_key_set = set(flow_keys)
    for key, value in input_gs.items():
        if torch.is_tensor(value):
            cloned = value.detach().clone()
            if key in flow_key_set and cloned.is_floating_point():
                cloned.requires_grad_(True)
                trainable[key] = cloned
            state[key] = cloned
        else:
            state[key] = value
    return state, trainable


def _render_l1_loss(gs, images, cameras):
    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, cameras)
    loss = 0
    for pred_img, gt_img in zip(pred_imgs, images):
        gt_rgb = gt_img[..., :3]
        if gt_img.shape[-1] == 4:
            mask = gt_img[..., 3:].to(pred_img.dtype)
            loss = loss + ((pred_img - gt_rgb) * mask).abs().mean()
        else:
            loss = loss + (pred_img - gt_rgb).abs().mean()
    return loss / max(len(pred_imgs), 1)


def build_optimized_gs_trajectory(input_gs, flow_keys, images, cameras, optim_steps, snapshot_interval, logger=None):
    if optim_steps <= 0:
        raise ValueError("flow_optim_steps must be positive for optimized_prefix trajectory mode")
    if snapshot_interval <= 0:
        raise ValueError("flow_optim_snapshot_interval must be positive")

    state, trainable = _clone_trainable_gs(input_gs, flow_keys)
    if len(trainable) == 0:
        raise ValueError("No trainable GS tensors selected for trajectory optimization")
    with gin.config_scope("flow_optim"):
        optimizer = build_3DGSoptimizer(trainable)

    trajectory = [(0, _detach_gs(state))]
    for opt_step in range(1, optim_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _render_l1_loss(state, images, cameras)
        loss.backward()
        optimizer.step()
        if opt_step % snapshot_interval == 0 or opt_step == optim_steps:
            trajectory.append((opt_step, _detach_gs(state)))
            if logger is not None:
                logger.info(f"flow_optim step={opt_step}/{optim_steps} image_l1={loss.item():.6f}")
    return trajectory


def sample_optimized_prefix_pair(trajectory, total_steps, device):
    if len(trajectory) < 2:
        raise ValueError("Optimized trajectory needs at least one nonzero snapshot")
    sample_idx = int(torch.randint(1, len(trajectory), (1,), device=device).item())
    end_step, end_gs = trajectory[sample_idx]
    start_gs = trajectory[0][1]
    t = torch.tensor([float(end_step) / float(total_steps)], device=device)
    return start_gs, end_gs, t, t.clamp_min(1e-6)


def integrate_flow(model, input_gs, scene_idx, num_steps=1):
    if num_steps <= 0:
        raise ValueError("flow_eval_steps must be positive")
    current_gs = input_gs
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.tensor([(step + 0.5) * dt], device=input_gs["means"].device)
        pred_gs = model(
            batch_normalized_gs=[current_gs],
            batch_scene_idx=[scene_idx],
            timestep=t,
        )[0]
        next_gs = dict(current_gs)
        for key in getattr(model, "output_features", []):
            if key in pred_gs and key in current_gs and torch.is_tensor(current_gs[key]):
                next_gs[key] = current_gs[key] + dt * (pred_gs[key] - current_gs[key])
        current_gs = next_gs
    return current_gs


def _default_wandb_name(output_dir):
    output_dir = output_dir.rstrip("/")
    parts = Path(output_dir).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else output_dir


def _init_wandb(output_dir):
    if not FLAGS.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

    run = wandb.init(
        project=FLAGS.wandb_project,
        dir=FLAGS.wandb_dir,
        name=FLAGS.wandb_name or _default_wandb_name(output_dir),
        config={
            "output_dir": output_dir,
            "eval_subdir": FLAGS.eval_subdir,
            "compare_with_input": FLAGS.compare_with_input,
            "save_viewer": FLAGS.save_viewer,
            "save_residuals": FLAGS.save_residuals,
            "min_train_splats_per_factor": FLAGS.min_train_splats_per_factor,
            "gin_config": gin.operative_config_str(),
        },
    )
    return run


def _wandb_log(data, step=None):
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def _list_pngs(image_dir: Path):
    return sorted([path for path in image_dir.iterdir() if path.suffix.lower() == ".png"])


def _load_scene_roots(root_or_txt: str, tag: str):
    if root_or_txt.endswith(".txt"):
        scene_roots = []
        with open(root_or_txt, "r") as f:
            for line in f:
                path = line.strip()
                if path:
                    scene_roots.append(Path(path))
    else:
        root = Path(root_or_txt)
        if not root.is_dir():
            raise ValueError(f"Invalid {tag} path: {root_or_txt}")
        scene_roots = sorted([path for path in root.iterdir() if path.is_dir()])

    for path in scene_roots:
        if not path.is_dir():
            raise FileNotFoundError(f"{tag} scene root does not exist: {path}")
    return scene_roots


def _scene_map(scene_roots, tag: str):
    scene_map = {}
    for path in scene_roots:
        scene_name = path.name
        if scene_name in scene_map:
            raise ValueError(f"Duplicate scene name '{scene_name}' in {tag}: {path}")
        scene_map[scene_name] = path
    return scene_map


def _validate_scene_structure(scene_info, factors, scope):
    scene_name = scene_info["scene_name"]
    colmap_dir = Path(scene_info["colmap_dir"])
    if not colmap_dir.is_dir():
        return f"{scope}:{scene_name}: missing_colmap_scene:{colmap_dir}"

    factor_paths = scene_info["factor_paths"]
    for factor in factors:
        if factor not in FACTOR_TO_IMAGE_DIR:
            return f"{scope}:{scene_name}: unsupported_factor:{factor}"

        paths = factor_paths[factor]
        nerfstudio_dir = Path(paths["nerfstudio_dir"])
        image_dir = Path(paths["image_dir"])
        expected_image_dir = colmap_dir / FACTOR_TO_IMAGE_DIR[factor]

        if image_dir != expected_image_dir:
            return (
                f"{scope}:{scene_name}: factor={factor}: unexpected_image_dir: "
                f"got={image_dir} expected={expected_image_dir}"
            )
        if not image_dir.is_dir():
            return f"{scope}:{scene_name}: factor={factor}: missing_image_dir:{image_dir}"

        image_paths = _list_pngs(image_dir)
        if len(image_paths) != EXPECTED_IMAGE_COUNT:
            return (
                f"{scope}:{scene_name}: factor={factor}: bad_image_count: "
                f"got={len(image_paths)} expected={EXPECTED_IMAGE_COUNT} dir={image_dir}"
            )

        models_dir = nerfstudio_dir / "nerfstudio_models"
        if not models_dir.is_dir():
            return f"{scope}:{scene_name}: factor={factor}: missing_nerfstudio_models:{models_dir}"
        if len(sorted(models_dir.glob("step-*.ckpt"))) == 0:
            return f"{scope}:{scene_name}: factor={factor}: missing_ckpt:{models_dir}"

        camera_path = nerfstudio_dir / CAMERA_METADATA_NAME
        if not camera_path.is_file():
            return f"{scope}:{scene_name}: factor={factor}: missing_camera_metadata:{camera_path}"

    return None


class _TensorShape:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _rebuild_shape_tensor(storage, storage_offset, size, stride):
    del storage, storage_offset, stride
    return _TensorShape(size)


def _rebuild_shape_tensor_v2(
    storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None
):
    del storage, storage_offset, stride, requires_grad, backward_hooks, metadata
    return _TensorShape(size)


def _rebuild_shape_tensor_v3(storage, storage_offset, size, stride, *args):
    del storage, storage_offset, stride, args
    return _TensorShape(size)


def _rebuild_shape_parameter(data, requires_grad, backward_hooks):
    del requires_grad, backward_hooks
    return data


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        del pid
        return object()

    def find_class(self, module, name):
        tensor_rebuilders = {
            "_rebuild_tensor": _rebuild_shape_tensor,
            "_rebuild_tensor_v2": _rebuild_shape_tensor_v2,
            "_rebuild_tensor_v3": _rebuild_shape_tensor_v3,
            "_rebuild_parameter": _rebuild_shape_parameter,
        }
        if module == "torch._utils" and name in tensor_rebuilders:
            return tensor_rebuilders[name]
        return super().find_class(module, name)


def _load_checkpoint_metadata(ckpt_file: Path):
    try:
        with zipfile.ZipFile(ckpt_file) as ckpt_zip:
            try:
                data_name = next(name for name in ckpt_zip.namelist() if name.endswith("/data.pkl"))
            except StopIteration as exc:
                raise ValueError(f"missing_data_pkl:{ckpt_file}") from exc
            with ckpt_zip.open(data_name) as f:
                return _CheckpointMetadataUnpickler(f).load()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"unsupported_checkpoint_format:{ckpt_file}") from exc


def _raw_splat_count_from_checkpoint(nerfstudio_dir: Path):
    models_dir = nerfstudio_dir / "nerfstudio_models"
    ckpt_files = sorted(
        models_dir.glob("step-*.ckpt"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"missing_ckpt:{models_dir}")

    ckpt_file = ckpt_files[-1]
    ckpt_metadata = _load_checkpoint_metadata(ckpt_file)
    if GAUSSIAN_MEANS_KEY not in ckpt_metadata:
        raise KeyError(f"missing_key:{GAUSSIAN_MEANS_KEY}:{ckpt_file}")
    return int(ckpt_metadata[GAUSSIAN_MEANS_KEY].shape[0])


def _validate_train_splat_counts(scene_info, factors, threshold):
    if threshold <= 0:
        return None

    required_factors = sorted({INPUT_FACTOR, TARGET_FACTOR})
    missing_factors = sorted(set(required_factors) - set(factors))
    if missing_factors:
        return (
            f"train_dataset:{scene_info['scene_name']}: missing_required_splat_factors:"
            f"{missing_factors}"
        )

    scene_name = scene_info["scene_name"]
    for factor in required_factors:
        nerfstudio_dir = Path(scene_info["factor_paths"][factor]["nerfstudio_dir"])
        try:
            count = _raw_splat_count_from_checkpoint(nerfstudio_dir)
        except Exception as exc:
            return (
                f"train_dataset:{scene_name}: factor={factor}: raw_splat_count_error: "
                f"{type(exc).__name__}: {exc}"
            )
        if count <= threshold:
            return (
                f"{scene_name}: factor={factor} count={count} "
                f"threshold={threshold}"
            )

    return None


def _prepare_dataset_roots(scope, output_dir):
    nerfstudio_folder = gin.query_parameter(f"{scope}/SplatFactoMultiLevelDataset.nerfstudio_folder")
    colmap_folder = gin.query_parameter(f"{scope}/SplatFactoMultiLevelDataset.colmap_folder")
    factors = sorted(set(gin.query_parameter(f"{scope}/SplatFactoMultiLevelDataset.factors")))

    ns_roots = _load_scene_roots(nerfstudio_folder, f"{scope}/nerfstudio")
    colmap_roots = _load_scene_roots(colmap_folder, f"{scope}/colmap")
    ns_map = _scene_map(ns_roots, f"{scope}/nerfstudio")
    colmap_map = _scene_map(colmap_roots, f"{scope}/colmap")

    common_names = sorted(set(ns_map.keys()) & set(colmap_map.keys()))
    missing_in_colmap = sorted(set(ns_map.keys()) - set(colmap_map.keys()))
    missing_in_nerfstudio = sorted(set(colmap_map.keys()) - set(ns_map.keys()))

    messages = []
    if missing_in_colmap:
        preview = missing_in_colmap[:10]
        messages.append(
            f"{scope}: skipping {len(missing_in_colmap)} scenes missing in colmap, e.g. {preview}"
        )
    if missing_in_nerfstudio:
        preview = missing_in_nerfstudio[:10]
        messages.append(
            f"{scope}: skipping {len(missing_in_nerfstudio)} scenes missing in nerfstudio, e.g. {preview}"
        )

    valid_ns_paths = []
    valid_colmap_paths = []
    invalid_messages = []
    low_splat_messages = []
    splat_filter_enabled = scope == "train_dataset" and FLAGS.min_train_splats_per_factor > 0
    for scene_name in tqdm(common_names):
        ns_scene_root = ns_map[scene_name]
        colmap_scene_root = colmap_map[scene_name]
        factor_paths = {}
        for factor in factors:
            factor_paths[factor] = {
                "nerfstudio_dir": str(ns_scene_root / f"df-{factor}" / "splatfacto"),
                "image_dir": str(colmap_scene_root / FACTOR_TO_IMAGE_DIR[factor]),
            }
        scene_info = {
            "scene_name": scene_name,
            "colmap_dir": str(colmap_scene_root),
            "factor_paths": factor_paths,
        }
        problem = _validate_scene_structure(scene_info, factors, scope)
        if problem is not None:
            invalid_messages.append(problem)
            continue

        if splat_filter_enabled:
            splat_problem = _validate_train_splat_counts(
                scene_info, factors, FLAGS.min_train_splats_per_factor
            )
            if splat_problem is not None:
                low_splat_messages.append(splat_problem)
                continue

        valid_ns_paths.append(ns_scene_root)
        valid_colmap_paths.append(colmap_scene_root)

    if invalid_messages:
        preview = invalid_messages[:20]
        messages.append(
            f"{scope}: skipping {len(invalid_messages)} structurally invalid scenes:\n" + "\n".join(preview)
        )
        if len(invalid_messages) > 20:
            messages.append(f"{scope}: ... and {len(invalid_messages) - 20} more invalid scenes")

    if low_splat_messages:
        preview = low_splat_messages[:20]
        messages.append(
            f"{scope}: skipping {len(low_splat_messages)} scenes below raw splat threshold "
            f"{FLAGS.min_train_splats_per_factor}:\n" + "\n".join(preview)
        )
        if len(low_splat_messages) > 20:
            messages.append(f"{scope}: ... and {len(low_splat_messages) - 20} more low-splat scenes")

    if len(valid_ns_paths) == 0:
        details = "\n".join(messages) if messages else "No shared valid scenes found."
        raise ValueError(f"No valid scenes available for {scope}.\n{details}")

    filter_dir = Path(output_dir) / "dataset_filters"
    filter_dir.mkdir(parents=True, exist_ok=True)
    ns_filter = filter_dir / f"{scope}_nerfstudio.txt"
    colmap_filter = filter_dir / f"{scope}_colmap.txt"

    with ns_filter.open("w", encoding="utf-8") as f:
        for path in valid_ns_paths:
            f.write(f"{path}\n")
    with colmap_filter.open("w", encoding="utf-8") as f:
        for path in valid_colmap_paths:
            f.write(f"{path}\n")

    messages.append(f"{scope}: using {len(valid_ns_paths)} shared valid scenes")
    return str(ns_filter), str(colmap_filter), messages


def _validate_dataset_structure(dataset, scope):
    problems = []
    dataset_factors = sorted(set(dataset.factors))
    for scene_info in dataset.folders:
        problem = _validate_scene_structure(scene_info, dataset_factors, scope)
        if problem is not None:
            problems.append(problem)

    if problems:
        preview = "\n".join(problems[:20])
        if len(problems) > 20:
            preview += f"\n... and {len(problems) - 20} more"
        raise ValueError(f"Dataset structure validation failed for {scope}:\n{preview}")


def _build_dataset(scope, output_dir):
    nerfstudio_filter, colmap_filter, prep_messages = _prepare_dataset_roots(scope, output_dir)
    for message in prep_messages:
        print(message)

    with gin.config_scope(scope):
        dataset = SplatFactoMultiLevelDataset(
            nerfstudio_folder=nerfstudio_filter,
            colmap_folder=colmap_filter,
        )
    required_factors = {INPUT_FACTOR, TARGET_FACTOR}
    dataset_factors = set(dataset.factors)
    if not required_factors.issubset(dataset_factors):
        raise ValueError(
            f"{scope} dataset factors {sorted(dataset.factors)} do not include "
            f"required INPUT_FACTOR={INPUT_FACTOR} and TARGET_FACTOR={TARGET_FACTOR}"
        )
    _validate_dataset_structure(dataset, scope)
    return dataset


def _background_for_dataset(dataset):
    if dataset.background_color == "random":
        return torch.rand(3)
    return torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0


def _build_split_payload(dataset, scene_idx, scene_name, factor_entry, split):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]
    background = _background_for_dataset(dataset)

    total_num = len(meta["camera_to_worlds"])
    if total_num == 0:
        raise ValueError(f"Scene '{scene_name}' has zero cameras for factor payload")

    if split == "train":
        if dataset.image_per_scene is None:
            sample_num = total_num
        else:
            sample_num = min(dataset.image_per_scene, total_num)
        cam_ids = np.random.permutation(total_num)[:sample_num]
    elif split == "test":
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


class SceneSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.remaining = []

    def next_scene(self):
        if len(self.remaining) == 0:
            self.remaining = list(range(len(self.dataset.folders)))
            random.shuffle(self.remaining)
        scene_idx = self.remaining.pop()
        return self.dataset.load_scene(scene_idx)


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
    flow_eval_steps=1,
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
        out_gs = integrate_flow(model, input_gs_device, scene_idx, num_steps=flow_eval_steps)

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
            pred_grid_rgb = make_grid(pred_preview)
            pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid_rgb = make_grid(gt_preview)
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
                compare_grid = make_grid(compare_preview)
                wandb_images[f"eval_images/{scene_name}/compare_grid"] = wandb.Image(
                    compare_grid,
                    caption=f"{scene_name} GT | input | pred",
                )
            if wandb_images:
                _wandb_log(wandb_images, step=wandb_step)

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
            residual_type = "flow_integrated_out_minus_input"
            residual_keys = [key for key in predicted_keys if key in out_gs and key in input_gs]
            if len(residual_keys) == 0:
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
                "residual_type": residual_type,
                "residual_keys": residual_keys,
                "residuals": _to_cpu(residuals),
                "input_gs": _to_cpu(input_gs_device),
                "output_gs": _to_cpu(out_gs),
                "cameras": _to_cpu(eval_cameras),
            }
            torch.save(pt_payload, os.path.join(residual_dir, f"{scene_stem}.pt"))

            stats_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "num_gaussians": int(input_gs["means"].shape[0]),
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
    flow_eval_steps=1,
):
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    all_metrics_input = []
    logger = ProcessSafeLogger(os.path.join(output_dir, "eval.log")).get_logger()

    for scene_idx in tqdm(range(len(dataset.folders)), desc="Evaluating"):
        scene = dataset.load_scene(scene_idx)
        input_factor_entry = scene["factor_data"][INPUT_FACTOR]
        target_factor_entry = scene["factor_data"][TARGET_FACTOR]
        eval_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test"
        )

        scene_output_dir = os.path.join(output_dir, scene["scene_name"])
        metrics, metrics_input = evaluate_single_scene(
            model=model,
            input_gs=input_factor_entry["gs_params"],
            gt_gs=target_factor_entry["gs_params"],
            scene_idx=scene["idx"],
            scene_name=scene["scene_name"],
            eval_images=eval_payload["images"],
            eval_cameras=eval_payload["cameras"],
            image_names=eval_payload["images_name"],
            output_dir=scene_output_dir,
            eval_chunk_size=len(eval_payload["images"]),
            compare_with_input=compare_with_input,
            save_viewer=save_viewer,
            save_residuals=save_residuals,
            output_gt=output_gt,
            wandb_step=wandb_step,
            flow_eval_steps=flow_eval_steps,
        )
        all_metrics.append(metrics)
        if compare_with_input:
            all_metrics_input.append(metrics_input)
        logger.info(
            f"Scene {scene['scene_name']}: "
            + " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        )

    reduced_metrics = {}
    if len(all_metrics) > 0:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            reduced_metrics[key] = float(np.mean([metrics[key] for metrics in all_metrics]))

    reduced_metrics_input = {}
    if compare_with_input and len(all_metrics_input) > 0:
        metric_keys = all_metrics_input[0].keys()
        for key in metric_keys:
            reduced_metrics_input[key] = float(np.mean([metrics[key] for metrics in all_metrics_input]))

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
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger()
    wandb_run = _init_wandb(FLAGS.output_dir)
    device = torch.device("cuda")

    train_dataset = _build_dataset("train_dataset", FLAGS.output_dir)
    test_dataset = _build_dataset("test_dataset", FLAGS.output_dir)
    train_sampler = SceneSampler(train_dataset)

    model = GSFlowFeaturePredictor().to(device)
    if getattr(model.backbone, "T_dim", -1) == -1:
        raise ValueError(
            "Flow scripts require timestep conditioning. Set PointTransformerV3FlowModel.T_dim, "
            "for example PointTransformerV3FlowModel.T_dim = 128 in gin."
        )
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
        logger.info(f"Loaded model checkpoint from {model.resume_ckpt}")

    if FLAGS.only_eval:
        model.eval()
    else:
        model.train()

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
    flow_loss_weight = train_cfg["flow_loss_weight"]
    flow_loss_type = train_cfg["flow_loss_type"]
    flow_t_min = train_cfg["flow_t_min"]
    flow_t_max = train_cfg["flow_t_max"]
    flow_eval_steps = train_cfg["flow_eval_steps"]
    flow_trajectory_mode = train_cfg["flow_trajectory_mode"]
    flow_optim_steps = train_cfg["flow_optim_steps"]
    flow_optim_snapshot_interval = train_cfg["flow_optim_snapshot_interval"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    if flow_trajectory_mode not in ["linear", "optimized_prefix"]:
        raise ValueError(f"Unsupported flow_trajectory_mode: {flow_trajectory_mode}")

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    if not FLAGS.only_eval:
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(range(resume_from_step, total_steps), desc="Training")
        for step in pbar:
            scene = train_sampler.next_scene()
            input_factor_entry = scene["factor_data"][INPUT_FACTOR]
            target_factor_entry = scene["factor_data"][TARGET_FACTOR]
            train_payload = _build_split_payload(
                train_dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
            )

            batch_gs = gpu_utils.move_to_device([input_factor_entry["gs_params"]], device)
            target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
            flow_keys = _flow_float_keys(model, batch_gs[0], target_gs)
            batch_scene_idx = [scene["idx"]]
            batch_cameras = gpu_utils.move_to_device([train_payload["cameras"]], device)
            batch_images = gpu_utils.move_to_device([train_payload["images"]], device)

            optimized_trajectory = None
            if flow_trajectory_mode == "optimized_prefix":
                optimized_trajectory = build_optimized_gs_trajectory(
                    input_gs=batch_gs[0],
                    flow_keys=flow_keys,
                    images=batch_images[0],
                    cameras=batch_cameras[0],
                    optim_steps=flow_optim_steps,
                    snapshot_interval=flow_optim_snapshot_interval,
                    logger=logger,
                )

            with torch.cuda.amp.autocast(enabled=enable_amp):
                if flow_trajectory_mode == "optimized_prefix":
                    xt_gs, flow_target_gs, t, time_delta = sample_optimized_prefix_pair(
                        optimized_trajectory, flow_optim_steps, device
                    )
                    flow_start_gs = optimized_trajectory[0][1]
                else:
                    t = _sample_flow_t(device, flow_t_min, flow_t_max)
                    xt_gs = _interpolate_gs(batch_gs[0], target_gs, t, flow_keys)
                    flow_target_gs = target_gs
                    flow_start_gs = batch_gs[0]
                    time_delta = torch.ones((), device=device)
                out_batch_gs = model(
                    batch_normalized_gs=[xt_gs],
                    batch_scene_idx=batch_scene_idx,
                    timestep=t,
                )
                out_gs = out_batch_gs[0]
                flow_loss, flow_feature_losses = _flow_velocity_loss(
                    out_gs,
                    xt_gs,
                    flow_start_gs,
                    flow_target_gs,
                    flow_keys,
                    loss_type=flow_loss_type,
                    time_delta=time_delta,
                )
                total_loss = flow_loss * flow_loss_weight

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

            pbar.set_postfix(
                {
                    "scene": scene["scene_name"],
                    "loss": f"{total_loss.item():.4f}",
                    "flow": f"{flow_loss.item():.4f}",
                    "t": f"{t.item():.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
                torch.cuda.empty_cache()

            if step % log_interval == 0:
                train_log = {
                    "train/total_loss": total_loss.item(),
                    "train/flow_loss": flow_loss.item(),
                    "train/t": t.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/scene_idx": scene["idx"],
                }
                for key, value in flow_feature_losses.items():
                    train_log[f"train/flow_{key}"] = value.item()
                _wandb_log(train_log, step=step)

                log_msg = (
                    f"step={step} scene={scene['scene_name']} total={total_loss.item():.6f} "
                    f"flow={flow_loss.item():.6f} t={t.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}"
                )
                logger.info(log_msg)

            if step % log_image_interval == 0:
                with torch.no_grad():
                    preview_gs = integrate_flow(model, batch_gs[0], scene["idx"], num_steps=flow_eval_steps)
                    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(preview_gs, batch_cameras[0])
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
                pred_grid_rgb = make_grid(pred_imgs_uint8)
                pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
                pred_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png")
                cv2.imwrite(pred_path, pred_grid)

                gt_imgs_uint8 = [
                    (img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in batch_images[0]
                ]
                gt_grid_rgb = make_grid(gt_imgs_uint8)
                gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
                gt_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_gt.png")
                cv2.imwrite(gt_path, gt_grid)

                if wandb is not None and wandb.run is not None:
                    _wandb_log(
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
                    flow_eval_steps=flow_eval_steps,
                )
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Eval step {step}: {metric_str}")
                _wandb_log({f"eval/{key}": value for key, value in metrics.items()}, step=step)
                if FLAGS.compare_with_input:
                    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                    logger.info(f"Eval step {step} input: {metric_str}")
                    _wandb_log(
                        {f"eval_input/{key}": value for key, value in metrics_input.items()},
                        step=step,
                    )
                model.train()

            if (step + 1) % save_interval == 0:
                ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved model checkpoint to {ckpt_path}")
                # if wandb is not None and wandb.run is not None:
                #     wandb.save(ckpt_path, base_path=FLAGS.output_dir)

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = evaluate_dataset(
        model=model,
        dataset=test_dataset,
        output_dir=final_eval_dir,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
        flow_eval_steps=flow_eval_steps,
    )
    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    _wandb_log({f"final_eval/{key}": value for key, value in metrics.items()})
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")
        _wandb_log({f"final_eval_input/{key}": value for key, value in metrics_input.items()})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)

import json
import os
import random

import cv2
import gin
import numpy as np
import torch
from absl import app, flags

# Register gin configurables that may appear in shared config files.
from models.feature_predictor import FeaturePredictor  # noqa: F401

from dataset.GS_multi import SplatFactoMultiLevelDataset
from gs_flow_model import GSFlowModel
from gs_path import GSPath
from utils import gpu_utils, gs_utils
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler  # noqa: F401


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("diagnostic_subdir", "test_flow", "Diagnostic output subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to diagnose")
flags.DEFINE_string("ckpt", None, "Optional trained flow checkpoint")
flags.DEFINE_integer("diagnostic_timesteps", -1, "Timestep index to diagnose; -1 means all")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

INPUT_FACTOR = 4
TARGET_FACTOR = 2


@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@gin.configurable("training")
def training_config(
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
    flow_eval_steps=10,
    flow_trajectory_mode="optimized_target_discrete",
    flow_num_timesteps=10,
    flow_segment_steps=10,
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
        "flow_loss_weight": flow_loss_weight,
        "flow_loss_type": flow_loss_type,
        "flow_eval_steps": flow_eval_steps,
        "flow_trajectory_mode": flow_trajectory_mode,
        "flow_num_timesteps": flow_num_timesteps,
        "flow_segment_steps": flow_segment_steps,
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


def flow_velocity_loss(pred_velocity, start_gs, target_gs, flow_keys, loss_type="mse", time_delta=1.0):
    losses = {}
    total = None
    for key in flow_keys:
        target_velocity = (target_gs[key] - start_gs[key]) / time_delta
        if loss_type == "l1":
            loss = (pred_velocity[key] - target_velocity).abs().mean()
        elif loss_type == "mse":
            loss = torch.nn.functional.mse_loss(pred_velocity[key], target_velocity)
        else:
            raise ValueError(f"Unsupported flow_loss_type: {loss_type}")
        losses[key] = loss
        total = loss if total is None else total + loss
    return total / len(flow_keys), losses


def _build_split_payload(dataset, scene_idx, scene_name, factor_entry, split):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    total_num = len(meta["camera_to_worlds"])
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


def _selected_timestep_indices(num_timesteps):
    if FLAGS.diagnostic_timesteps == -1:
        return list(range(num_timesteps))
    if FLAGS.diagnostic_timesteps < 0 or FLAGS.diagnostic_timesteps >= num_timesteps:
        raise ValueError(
            f"diagnostic_timesteps must be -1 or in [0, {num_timesteps - 1}], got {FLAGS.diagnostic_timesteps}"
        )
    return [FLAGS.diagnostic_timesteps]


def _tensor_stats(tensor):
    flat = tensor.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "norm": float(flat.norm().item()),
        "max_abs": float(flat.abs().max().item()),
    }


def _gs_pair_stats(gs_a, gs_b, flow_keys):
    stats = {}
    for key in flow_keys:
        if key not in gs_a or key not in gs_b:
            continue
        diff = gs_b[key] - gs_a[key]
        stats[key] = {
            "mse": float(torch.nn.functional.mse_loss(gs_a[key], gs_b[key]).item()),
            "l1": float(diff.abs().mean().item()),
            **{f"diff_{k}": v for k, v in _tensor_stats(diff).items()},
        }
    return stats


def _velocity_stats(pred_velocity, start_gs, target_gs, flow_keys, dt):
    stats = {}
    for key in flow_keys:
        target_velocity = (target_gs[key] - start_gs[key]) / dt
        pred = pred_velocity[key].detach().float().reshape(-1)
        target = target_velocity.detach().float().reshape(-1)
        mse = torch.nn.functional.mse_loss(pred, target)
        l1 = (pred - target).abs().mean()
        pred_norm = pred.norm()
        target_norm = target.norm()
        cosine = torch.nn.functional.cosine_similarity(pred, target, dim=0, eps=1e-12)
        stats[key] = {
            "mse": float(mse.item()),
            "l1": float(l1.item()),
            "pred_norm": float(pred_norm.item()),
            "target_norm": float(target_norm.item()),
            "norm_ratio": float((pred_norm / target_norm.clamp_min(1e-12)).item()),
            "cosine": float(cosine.item()),
        }
    return stats


def _mean_feature_stats(per_timestep):
    accum = {}
    counts = {}
    for entry in per_timestep:
        for feature, values in entry["feature_stats"].items():
            accum.setdefault(feature, {})
            counts.setdefault(feature, {})
            for key, value in values.items():
                accum[feature][key] = accum[feature].get(key, 0.0) + float(value)
                counts[feature][key] = counts[feature].get(key, 0) + 1
    return {
        feature: {key: accum[feature][key] / counts[feature][key] for key in accum[feature]}
        for feature in accum
    }


def _apply_velocity(gs, velocity, dt, flow_keys):
    out = dict(gs)
    for key in flow_keys:
        if key in gs and key in velocity and torch.is_tensor(gs[key]):
            out[key] = gs[key] + dt * velocity[key]
    return out


def _apply_oracle_delta(current_gs, teacher_start_gs, teacher_target_gs, flow_keys):
    out = dict(current_gs)
    for key in flow_keys:
        if key in current_gs and key in teacher_start_gs and key in teacher_target_gs:
            out[key] = current_gs[key] + (teacher_target_gs[key] - teacher_start_gs[key])
    return out


def _to_uint8_rgb(imgs):
    out = []
    for img in imgs:
        if img.shape[-1] == 4:
            img = img[..., :3]
        out.append((img * 255).detach().cpu().numpy().astype(np.uint8))
    return out


def _write_preview(imgs, output_dir, name):
    if len(imgs) == 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    grid = cv2.cvtColor(make_grid(imgs[:9]), cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, grid)
    return path


def evaluate_gs_against_images(gs_params, images, cameras, output_dir, chunk_size=None, preview_name="preview.png"):
    device = gs_params["means"].device
    os.makedirs(output_dir, exist_ok=True)
    num_views = len(images)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_views
    chunk_size = min(chunk_size, num_views)

    metric_computer = MetricComputer()
    preview = []
    with torch.no_grad():
        for start in range(0, num_views, chunk_size):
            end = min(start + chunk_size, num_views)
            chunk_images = gpu_utils.move_to_device(images[start:end], device)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)

            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_params, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            gt_imgs = torch.stack(chunk_images, dim=0)
            if gt_imgs.shape[-1] == 4:
                masks = gt_imgs[..., 3].unsqueeze(-1)
                pred_imgs = pred_imgs * masks
                gt_imgs = gt_imgs[..., :3]
            metric_computer.update(pred_imgs, gt_imgs, name=f"{start:06d}")

            slots = 9 - len(preview)
            if slots > 0:
                preview.extend(_to_uint8_rgb(pred_imgs[:slots]))

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics_per_view.json"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _write_preview(preview, output_dir, preview_name)
    return metrics


def compare_gs_renders(pred_gs, target_gs, cameras, output_dir, chunk_size=None, preview_name="preview.png"):
    device = pred_gs["means"].device
    os.makedirs(output_dir, exist_ok=True)
    num_views = int(cameras["camera_to_worlds"].shape[0])
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_views
    chunk_size = min(chunk_size, num_views)

    metric_computer = MetricComputer()
    preview = []
    with torch.no_grad():
        for start in range(0, num_views, chunk_size):
            end = min(start + chunk_size, num_views)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(pred_gs, chunk_cameras)
            target_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(target_gs, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            target_imgs = torch.stack(target_imgs, dim=0)
            metric_computer.update(pred_imgs, target_imgs, name=f"{start:06d}")

            slots = 9 - len(preview)
            if slots > 0:
                preview.extend(_to_uint8_rgb(pred_imgs[:slots]))

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics_per_view.json"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _write_preview(preview, output_dir, preview_name)
    return metrics


def teacher_final_gs(gs_path, path_data_dict, device):
    num_timesteps = gs_path.flow_num_timesteps
    if gs_path.path_mode == GSPath.MODE_OPTIMIZED_TARGET_DISCRETE:
        t = torch.zeros(1, device=device)
        gs_path.sample_path(path_data_dict, t)
        return path_data_dict["optimized_target_gs"]

    t = torch.tensor([float(num_timesteps - 1) / float(num_timesteps)], device=device)
    _, final_gs = gs_path.sample_path(path_data_dict, t)
    return final_gs


def oracle_rollout(gs_path, path_data_dict, input_gs, flow_keys, device):
    current_gs = input_gs
    mismatches = []
    for idx in range(gs_path.flow_num_timesteps):
        t = torch.tensor([float(idx) / float(gs_path.flow_num_timesteps)], device=device)
        teacher_start_gs, teacher_target_gs = gs_path.sample_path(path_data_dict, t)
        mismatches.append(
            {
                "timestep_idx": idx,
                "sample_t": float(t.item()),
                "start_state_mismatch": _gs_pair_stats(current_gs, teacher_start_gs, flow_keys),
            }
        )
        current_gs = _apply_oracle_delta(current_gs, teacher_start_gs, teacher_target_gs, flow_keys)
    return current_gs, mismatches


def learned_one_step_diagnostics(
    model,
    gs_path,
    path_data_dict,
    flow_keys,
    train_cameras,
    output_dir,
    timestep_indices,
    device,
):
    dt = torch.tensor(1.0 / float(gs_path.flow_num_timesteps), device=device)
    per_timestep = []
    for idx in timestep_indices:
        t = torch.tensor([float(idx) / float(gs_path.flow_num_timesteps)], device=device)
        start_gs, target_gs = gs_path.sample_path(path_data_dict, t)
        with torch.no_grad():
            pred_velocity = model(batch_normalized_gs=[start_gs], timestep=t)[0]
            pred_next_gs = _apply_velocity(start_gs, pred_velocity, dt, flow_keys)
        feature_stats = _velocity_stats(pred_velocity, start_gs, target_gs, flow_keys, dt)
        image_metrics = compare_gs_renders(
            pred_next_gs,
            target_gs,
            train_cameras,
            os.path.join(output_dir, f"timestep_{idx:03d}"),
            preview_name="pred_next_vs_teacher.png",
        )
        per_timestep.append(
            {
                "timestep_idx": idx,
                "sample_t": float(t.item()),
                "feature_stats": feature_stats,
                "image_metrics": image_metrics,
            }
        )
    return {"per_timestep": per_timestep, "feature_means": _mean_feature_stats(per_timestep)}


def learned_rollout(model, input_gs, flow_keys, num_timesteps, mode):
    if mode not in ["left", "midpoint"]:
        raise ValueError(f"Unsupported rollout mode: {mode}")
    current_gs = input_gs
    device = input_gs["means"].device
    dt = 1.0 / float(num_timesteps)
    for idx in range(num_timesteps):
        if mode == "left":
            t_value = float(idx) * dt
        else:
            t_value = (float(idx) + 0.5) * dt
        t = torch.tensor([t_value], device=device)
        with torch.no_grad():
            pred_velocity = model(batch_normalized_gs=[current_gs], timestep=t)[0]
        current_gs = _apply_velocity(current_gs, pred_velocity, dt, flow_keys)
    return current_gs


def load_scene_context(device):
    dataset = None
    with gin.config_scope("train_dataset"):
        dataset = SplatFactoMultiLevelDataset()

    scene_idx = 0
    if FLAGS.scene_name != "":
        for idx in range(len(dataset.folders)):
            if dataset.folders[idx]["scene_name"] == FLAGS.scene_name:
                scene_idx = idx
                break

    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][INPUT_FACTOR]
    target_factor_entry = scene["factor_data"][TARGET_FACTOR]

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

    input_gs = gpu_utils.move_to_device(input_factor_entry["gs_params"], device)
    target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
    train_cameras = gpu_utils.move_to_device(train_payload["cameras"], device)
    train_images = gpu_utils.move_to_device(train_payload["images"], device)

    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_payload["images"])
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_payload["images"])

    return {
        "dataset": dataset,
        "scene": scene,
        "input_gs": input_gs,
        "target_gs": target_gs,
        "train_payload": train_payload,
        "train_images": train_images,
        "train_cameras": train_cameras,
        "eval_payload": eval_payload,
        "eval_images": eval_payload["images"],
        "eval_cameras": eval_payload["cameras"],
        "eval_chunk_size": eval_chunk_size,
    }


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    diagnostic_dir = os.path.join(FLAGS.output_dir, FLAGS.diagnostic_subdir)
    os.makedirs(diagnostic_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training_config(output_dir=FLAGS.output_dir)
    set_seed()

    device = torch.device("cuda")
    context = load_scene_context(device)

    model = GSFlowModel().to(device)
    ckpt = FLAGS.ckpt or model.resume_ckpt
    learned_enabled = ckpt is not None
    if learned_enabled:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    gs_path = GSPath(
        path_mode=train_cfg["flow_trajectory_mode"],
        flow_num_timesteps=train_cfg["flow_num_timesteps"],
        flow_segment_steps=train_cfg["flow_segment_steps"],
    )
    flow_keys = gs_path.flow_float_keys(context["input_gs"], context["target_gs"], model.output_features)
    path_data_dict = {
        "input_gs": context["input_gs"],
        "target_gs": context["target_gs"],
        "images": context["train_images"],
        "cameras": context["train_cameras"],
        "flow_keys": flow_keys,
    }

    summary = {
        "scene_idx": int(context["scene"]["idx"]),
        "scene_name": context["scene"]["scene_name"],
        "path_mode": gs_path.path_mode,
        "flow_num_timesteps": gs_path.flow_num_timesteps,
        "flow_segment_steps": gs_path.flow_segment_steps,
        "checkpoint": ckpt,
        "learned_enabled": learned_enabled,
        "flow_keys": flow_keys,
    }

    teacher_gs = teacher_final_gs(gs_path, path_data_dict, device)
    summary["teacher_final"] = {
        "input_metrics": evaluate_gs_against_images(
            context["input_gs"],
            context["eval_images"],
            context["eval_cameras"],
            os.path.join(diagnostic_dir, "teacher_final", "input"),
            chunk_size=context["eval_chunk_size"],
            preview_name="input.png",
        ),
        "teacher_metrics": evaluate_gs_against_images(
            teacher_gs,
            context["eval_images"],
            context["eval_cameras"],
            os.path.join(diagnostic_dir, "teacher_final", "teacher"),
            chunk_size=context["eval_chunk_size"],
            preview_name="teacher.png",
        ),
        "target_gs_metrics": evaluate_gs_against_images(
            context["target_gs"],
            context["eval_images"],
            context["eval_cameras"],
            os.path.join(diagnostic_dir, "teacher_final", "target_gs"),
            chunk_size=context["eval_chunk_size"],
            preview_name="target_gs.png",
        ),
        "gs_loss": None if path_data_dict.get("last_gs_loss") is None else float(path_data_dict["last_gs_loss"].item()),
    }

    oracle_gs, oracle_mismatches = oracle_rollout(
        gs_path,
        path_data_dict,
        context["input_gs"],
        flow_keys,
        device,
    )
    summary["oracle_rollout"] = {
        "metrics": evaluate_gs_against_images(
            oracle_gs,
            context["eval_images"],
            context["eval_cameras"],
            os.path.join(diagnostic_dir, "oracle_rollout"),
            chunk_size=context["eval_chunk_size"],
            preview_name="oracle_rollout.png",
        ),
        "final_vs_teacher_gs": _gs_pair_stats(oracle_gs, teacher_gs, flow_keys),
        "start_state_mismatches": oracle_mismatches,
    }

    if learned_enabled:
        timestep_indices = _selected_timestep_indices(gs_path.flow_num_timesteps)
        summary["learned_one_step"] = learned_one_step_diagnostics(
            model,
            gs_path,
            path_data_dict,
            flow_keys,
            context["train_cameras"],
            os.path.join(diagnostic_dir, "learned_one_step"),
            timestep_indices,
            device,
        )
        for mode in ["left", "midpoint"]:
            rollout_gs = learned_rollout(
                model,
                context["input_gs"],
                flow_keys,
                gs_path.flow_num_timesteps,
                mode=mode,
            )
            summary[f"learned_rollout_{mode}"] = {
                "metrics": evaluate_gs_against_images(
                    rollout_gs,
                    context["eval_images"],
                    context["eval_cameras"],
                    os.path.join(diagnostic_dir, f"learned_rollout_{mode}"),
                    chunk_size=context["eval_chunk_size"],
                    preview_name=f"learned_rollout_{mode}.png",
                ),
                "final_vs_teacher_gs": _gs_pair_stats(rollout_gs, teacher_gs, flow_keys),
            }
    else:
        summary["learned_one_step"] = None
        summary["learned_rollout_left"] = None
        summary["learned_rollout_midpoint"] = None

    with open(os.path.join(diagnostic_dir, "summary.json"), "w") as f:
        json.dump(to_cpu(summary), f, indent=2)
    print(json.dumps(summary, indent=2))


app.run(main)

import json
import os
import random

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

# Register configurables referenced by the shared model and x1 Gin files.
from models.feature_predictor import FeaturePredictor  # noqa: F401

from dataset.GS_multi import SplatFactoMultiLevelDataset
from gs_flow_model import GSFlowModel  # noqa: F401
from gs_path import GSPath
from utils import gpu_utils, gs_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer
from utils.optimizers import build_3DGSoptimizer, build_optimizer, build_scheduler  # noqa: F401


flags.DEFINE_string("output_dir", "output_gs_interpolation", "Output directory")
flags.DEFINE_string("scene_name", "", "Scene name to validate")
flags.DEFINE_integer("optimization_steps", 1000, "Number of endpoint GS optimization steps")
flags.DEFINE_integer("num_interpolations", 10, "Number of interpolation states in (0, 1]")
flags.DEFINE_integer("render_chunk_size", 16, "Maximum views rendered per evaluation chunk")
flags.DEFINE_integer("training_video_view_index", 0, "Fixed training view recorded during optimization")
flags.DEFINE_float("training_video_fps", 30.0, "Endpoint optimization video frame rate")
flags.DEFINE_multi_string("gin_file", None, "List of Gin configuration files")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter overrides")

FLAGS = flags.FLAGS

INPUT_FACTOR = 4
TARGET_FACTOR = 2
REQUIRED_GS_KEYS = (
    "means",
    "scales",
    "opacities",
    "quats",
    "features_dc",
    "features_rest",
)


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
    x1_loss_mode="point_mse",
    x1_prediction_type="residual",
    x1_optimization_steps=10000,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
):
    del (
        output_dir,
        total_steps,
        pretrain_steps,
        eval_interval,
        log_interval,
        save_interval,
        log_image_interval,
        grad_clip_norm,
        x1_loss_mode,
        x1_prediction_type,
        x1_optimization_steps,
        resume_from_step,
        enable_amp,
        empty_cache_fre,
    )


def make_grid(images, nrow=3, ncols=3):
    if len(images) == 0:
        raise ValueError("Cannot create a grid from zero images")
    image_h, image_w = images[0].shape[:2]
    grid = np.zeros((image_h * nrow, image_w * ncols, 3), dtype=np.uint8)
    for index, image in enumerate(images[: nrow * ncols]):
        row, col = divmod(index, ncols)
        grid[row * image_h : (row + 1) * image_h, col * image_w : (col + 1) * image_w] = image
    return grid


def build_payload(dataset, scene_idx, scene_name, factor_entry):
    meta = factor_entry["meta"]
    if dataset.background_color == "random":
        raise ValueError("Interpolation validation requires a deterministic background color")
    background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0
    num_views = len(meta["camera_to_worlds"])
    if num_views == 0:
        raise ValueError("Interpolation validation received zero views")

    images = [dataset.read_image(path, background=background) for path in factor_entry["imgs_path"]]
    cameras = {
        "camera_to_worlds": torch.as_tensor(meta["camera_to_worlds"]).float(),
        "fx": torch.as_tensor(meta["fx"]).float(),
        "fy": torch.as_tensor(meta["fy"]).float(),
        "cx": torch.as_tensor(meta["cx"]).float(),
        "cy": torch.as_tensor(meta["cy"]).float(),
        "width": torch.as_tensor(meta["width"]).float(),
        "height": torch.as_tensor(meta["height"]).float(),
        "background_color": background,
    }
    return {
        "images": images,
        "images_name": list(factor_entry["imgs_name"]),
        "cameras": cameras,
        "scene_idx": int(scene_idx),
        "scene_name": scene_name,
    }


def floating_gs_keys(gs):
    missing = [key for key in REQUIRED_GS_KEYS if key not in gs]
    if missing:
        raise KeyError(f"Source GS is missing required interpolation parameters: {missing}")
    keys = [
        key
        for key, value in gs.items()
        if torch.is_tensor(value) and value.is_floating_point()
    ]
    if len(keys) == 0:
        raise ValueError("Source GS has no floating parameters to optimize and interpolate")
    return keys


def slice_camera(cameras, view_index):
    return {
        key: (value[view_index : view_index + 1] if key == "camera_to_worlds" else value)
        for key, value in cameras.items()
    }


def masked_l1(pred_image, gt_image):
    gt_rgb = gt_image[..., :3]
    if gt_image.shape[-1] == 4:
        mask = gt_image[..., 3:].to(pred_image.dtype)
        return ((pred_image - gt_rgb) * mask).abs().mean()
    return (pred_image - gt_rgb).abs().mean()


def render_video_frame(gs, gt_image, camera, iteration):
    with torch.no_grad():
        rendered, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, camera)
        if len(rendered) != 1:
            raise RuntimeError(f"Expected one video render, received {len(rendered)}")
        pred_image = rendered[0]
        view_l1 = masked_l1(pred_image, gt_image)
        if not torch.isfinite(pred_image).all():
            raise FloatingPointError(f"Video render became non-finite at iteration {iteration}")

    frame_rgb = (pred_image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
    label = f"iteration={iteration}  view_l1={view_l1.item():.6f}"
    cv2.putText(frame_bgr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr, float(view_l1.item())


class TrainingVideoRecorder:
    def __init__(self, path, fps, first_frame, writer_factory=cv2.VideoWriter):
        if fps <= 0:
            raise ValueError("training_video_fps must be positive")
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = writer_factory(path, fourcc, float(fps), (int(width), int(height)))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Could not open MP4 video writer for: {path}")
        self.frame_size = (height, width)
        self.closed = False

    def write(self, frame):
        if self.closed:
            raise RuntimeError("Cannot write to a closed training video")
        if frame.shape[:2] != self.frame_size:
            raise ValueError(
                f"Video frame size changed from {self.frame_size} to {frame.shape[:2]}"
            )
        self.writer.write(frame)

    def close(self):
        if not self.closed:
            self.writer.release()
            self.closed = True


def optimize_endpoint_with_video(
    x0,
    images,
    cameras,
    float_keys,
    optimization_steps,
    video_view_index,
    video_fps,
    video_path,
    logger,
    recorder_factory=TrainingVideoRecorder,
):
    if optimization_steps <= 0:
        raise ValueError("optimization_steps must be positive")
    if video_view_index < 0 or video_view_index >= len(images):
        raise IndexError(
            f"training_video_view_index={video_view_index} is outside [0, {len(images) - 1}]"
        )

    state, trainable = GSPath.clone_trainable_gs(x0, float_keys)
    if len(trainable) == 0:
        raise ValueError("No trainable GS parameters were selected")
    with gin.config_scope("flow_optim"):
        optimizer = build_3DGSoptimizer(trainable)

    video_camera = slice_camera(cameras, video_view_index)
    video_gt = images[video_view_index]
    first_frame, first_view_l1 = render_video_frame(state, video_gt, video_camera, 0)
    recorder = recorder_factory(video_path, video_fps, first_frame)
    history = [
        {
            "iteration": 0,
            "optimization_loss_before_update": None,
            "video_view_l1_after_update": first_view_l1,
        }
    ]

    try:
        recorder.write(first_frame)
        progress = tqdm(range(1, optimization_steps + 1), desc="Optimize endpoint GS")
        for iteration in progress:
            optimizer.zero_grad(set_to_none=True)
            optimization_loss = GSPath.render_l1_loss(state, images, cameras)
            if not torch.isfinite(optimization_loss):
                raise FloatingPointError(
                    f"Endpoint optimization loss became non-finite at iteration {iteration}"
                )
            optimization_loss.backward()
            optimizer.step()

            frame, view_l1 = render_video_frame(
                state,
                video_gt,
                video_camera,
                iteration,
            )
            recorder.write(frame)
            record = {
                "iteration": int(iteration),
                "optimization_loss_before_update": float(optimization_loss.detach().item()),
                "video_view_l1_after_update": view_l1,
            }
            history.append(record)
            progress.set_postfix(loss=f"{record['optimization_loss_before_update']:.6f}", view_l1=f"{view_l1:.6f}")
            if iteration == 1 or iteration == optimization_steps or iteration % max(optimization_steps // 10, 1) == 0:
                logger.info(
                    f"endpoint iteration={iteration}/{optimization_steps} "
                    f"optimization_l1={record['optimization_loss_before_update']:.6f} "
                    f"video_view_l1={view_l1:.6f}"
                )
    finally:
        recorder.close()

    return GSPath.detach_gs(state), history


def tensor_stats(tensor):
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    if not finite.all():
        return {
            "shape": list(value.shape),
            "finite": False,
            "nonfinite_count": int((~finite).sum().item()),
        }
    return {
        "shape": list(value.shape),
        "finite": True,
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def interpolation_stats(gs, float_keys):
    stats = {key: tensor_stats(gs[key]) for key in float_keys}
    if not all(entry["finite"] for entry in stats.values()):
        bad_keys = [key for key, entry in stats.items() if not entry["finite"]]
        raise FloatingPointError(f"Interpolated GS contains non-finite parameters: {bad_keys}")

    scales = torch.exp(gs["scales"].detach().float())
    opacities = torch.sigmoid(gs["opacities"].detach().float())
    quat_norms = torch.linalg.vector_norm(gs["quats"].detach().float(), dim=-1)
    activated = {
        "scales_exp": tensor_stats(scales),
        "opacities_sigmoid": tensor_stats(opacities),
        "quaternion_norm": tensor_stats(quat_norms),
        "near_zero_quaternion_count": int((quat_norms < 1e-8).sum().item()),
    }
    return {"parameters": stats, "activated": activated}


def interpolation_state(x0, x1, t, float_keys):
    if x0["means"].shape[0] != x1["means"].shape[0]:
        raise ValueError(
            f"GS topology mismatch: x0 has {x0['means'].shape[0]} splats, "
            f"x1 has {x1['means'].shape[0]}"
        )
    for key in float_keys:
        if key not in x1:
            raise KeyError(f"Optimized endpoint is missing parameter '{key}'")
        if x0[key].shape != x1[key].shape:
            raise ValueError(
                f"Interpolation shape mismatch for '{key}': "
                f"x0={tuple(x0[key].shape)} x1={tuple(x1[key].shape)}"
            )

    interpolated = GSPath.interpolate_gs(x0, x1, t, float_keys)
    t_value = torch.as_tensor(t, device=x0["means"].device, dtype=torch.float32).reshape(())
    for key in float_keys:
        expected = x0[key] + t_value * (x1[key] - x0[key])
        if not torch.allclose(interpolated[key], expected, rtol=1e-6, atol=1e-7):
            raise AssertionError(f"Interpolation formula validation failed for '{key}' at t={float(t_value)}")
    return interpolated


def t_label(t):
    label = f"{float(t):.4f}".rstrip("0").rstrip(".")
    if "." not in label:
        label += ".0"
    return label


def to_uint8_rgb(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    return (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()


def save_ground_truth_preview(images, output_dir):
    preview = [to_uint8_rgb(image) for image in images[:9]]
    grid = make_grid(preview)
    cv2.imwrite(os.path.join(output_dir, "ground_truth_preview.png"), grid[..., ::-1])


def evaluate_interpolation(
    gs,
    images,
    image_names,
    cameras,
    output_dir,
    render_chunk_size,
):
    if len(images) == 0:
        raise ValueError("Cannot evaluate interpolation with zero images")
    if len(images) != len(image_names):
        raise ValueError("Evaluation image tensors and names have different lengths")
    chunk_size = min(max(int(render_chunk_size), 1), 16, len(images))
    views_dir = os.path.join(output_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    metric_computer = MetricComputer()
    per_view_metrics = {}
    preview = []
    l1_values = []
    with torch.no_grad():
        for start in range(0, len(images), chunk_size):
            end = min(start + chunk_size, len(images))
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in cameras.items()
            }
            pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, chunk_cameras)
            if len(pred_images) != end - start:
                raise RuntimeError(
                    f"Rasterizer returned {len(pred_images)} images for a chunk of {end - start}"
                )
            pred_batch = torch.stack(pred_images, dim=0)
            gt_batch = torch.stack(images[start:end], dim=0)
            if not torch.isfinite(pred_batch).all():
                raise FloatingPointError(f"Non-finite render in views [{start}, {end})")

            if gt_batch.shape[-1] == 4:
                masks = gt_batch[..., 3:].to(pred_batch.dtype)
                metric_pred = pred_batch * masks
                metric_gt = gt_batch[..., :3]
                chunk_l1 = ((pred_batch - metric_gt) * masks).abs().flatten(1).mean(1)
            else:
                metric_pred = pred_batch
                metric_gt = gt_batch[..., :3]
                chunk_l1 = (metric_pred - metric_gt).abs().flatten(1).mean(1)

            chunk_name = f"views_{start:06d}_{end:06d}"
            metric_computer.update(metric_pred, metric_gt, name=chunk_name)
            chunk_metrics = metric_computer.results_dict[chunk_name]
            for local_index, image_name in enumerate(image_names[start:end]):
                safe_name = os.path.basename(image_name)
                pred_uint8 = to_uint8_rgb(metric_pred[local_index])
                cv2.imwrite(os.path.join(views_dir, safe_name), pred_uint8[..., ::-1])
                if len(preview) < 9:
                    preview.append(pred_uint8)
                per_view_metrics[safe_name] = {
                    key: float(values[local_index]) for key, values in chunk_metrics.items()
                }
                per_view_metrics[safe_name]["render_l1"] = float(chunk_l1[local_index].item())
            l1_values.append(chunk_l1)

    metrics = metric_computer.finalize()
    metrics["render_l1"] = float(torch.cat(l1_values).mean().item())
    with open(os.path.join(output_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)
    with open(os.path.join(output_dir, "metrics_per_view.json"), "w") as file:
        json.dump(per_view_metrics, file, indent=2)
    preview_grid = make_grid(preview)
    cv2.imwrite(os.path.join(output_dir, "preview.png"), preview_grid[..., ::-1])
    del metric_computer
    return metrics


def find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for index, entry in enumerate(dataset.folders):
        if entry["scene_name"] == scene_name:
            return index
    raise ValueError(f"Scene '{scene_name}' was not found in the configured dataset")


def main(argv):
    del argv
    if FLAGS.optimization_steps <= 0:
        raise ValueError("optimization_steps must be positive")
    if FLAGS.num_interpolations <= 0:
        raise ValueError("num_interpolations must be positive")
    if FLAGS.render_chunk_size <= 0:
        raise ValueError("render_chunk_size must be positive")
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    set_seed()
    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as file:
        file.write(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "validate_interpolation.log")).get_logger()
    device = torch.device("cuda")
    with gin.config_scope("train_dataset"):
        dataset = SplatFactoMultiLevelDataset()

    scene_index = find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_index)
    input_entry = scene["factor_data"][INPUT_FACTOR]
    target_entry = scene["factor_data"][TARGET_FACTOR]
    payload = build_payload(dataset, scene["idx"], scene["scene_name"], target_entry)

    x0 = gpu_utils.move_to_device(input_entry["gs_params"], device)
    images = gpu_utils.move_to_device(payload["images"], device)
    cameras = gpu_utils.move_to_device(payload["cameras"], device)
    float_keys = floating_gs_keys(x0)
    save_ground_truth_preview(images, FLAGS.output_dir)

    logger.info(
        f"scene={scene['scene_name']} splats={x0['means'].shape[0]} views={len(images)} "
        f"optimization_steps={FLAGS.optimization_steps} interpolation_count={FLAGS.num_interpolations} "
        f"float_keys={float_keys}"
    )
    video_path = os.path.join(FLAGS.output_dir, "endpoint_optimization.mp4")
    x1, optimization_history = optimize_endpoint_with_video(
        x0=x0,
        images=images,
        cameras=cameras,
        float_keys=float_keys,
        optimization_steps=FLAGS.optimization_steps,
        video_view_index=FLAGS.training_video_view_index,
        video_fps=FLAGS.training_video_fps,
        video_path=video_path,
        logger=logger,
    )

    ply_dir = os.path.join(FLAGS.output_dir, "ply")
    renders_root = os.path.join(FLAGS.output_dir, "renders")
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(renders_root, exist_ok=True)
    summary = {
        "scene_idx": int(scene["idx"]),
        "scene_name": scene["scene_name"],
        "num_splats": int(x0["means"].shape[0]),
        "num_views": len(images),
        "optimization_steps": int(FLAGS.optimization_steps),
        "num_interpolations": int(FLAGS.num_interpolations),
        "training_video_view_index": int(FLAGS.training_video_view_index),
        "training_video_fps": float(FLAGS.training_video_fps),
        "training_video_path": video_path,
        "float_keys": float_keys,
        "optimization_history": optimization_history,
        "interpolations": [],
    }

    for index in range(1, FLAGS.num_interpolations + 1):
        t = float(index) / float(FLAGS.num_interpolations)
        label = t_label(t)
        interpolated = interpolation_state(x0, x1, t, float_keys)
        stats = interpolation_stats(interpolated, float_keys)
        ply_path = os.path.join(ply_dir, f"interpolated_t_{label}.ply")
        gs_utils.export_ply_forviewer(interpolated, ply_path)

        render_dir = os.path.join(renders_root, f"t_{label}")
        os.makedirs(render_dir, exist_ok=True)
        metrics = evaluate_interpolation(
            interpolated,
            images,
            payload["images_name"],
            cameras,
            render_dir,
            FLAGS.render_chunk_size,
        )
        endpoint_error = {
            key: float((interpolated[key] - x1[key]).detach().abs().max().item())
            for key in float_keys
        }
        summary["interpolations"].append(
            {
                "index": int(index),
                "t": t,
                "ply_path": ply_path,
                "render_dir": render_dir,
                "metrics": metrics,
                "stats": stats,
                "max_abs_error_to_endpoint": endpoint_error,
            }
        )
        logger.info(
            f"interpolation t={label} psnr={metrics['psnr']:.4f} "
            f"ssim={metrics['ssim']:.4f} lpips={metrics['lpips']:.4f} "
            f"render_l1={metrics['render_l1']:.6f}"
        )

    with open(os.path.join(FLAGS.output_dir, "interpolation_summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    logger.info(f"Interpolation validation complete: {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)

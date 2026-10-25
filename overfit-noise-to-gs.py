"""Overfit unconditional Gaussian velocity models on native HR scenes."""

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
from models.gaussian_dit_predictor import GaussianDiTPredictor
from sr import interpolants, noise_to_gs
from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_normalization import GaussianStandardizer
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit_noise_to_gs", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Final evaluation subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name for one-scene overfitting")
flags.DEFINE_string("source_gs_root", "", "Optional root for source GS checkpoints")
flags.DEFINE_string("target_gs_root", "", "Optional root for target GS checkpoints")
flags.DEFINE_enum("predictor", "dit", ["ptv3", "dipt", "gdpt", "dit"], "Gaussian velocity predictor")
flags.DEFINE_enum("scene_mode", "one", ["one", "many"], "Overfit one or several scenes")
flags.DEFINE_integer("scene_count", 1, "Number of scenes in many mode")
flags.DEFINE_integer("batch_size", 1, "Scene samples per optimizer update")
flags.DEFINE_integer("grad_accum_steps", 1, "Microbatches per optimizer update")
flags.DEFINE_integer("eval_max_views", 0, "Evaluation views per scene; 0 uses every view")
flags.DEFINE_multi_string("gin_file", None, "Gin configuration files")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter bindings")
flags.register_multi_flags_validator(
    ["batch_size", "grad_accum_steps"],
    lambda values: 1 <= values["grad_accum_steps"] <= values["batch_size"],
    message="grad_accum_steps must be between 1 and batch_size",
)

FLAGS = flags.FLAGS
PREDICTORS = {
    "ptv3": GSFlowPredictor,
    "dipt": DiffusionGaussianPredictor,
    "gdpt": DiffusionGaussianPredictor,
    "dit": GaussianDiTPredictor,
}


@gin.configurable
def set_seed(seed=42):
    seed_everything(seed)
    return seed


@gin.configurable
def flow_matching(gs_statistics_path, flow_steps=10, flow_t_eps=1e-4, eval_noise_seed=0,
                  normalization_variance_floor=1e-8):
    if flow_steps < 1 or not 0 < flow_t_eps < 0.5:
        raise ValueError("Expected positive flow_steps and 0 < flow_t_eps < 0.5")
    return {
        "gs_statistics_path": gs_statistics_path,
        "flow_steps": flow_steps,
        "flow_t_eps": flow_t_eps,
        "eval_noise_seed": eval_noise_seed,
        "normalization_variance_floor": normalization_variance_floor,
    }


@torch.no_grad()
def evaluate_scene(model, scene, standardizer, flow_cfg, output_dir, ground_truth=False):
    device = next(model.parameters()).device
    if ground_truth:
        gaussians = gpu_utils.move_to_device(scene["gt_gs"], device)
    else:
        count = len(scene["target_gs"]["means"])
        gaussians = noise_to_gs.sample_flow_model(
            model, count, scene["scene_idx"], flow_cfg["flow_steps"], standardizer,
            noise_seed=flow_cfg["eval_noise_seed"], t_eps=flow_cfg["flow_t_eps"]
        )
    os.makedirs(output_dir, exist_ok=True)
    sh_degree_zero = gaussians["features_rest"].shape[1] == 0
    ply_name = "gt.ply" if ground_truth else "output.ply"
    gs_utils.export_ply_forviewer(gaussians, os.path.join(output_dir, "point_cloud", ply_name),
                                  sh_degree_zero=sh_degree_zero)
    pred_dir = os.path.join(output_dir, "gt" if ground_truth else "pred")
    os.makedirs(pred_dir, exist_ok=True)
    metrics = MetricComputer()
    images = scene["target_images"]
    cameras = scene["target_cameras"]
    chunk_size = min(scene["eval_chunk_size"], len(images))
    for start in range(0, len(images), chunk_size):
        end = min(start + chunk_size, len(images))
        chunk_cameras = {
            key: value[start:end] if key == "camera_to_worlds" else value
            for key, value in cameras.items()
        }
        chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)
        predicted, _ = gs_utils.rasterize_gaussians_to_multiimgs(
            gaussians, chunk_cameras, sh_degree_zero=sh_degree_zero
        )
        predicted = torch.stack(predicted)
        target = torch.stack(gpu_utils.move_to_device(images[start:end], device))
        if target.shape[-1] == 4:
            predicted = predicted * target[..., 3:4]
        target_uint8 = (target[..., :3] * 255).to(torch.uint8)
        predicted_uint8 = (predicted * 255).to(torch.uint8)
        metrics.update(predicted_uint8, target_uint8, name=f"{scene['scene_idx']}_{start:06d}")
        for name, image in zip(scene["target_image_names"][start:end], predicted_uint8):
            image = image.cpu().numpy()
            cv2.imwrite(os.path.join(pred_dir, name), image[:, :, ::-1])
    result = metrics.finalize()
    metrics.write_to_file(os.path.join(output_dir, "metrics_per_view.json"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


@gin.configurable
def training(dataset, scene_indices, output_dir, logger, device, total_steps=4000, eval_interval=500,
             save_interval=4000, log_interval=20, grad_clip_norm=2.0, enable_amp=True):
    if total_steps < 1 or eval_interval < 1 or save_interval < 1 or log_interval < 1:
        raise ValueError("Training and reporting intervals must be positive")
    flow_cfg = flow_matching()
    if dataset.max_gs_num != 1000 or not dataset.load_tgt_gs or not dataset.load_tgt_images:
        raise ValueError("Noise-to-GS requires native HR Gaussians and views capped at 1000")

    model_class = PREDICTORS[FLAGS.predictor]
    model = model_class().to(device)
    if FLAGS.predictor in ("dipt", "gdpt") and model.backbone_type.lower() != FLAGS.predictor:
        raise ValueError(f"Predictor {FLAGS.predictor} requires its matching backbone Gin config")
    required_outputs = set(SUPPORTED_GS_KEYS) - ({"features_rest"} if model.sh_degree == 0 else set())
    if required_outputs - set(model.output_features):
        raise ValueError("Noise-to-GS requires velocities for all active Gaussian attributes")
    if FLAGS.predictor != "dit":
        if model.quat_residual_mode != "add" or any(
            not isinstance(model.res_feature_activation[key], torch.nn.Identity) for key in required_outputs
        ):
            raise ValueError("Noise-to-GS requires additive quaternion velocity and identity output activations")
    logger.info(f"Predictor={FLAGS.predictor} class={model_class.__name__} sh_degree={model.sh_degree}")
    standardizer = GaussianStandardizer(
        flow_cfg["gs_statistics_path"], flow_cfg["normalization_variance_floor"],
        sh_degree=0 if model.sh_degree == 0 else None
    )

    prepared = []
    for scene_idx in scene_indices:
        raw = dataset.load_scene(scene_idx)
        target = raw["data"][dataset.tgt_resolution]
        target_gs = target["gs_params"]
        if model.sh_degree == 0:
            # Retain the native DC coefficient while discarding higher-order SH.
            target_gs = {**target_gs, "features_rest": target_gs["features_rest"][:, :0, :]}
        if not 1 <= len(target_gs["means"]) <= 1000 or not target["images"]:
            raise ValueError(f"Scene {raw['scene_name']} needs 1 to 1000 HR Gaussians and target views")
        view_count = min(FLAGS.eval_max_views, len(target["images"])) if FLAGS.eval_max_views > 0 else len(target["images"])
        cameras = {key: value[:view_count] if key == "camera_to_worlds" else value
                   for key, value in target["cameras"].items()}
        prepared.append({
            "scene_idx": scene_idx,
            "scene_name": raw["scene_name"],
            "gt_gs": target["gs_params"],
            "target_gs": target_gs,
            "target_flow_gs": standardizer.encode(target_gs),
            "target_images": target["images"][:view_count],
            "target_cameras": cameras,
            "target_image_names": target["images_name"][:view_count],
            "eval_chunk_size": dataset.image_per_scene if dataset.image_per_scene and dataset.image_per_scene > 0
                               else len(target["images"]),
        })
        logger.info(f"Prepared native HR scene {raw['scene_name']} with {len(target_gs['means'])} Gaussians")
    if len(prepared) > 1:
        prepared = [gpu_utils.to_cpu(scene) for scene in prepared]

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(output_dir, "config.gin"), "w") as handle:
        handle.write(gin.operative_config_str())
    with open(os.path.join(output_dir, "normalization_statistics.json"), "w") as handle:
        json.dump(standardizer.report(), handle, indent=2)
        handle.write("\n")

    # Render the native HR Gaussians once on the same views used for model evaluation.
    gt_scene_metrics = {}
    with torch.random.fork_rng(devices=[device]):
        for scene in prepared:
            scene_dir = os.path.join(output_dir, "gt_gs", scene["scene_name"])
            gt_scene_metrics[scene["scene_name"]] = evaluate_scene(
                model, scene, standardizer, flow_cfg, scene_dir, ground_truth=True
            )
            logger.info(f"GT GS scene={scene['scene_name']} metrics={gt_scene_metrics[scene['scene_name']]}")
    gt_mean = {key: sum(metrics[key] for metrics in gt_scene_metrics.values()) / len(gt_scene_metrics)
               for key in next(iter(gt_scene_metrics.values()))}

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    sizes = [FLAGS.batch_size // FLAGS.grad_accum_steps + (index < FLAGS.batch_size % FLAGS.grad_accum_steps)
             for index in range(FLAGS.grad_accum_steps)]
    scene_order = []
    for step in tqdm(range(total_steps)):
        selected = []
        for _ in range(FLAGS.batch_size):
            if len(prepared) == 1:
                selected.append(prepared[0])
            else:
                if not scene_order:
                    scene_order = np.random.permutation(len(prepared)).tolist()
                selected.append(prepared[scene_order.pop()])

        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        left = 0
        for size in sizes:
            microbatch = selected[left:left + size]
            left += size
            targets = [gpu_utils.move_to_device(scene["target_flow_gs"], device) for scene in microbatch]
            times = [torch.empty((), device=device).uniform_(flow_cfg["flow_t_eps"], 1 - flow_cfg["flow_t_eps"])
                     for _ in microbatch]
            paths = [noise_to_gs.construct_path(target, time) for target, time in zip(targets, times)]
            with torch.cuda.amp.autocast(enabled=enable_amp):
                predictions = noise_to_gs.predict_velocity(
                    model, [query for query, _ in paths], [scene["scene_idx"] for scene in microbatch],
                    torch.stack(times), standardizer
                )
                losses = [interpolants.attribute_mse(prediction, velocity)[0]
                          for prediction, (_, velocity) in zip(predictions, paths)]
                loss = sum(losses) / FLAGS.batch_size
            if enable_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            weighted_loss += loss.detach().item()

        if enable_amp:
            previous_scale = scaler.get_scale()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        else:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer_stepped = True
        if optimizer_stepped:
            scheduler.step()
        if step % log_interval == 0:
            logger.info(f"step={step} loss={weighted_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f} "
                        f"scenes={[scene['scene_name'] for scene in selected]}")

        final = step == total_steps - 1
        # Save a Gaussian snapshot for every evaluation and checkpoint step.
        if step % eval_interval == 0 or (step + 1) % save_interval == 0 or final:
            eval_dir = os.path.join(output_dir, FLAGS.eval_subdir if final else f"eval/{step:08d}")
            scene_metrics = {}
            # Rendering and metric setup must not change subsequent training noise.
            with torch.random.fork_rng(devices=[device]):
                for scene in prepared:
                    scene_dir = os.path.join(eval_dir, scene["scene_name"])
                    scene_metrics[scene["scene_name"]] = evaluate_scene(model, scene, standardizer, flow_cfg, scene_dir)
                    logger.info(f"Eval step={step} scene={scene['scene_name']} metrics={scene_metrics[scene['scene_name']]} "
                                f"gt_gs_metrics={gt_scene_metrics[scene['scene_name']]}")
            mean = {key: sum(metrics[key] for metrics in scene_metrics.values()) / len(scene_metrics)
                    for key in next(iter(scene_metrics.values()))}
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "metrics.json"), "w") as handle:
                json.dump({"mean": mean, "scenes": scene_metrics,
                           "gt_gs": {"mean": gt_mean, "scenes": gt_scene_metrics}}, handle, indent=2)
                handle.write("\n")
        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"model_{step:08d}.pth"))
    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "model_last.pth"))


def main(argv):
    del argv
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    seed = set_seed()
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    dataset = SplatFactoSRDataset.from_gin_scope("test_dataset")
    if dataset.src_resolution == dataset.tgt_resolution and FLAGS.source_gs_root and FLAGS.target_gs_root:
        if os.path.abspath(FLAGS.source_gs_root) != os.path.abspath(FLAGS.target_gs_root):
            raise ValueError("Equal source and target resolution requires the same GS checkpoint root")
    # Keep image paths under the dataset root while selecting separate GS checkpoints.
    for resolution, root in ((dataset.src_resolution, FLAGS.source_gs_root),
                             (dataset.tgt_resolution, FLAGS.target_gs_root)):
        if root:
            for entry in dataset.folders:
                entry["resolution_paths"][resolution]["gsplat_dir"] = os.path.join(
                    root, f"{dataset.train_or_test}-set", dataset.dataset_name,
                    str(resolution), "gsplat", entry["scene_name"]
                )
    if FLAGS.scene_mode == "one":
        scene_indices = [dataset.scene_index(FLAGS.scene_name)]
    else:
        unique = {}
        for index, entry in enumerate(dataset.folders):
            unique.setdefault(entry["scene_name"], index)
        if not 1 <= FLAGS.scene_count <= len(unique):
            raise ValueError(f"scene_count must be between 1 and {len(unique)}")
        scene_indices = np.random.default_rng(seed).choice(list(unique.values()), size=FLAGS.scene_count,
                                                            replace=False).tolist()
    with open(os.path.join(FLAGS.output_dir, "selected_scenes.json"), "w") as handle:
        json.dump([{"scene_idx": index, "scene_name": dataset.folders[index]["scene_name"]}
                   for index in scene_indices], handle, indent=2)
        handle.write("\n")
    training(dataset, scene_indices, FLAGS.output_dir, logger, torch.device("cuda"))


if __name__ == "__main__":
    app.run(main)

"""Overfit SR-MSE using a fitted, count-preserving Gaussian target.

Unlike ``overfit-sr-mse.py``, this entrypoint never densifies or aligns point
sets.  It first fits the input-factor Gaussians to target-factor image views
with gsplat.  Optimisation changes values but never changes parameter order or
cardinality, so the fitted result is an identity-indexed MSE target.
"""

import importlib.util
import json
import os
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import copy_gt_attributes, make_grid, scale_means_origin, unscale_means_origin
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import fixed_attribute_keys as select_fixed_attribute_keys
from utils.loss_utils import load_gs_statistics_normalizers
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_dataset_utils import build_dataset, find_scene_index
from utils.sr_matching_utils import (
    build_matching_source as build_shared_matching_source,
    get_or_fit_matching_target as get_or_fit_shared_matching_target,
    matching_fit,
    save_matching_artifacts as save_shared_matching_artifacts,
    _write_preview
)


def _load_base_module():
    """Reuse the mature SR-MSE evaluation and flag surface without copying it."""
    base_path = Path(__file__).with_name("overfit-sr-mse.py")
    spec = importlib.util.spec_from_file_location("_overfit_sr_mse_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base SR-MSE script: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
FLAGS = BASE.FLAGS
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



def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = BASE.training(output_dir=FLAGS.output_dir)
    matching_cfg = matching_fit()
    mse_loss_cfg = BASE.feature_mse_loss()
    BASE.set_seed()

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = build_dataset()
    scene_idx = find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_dict = scene["factor_data"][FLAGS.input_factor]
    target_factor_dict = scene["factor_data"][FLAGS.target_factor]
    target_images, target_image_names, target_cameras = dataset.load_factor_views(target_factor_dict)

    source_gs = build_shared_matching_source(input_factor_dict, target_factor_dict, device)
    matching_target_gs, matching_cache = get_or_fit_shared_matching_target(
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
    matching_metrics = save_shared_matching_artifacts(
        FLAGS.output_dir,
        source_gs,
        matching_target_gs,
        target_images,
        target_cameras,
        eval_chunk_size,
        device,
    )
    for ply_name, metrics in matching_metrics.items():
        logger.info("Matching init %s: %s", ply_name, " ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    loss_features = BASE.resolve_loss_features(FLAGS.loss_features, matching_target_gs)
    model_input_features = BASE.resolve_model_input_features(matching_target_gs)
    with gin.unlock_config():
        gin.bind_parameter("FeaturePredictor.input_features", model_input_features)
        if FLAGS.model_features_from_loss:
            gin.bind_parameter("FeaturePredictor.output_features", loss_features)

    model = FeaturePredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    fixed_attribute_keys = select_fixed_attribute_keys(loss_features, matching_target_gs)
    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = requested_means_origin_scale if "means" in loss_features else 1.0

    component_normalizers = None
    selected_statistics = None
    loss_target_gs = scale_means_origin(matching_target_gs, means_origin_scale)
    effective_loss_weights = mse_loss_cfg["loss_weights"]
    if FLAGS.gs_statistics_path is not None:
        if FLAGS.post_activate_loss:
            raise ValueError("--gs_statistics_path cannot be used with --post_activate_loss")
        component_normalizers, selected_statistics = load_gs_statistics_normalizers(
            FLAGS.gs_statistics_path,
            FLAGS.target_factor,
            loss_features,
            matching_target_gs,
        )
        if "means" in component_normalizers:
            component_normalizers["means"] = component_normalizers["means"] * means_origin_scale
        effective_loss_weights = {key: 1.0 for key in loss_features}
        statistics_message = f"GS statistics df-{FLAGS.target_factor}:\n{json.dumps(selected_statistics, indent=2)}"
        print(statistics_message)
        logger.info(statistics_message)

    batch_gs = [source_gs]
    batch_scene_idx = [scene["idx"]]
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as handle:
        handle.write(gin.operative_config_str())

    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]

    training_brief = (
        f"No-EMD SR-MSE scene={scene['scene_name']} idx={scene['idx']}\n"
        f"target_views={len(target_images)}\n"
        f"input_gaussians={input_factor_dict['gs_params']['means'].shape[0]}\n"
        f"matching_target_gaussians={matching_target_gs['means'].shape[0]}\n"
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor}\n"
        f"matching_cache={matching_cache['status']}\n"
        f"matching_cache_path={matching_cache['checkpoint_path']}\n"
        f"matching_steps={matching_cfg['total_steps']}\n"
        f"matching_images_per_step={matching_cfg['image_per_step']}\n"
        f"loss_features={','.join(loss_features)}\n"
        f"model_features_from_loss={FLAGS.model_features_from_loss}\n"
        f"model_input_features={','.join(model.input_features)}\n"
        f"model_output_features={','.join(model.output_features)}\n"
        f"post_activate_loss={FLAGS.post_activate_loss}\n"
        f"means_origin_scale={requested_means_origin_scale}\n"
        f"effective_means_origin_scale={means_origin_scale}\n"
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']}\n"
        f"gs_statistics_path={FLAGS.gs_statistics_path or 'none'}\n"
        f"statistics_factor=df-{FLAGS.target_factor}\n"
        f"loss_weights={effective_loss_weights}\n"
        f"eval_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'}\n"
    )
    print(training_brief)
    logger.info(training_brief)

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)
    target_images_device = gpu_utils.move_to_device(target_images, device)
    gt_images = [(image[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for image in target_images_device]
    cv2.imwrite(
        os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"),
        cv2.cvtColor(make_grid(gt_images), cv2.COLOR_RGB2BGR),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps), desc="SR-MSE")
    for step in pbar:
        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
            total_loss, feature_losses, weighted_feature_losses = BASE.compute_feature_mse_loss(
                out_gs,
                loss_target_gs,
                loss_features,
                post_activate_loss=FLAGS.post_activate_loss,
                loss_weights=effective_loss_weights,
                quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                component_normalizers=component_normalizers,
                means_loss_reduction=BASE.MEANS_LOSS_REDUCTION,
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
        pbar.set_postfix(loss=f"{total_loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()
        if step % log_interval == 0:
            feature_loss_str = " ".join(
                f"{key}_loss={value:.6f} {key}_weighted={weighted_loss_values[key]:.6f}"
                for key, value in feature_loss_values.items()
            )
            logger.info("step=%d total=%.6f %s lr=%.8f", step, total_loss.item(), feature_loss_str, optimizer.param_groups[0]["lr"])
        if step % log_image_interval == 0:
            with torch.no_grad():
                log_out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                log_out_gs = copy_gt_attributes(log_out_gs, matching_target_gs, fixed_attribute_keys)
                log_out_gs = unscale_means_origin(log_out_gs, means_origin_scale)
                batch_cameras = gpu_utils.move_to_device(target_cameras, device)
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(log_out_gs, batch_cameras)
            _write_preview(pred_imgs, os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"))
        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = BASE.evaluate_single_scene(
                model=model,
                input_gs=source_gs,
                gt_gs=matching_target_gs,
                scene_idx=scene["idx"],
                scene_name=scene["scene_name"],
                eval_images=target_images,
                eval_cameras=target_cameras,
                image_names=target_image_names,
                output_dir=eval_dir,
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                output_gt=(step == 0),
                fixed_gt_gs=matching_target_gs,
                fixed_attribute_keys=fixed_attribute_keys,
                means_origin_scale=means_origin_scale,
            )
            logger.info("Eval step %d: %s", step, " ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            if FLAGS.compare_with_input:
                logger.info("Eval input step %d: %s", step, " ".join(f"{k}: {v:.4f}" for k, v in metrics_input.items()))
        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))
    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = BASE.evaluate_single_scene(
        model=model,
        input_gs=source_gs,
        gt_gs=matching_target_gs,
        scene_idx=scene["idx"],
        scene_name=scene["scene_name"],
        eval_images=target_images,
        eval_cameras=target_cameras,
        image_names=target_image_names,
        output_dir=final_eval_dir,
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        output_gt=True,
        fixed_gt_gs=matching_target_gs,
        fixed_attribute_keys=fixed_attribute_keys,
        means_origin_scale=means_origin_scale,
    )
    logger.info("Final eval: %s", " ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    print(f"Final eval: {metrics}")
    if FLAGS.compare_with_input:
        logger.info("Final eval input: %s", " ".join(f"{k}: {v:.4f}" for k, v in metrics_input.items()))
        print(f"Final eval input: {metrics_input}")


if __name__ == "__main__":
    app.run(main)

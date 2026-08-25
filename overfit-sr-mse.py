"""Single-scene SR overfitting with four identity-indexed alignment modes."""

import json
import os

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_predictor import FeaturePredictor
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import (
    SUPPORTED_GS_KEYS,
    feature_loss_value,
    load_gs_statistics_normalizers,
)
from utils.metrics import MetricComputer
from utils.optimizers import build_3DGSoptimizer, build_optimizer, build_scheduler

mathcing_types = ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"]
init_options = ["aligned", "3dgs"]


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with aligned input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_integer("input_resolution", 128, "Low-resolution GS/image resolution")
flags.DEFINE_integer("target_resolution", 512, "High-resolution GS/image resolution")
flags.DEFINE_enum("alignment", "emd", mathcing_types, "Matching input/target Gaussian pair.")
flags.DEFINE_enum("attribute_init", "aligned", init_options, "For emd and random only.")
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_boolean("post_activate_loss", True, "Post activation loss")
flags.DEFINE_string("gs_statistics_path", None, "Channel-normalized attribute MSE.")
flags.DEFINE_string("matching_cache_root", "/project2/ricky/splatformer-data-to-4x", "Reverse-fit cache")
flags.DEFINE_boolean(
    "force_matching_fit",
    False,
    "Rerun the selected fit without reading or writing its persistent cache",
)
flags.DEFINE_multi_string("gin_file", None, "List of paths to Gin config files")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter bindings")

FLAGS = flags.FLAGS
MEANS_LOSS_REDUCTION = "mean"

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
def matching_fit(
    total_steps=1000,
    image_per_step=16,
    log_interval=20,
    preview_interval=200,
    grad_clip_norm=0.0,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=1.0,
    enable_amp=True,
    empty_cache_fre=-1,
):
    return {
        "total_steps": total_steps,
        "image_per_step": image_per_step,
        "log_interval": log_interval,
        "preview_interval": preview_interval,
        "grad_clip_norm": grad_clip_norm,
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


def build_matching_optimization(gaussians):
    with gin.config_scope("matching_fit"):
        optimizer = build_3DGSoptimizer(gaussians)
        scheduler = build_scheduler(optimizer)
    return optimizer, scheduler


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
    post_activate_loss,
    quat_direct_mse,
    component_normalizers=None,
):
    missing_output = [
        key for key in SUPPORTED_GS_KEYS if key not in out_gs
    ]
    missing_target = [
        key for key in SUPPORTED_GS_KEYS if key not in target_gs
    ]
    if missing_output:
        raise ValueError(
            f"FeaturePredictor must output every Gaussian attribute; "
            f"missing {missing_output}"
        )
    if missing_target:
        raise ValueError(
            f"All-attribute loss target is missing {missing_target}"
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
        normalizer = None
        if component_normalizers is not None:
            normalizer = component_normalizers[key].to(
                device=pred.device, dtype=pred.dtype
            )
            if normalizer.shape != pred.shape[1:]:
                raise ValueError(
                    f"Normalizer shape mismatch for Gaussian attribute {key!r}: "
                    f"normalizer {tuple(normalizer.shape)} vs "
                    f"feature {tuple(pred.shape[1:])}"
                )
        loss = feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=post_activate_loss,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=MEANS_LOSS_REDUCTION,
            component_normalizer=normalizer,
        )
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    if not total_loss.requires_grad:
        raise ValueError(
            "FeaturePredictor outputs do not receive gradients for the "
            "all-attribute loss"
        )
    return total_loss, losses, weighted_losses


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
    output_gt=True,
):
    """Render one predicted scene and write image metrics and viewer artifacts."""
    model.eval()
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
        out_gs = model(batch_normalized_gs=[input_gs], batch_scene_idx=[scene_idx])[0]
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
                pred_preview.extend(
                    image.cpu().numpy().astype(np.uint8)
                    for image in pred_imgs[:preview_slots]
                )
                if output_gt:
                    gt_preview.extend(
                        image.cpu().numpy().astype(np.uint8)
                        for image in gt_imgs[:preview_slots]
                    )

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
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)
                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    comparison = np.concatenate(
                        [
                            gt_img.cpu().numpy().astype(np.uint8),
                            input_img.cpu().numpy().astype(np.uint8),
                            pred_img.cpu().numpy().astype(np.uint8),
                        ],
                        axis=1,
                    )
                    cv2.imwrite(
                        os.path.join(compare_dir, f"{global_idx:04d}.png"),
                        comparison[:, :, ::-1],
                    )

        if pred_preview:
            cv2.imwrite(
                os.path.join(output_dir, f"scene{scene_idx}_pred.png"),
                cv2.cvtColor(make_grid(pred_preview), cv2.COLOR_RGB2BGR),
            )
        if output_gt and gt_preview:
            cv2.imwrite(
                os.path.join(output_dir, f"scene{scene_idx}_gt.png"),
                cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR),
            )
        if save_viewer:
            viewer_dir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewer_dir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewer_dir, model.sh_degree)
            point_cloud_dir = os.path.join(viewer_dir, "point_cloud")
            gs_utils.export_ply_forviewer(input_gs, os.path.join(point_cloud_dir, "input.ply"))
            gs_utils.export_ply_forviewer(out_gs, os.path.join(point_cloud_dir, "output.ply"))
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(gt_gs, os.path.join(point_cloud_dir, "gt.ply"))

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
    if FLAGS.gs_statistics_path is not None and FLAGS.post_activate_loss:
        raise ValueError("--gs_statistics_path cannot be used with --post_activate_loss")

    train_cfg = training(output_dir=FLAGS.output_dir)
    mse_loss_cfg = feature_mse_loss()
    set_seed()
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    # Load one input/target-resolution scene and its high-resolution evaluation views.
    dataset = SplatFactoSRDataset.from_gin_scope("test_dataset")
    scene_idx = dataset.scene_index(FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_resolution_entry = scene["resolution_data"][FLAGS.input_resolution]
    target_resolution_entry = scene["resolution_data"][FLAGS.target_resolution]
    target_images, target_image_names, target_cameras = dataset.load_resolution_views(
        target_resolution_entry
    )
    target_gs = gpu_utils.move_to_device(target_resolution_entry["gs_params"], device)
    eval_chunk_size = dataset.image_per_scene or len(target_images)

    aligned_input_gs, attribute_target_gs, alignment_info = prepare_alignment(
        dataset=dataset,
        scene=scene,
        input_resolution_entry=input_resolution_entry,
        target_resolution_entry=target_resolution_entry,
        target_images=target_images,
        target_cameras=target_cameras,
        target_gs=target_gs,
        output_dir=FLAGS.output_dir,
        logger=logger,
        device=device,
        eval_chunk_size=eval_chunk_size,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        input_resolution=FLAGS.input_resolution,
        target_resolution=FLAGS.target_resolution,
        matching_cache_root=FLAGS.matching_cache_root,
        force_matching_fit=FLAGS.force_matching_fit,
        matching_config=matching_fit(),
        matching_optimizer_factory=build_matching_optimization,
    )

    # Unified training consumes, predicts, and supervises every Gaussian attribute.
    model = FeaturePredictor().to(device)
    missing_outputs = sorted(set(SUPPORTED_GS_KEYS) - set(model.output_features))
    if missing_outputs:
        raise ValueError(
            f"FeaturePredictor.output_features must include all Gaussian "
            f"attributes; missing {missing_outputs}"
        )
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    loss_target_gs = attribute_target_gs
    component_normalizers = None
    selected_statistics = None
    effective_loss_weights = mse_loss_cfg["loss_weights"]
    if FLAGS.gs_statistics_path is not None:
        component_normalizers, selected_statistics = load_gs_statistics_normalizers(
            FLAGS.gs_statistics_path,
            FLAGS.target_resolution,
            SUPPORTED_GS_KEYS,
            attribute_target_gs,
            alignment=FLAGS.alignment,
            resolutions=dataset.resolutions,
        )
        effective_loss_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
        statistics_message = (
            f"GS statistics resolution={FLAGS.target_resolution}:\n"
            f"{json.dumps(selected_statistics, indent=2)}"
        )
        print(statistics_message)
        logger.info(statistics_message)

    batch_gs = [aligned_input_gs]
    batch_scene_idx = [scene["idx"]]
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)
    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as handle:
        handle.write(gin.operative_config_str())

    training_brief = (
        f"Unified SR-MSE scene={scene['scene_name']} idx={scene['idx']}\n"
        f"alignment={FLAGS.alignment}\n"
        f"attribute_init={FLAGS.attribute_init if FLAGS.alignment not in {'fit_lr_to_hr', 'fit_hr_to_lr'} else 'inactive'}\n"
        f"input_resolution={FLAGS.input_resolution} target_resolution={FLAGS.target_resolution}\n"
        f"original_input_gaussians={input_resolution_entry['gs_params']['means'].shape[0]}\n"
        f"aligned_input_gaussians={aligned_input_gs['means'].shape[0]}\n"
        f"attribute_target_gaussians={attribute_target_gs['means'].shape[0]}\n"
        f"attribute_keys={','.join(SUPPORTED_GS_KEYS)}\n"
        f"loss_type=attribute_mse_only\n"
        f"post_activate_loss={FLAGS.post_activate_loss}\n"
        f"gs_statistics_path={FLAGS.gs_statistics_path or 'none'}\n"
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']}\n"
        f"loss_weights={effective_loss_weights}\n"
        f"alignment_info={alignment_info}\n"
    )
    print(training_brief)
    logger.info(training_brief)

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)
    target_images_device = gpu_utils.move_to_device(target_images, device)
    gt_images = [
        (image[..., :3] * 255).detach().cpu().numpy().astype(np.uint8)
        for image in target_images_device
    ]
    cv2.imwrite(
        os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"),
        cv2.cvtColor(make_grid(gt_images), cv2.COLOR_RGB2BGR),
    )

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
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps), desc="SR-MSE")
    for step in pbar:
        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
            total_loss, feature_losses, weighted_feature_losses = (
                compute_all_feature_mse_loss(
                    out_gs=out_gs,
                    target_gs=loss_target_gs,
                    loss_weights=effective_loss_weights,
                    post_activate_loss=FLAGS.post_activate_loss,
                    quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                    component_normalizers=component_normalizers,
                )
            )

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

        feature_values = {key: value.item() for key, value in feature_losses.items()}
        weighted_values = {key: value.item() for key, value in weighted_feature_losses.items()}
        pbar.set_postfix(loss=f"{total_loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()
        if step % log_interval == 0:
            feature_message = " ".join(
                f"{key}_loss={value:.6f} {key}_weighted={weighted_values[key]:.6f}"
                for key, value in feature_values.items()
            )
            logger.info(
                "step=%d total=%.6f %s lr=%.8f",
                step,
                total_loss.item(),
                feature_message,
                optimizer.param_groups[0]["lr"],
            )
        if step % log_image_interval == 0:
            with torch.no_grad():
                preview_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                preview_cameras = gpu_utils.move_to_device(target_cameras, device)
                pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(preview_gs, preview_cameras)
            pred_images = [(image * 255).detach().cpu().numpy().astype(np.uint8) for image in pred_images]
            cv2.imwrite(
                os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"),
                cv2.cvtColor(make_grid(pred_images), cv2.COLOR_RGB2BGR),
            )
        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, input_metrics = evaluate_single_scene(
                model=model,
                input_gs=aligned_input_gs,
                gt_gs=attribute_target_gs,
                scene_idx=scene["idx"],
                scene_name=scene["scene_name"],
                eval_images=target_images,
                eval_cameras=target_cameras,
                image_names=target_image_names,
                output_dir=eval_dir,
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                output_gt=step == 0,
            )
            logger.info("Eval step %d: %s", step, " ".join(f"{key}: {value:.4f}" for key, value in metrics.items()),)
            if FLAGS.compare_with_input:
                logger.info("Eval input step %d: %s", step, " ".join(f"{key}: {value:.4f}" for key, value in input_metrics.items()),)
        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"),)

    torch.save(
        model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth")
    )
    final_metrics, final_input_metrics = evaluate_single_scene(
        model=model,
        input_gs=aligned_input_gs,
        gt_gs=attribute_target_gs,
        scene_idx=scene["idx"],
        scene_name=scene["scene_name"],
        eval_images=target_images,
        eval_cameras=target_cameras,
        image_names=target_image_names,
        output_dir=os.path.join(FLAGS.output_dir, FLAGS.eval_subdir),
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        output_gt=True,
    )
    final_message = " ".join(
        f"{key}: {value:.4f}" for key, value in final_metrics.items()
    )
    logger.info("Final eval: %s", final_message)
    print(f"Final eval: {final_message}")
    if FLAGS.compare_with_input:
        input_message = " ".join(
            f"{key}: {value:.4f}" for key, value in final_input_metrics.items()
        )
        logger.info("Final eval input: %s", input_message)
        print(f"Final eval input: {input_message}")


if __name__ == "__main__":
    app.run(main)

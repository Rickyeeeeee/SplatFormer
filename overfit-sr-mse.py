import os

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
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.


@gin.configurable
def set_seed(seed):
    seed_everything(seed)


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
        if fixed_gt_gs is not None:
            out_gs = copy_gt_attributes(out_gs, fixed_gt_gs, fixed_attribute_keys)
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

    # Dataset
    dataset = build_dataset()
    scene_idx = find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_dict = scene["factor_data"][FLAGS.input_factor]
    target_factor_dict = scene["factor_data"][FLAGS.target_factor]

    target_images, target_image_names, target_cameras = dataset.load_factor_views(target_factor_dict)

    target_gs = gpu_utils.move_to_device(target_factor_dict["gs_params"], device)

    # Model
    model = FeaturePredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    # Loss Function
    loss_features = parse_loss_features(FLAGS.loss_features, model, target_gs)
    fixed_attribute_keys = select_fixed_attribute_keys(loss_features, target_gs)
    requested_means_origin_scale = float(FLAGS.means_origin_scale)
    if requested_means_origin_scale <= 0.0:
        raise ValueError(f"--means_origin_scale must be > 0, got {requested_means_origin_scale}")
    means_origin_scale = requested_means_origin_scale if "means" in loss_features else 1.0

    # Stpe 1: Densify Gaussians
    densified_input_gs, densify_stage_gs = build_densified_input_gs(
        input_factor_dict=input_factor_dict,
        target_factor_dict=target_factor_dict,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        return_stages=True,
    )

    # Step 2: Choose parameter subsets
    # Toggle this line to test the old behavior: non-loss attributes are replaced
    # with GT before the densified GS is used as model input.
    # densified_input_gs = copy_gt_attributes(densified_input_gs, target_gs, fixed_attribute_keys)
    loss_target_gs = scale_means_origin(target_gs, means_origin_scale)
    densify_stage_gs["03_input_high_res_gs.ply"] = densified_input_gs
    save_densify_stage_plys(
        output_dir=FLAGS.output_dir,
        low_res_gs=densify_stage_gs["00_low_res_gs.ply"],
        interpolated_gs=densify_stage_gs["01_interpolated_high_res_gs.ply"],
        gt_high_res_gs=densify_stage_gs["02_gt_high_res_gs.ply"],
        input_high_res_gs=densify_stage_gs["03_input_high_res_gs.ply"],
    )
    batch_gs = gpu_utils.move_to_device([densified_input_gs], device)
    batch_scene_idx = [scene["idx"]]

    # For Debugging purpose
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

    # Step 3: Prepare model training
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

    training_brief = (
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} \n"
        f"target_views={len(target_images)} \n"
        f"input_gaussians={input_factor_dict['gs_params']['means'].shape[0]} \n"
        f"densified_gaussians={batch_gs[0]['means'].shape[0]} \n"
        f"target_gaussians={target_factor_dict['gs_params']['means'].shape[0]} \n"
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} \n"
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} \n"
        f"loss_features={','.join(loss_features)} \n"
        f"post_activate_loss={FLAGS.post_activate_loss} \n"
        f"means_origin_scale={requested_means_origin_scale} \n"
        f"effective_means_origin_scale={means_origin_scale} \n"
        f"quat_direct_mse={mse_loss_cfg['quat_direct_mse']} \n"
        f"loss_weights={mse_loss_cfg['loss_weights']} \n"
        f"eval_gt_attributes={','.join(fixed_attribute_keys) if fixed_attribute_keys else 'none'} \n"
    )

    print(training_brief)
    logger.info(training_brief)

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([target_images], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)


    # Step 4: Start training
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_batch_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)
            out_gs = out_batch_gs[0]
            total_loss, feature_losses, weighted_feature_losses = compute_feature_mse_loss(
                out_gs,
                loss_target_gs,
                loss_features,
                post_activate_loss=FLAGS.post_activate_loss,
                loss_weights=mse_loss_cfg["loss_weights"],
                quat_direct_mse=mse_loss_cfg["quat_direct_mse"],
                means_loss_reduction=MEANS_LOSS_REDUCTION,
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
                log_out_gs = copy_gt_attributes(log_out_gs, target_gs, fixed_attribute_keys)
                log_out_gs = unscale_means_origin(log_out_gs, means_origin_scale)
                batch_cameras = gpu_utils.move_to_device(target_cameras, device)
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
                fixed_gt_gs=target_gs,
                fixed_attribute_keys=fixed_attribute_keys,
                means_origin_scale=means_origin_scale,
            )
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Eval step {step}: {metric_str}")
            print(f"Eval step {step}: {metric_str}")
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
        fixed_gt_gs=target_gs,
        fixed_attribute_keys=fixed_attribute_keys,
        means_origin_scale=means_origin_scale,
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    print(f"Final eval input: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")
        print(f"Final eval input: {metric_str}")


if __name__ == "__main__":
    app.run(main)

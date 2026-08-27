import json
import os
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

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy Gin keys.
from sr import flow
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_sr_gsfm", "Output directory")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("save_residuals", False, "Accepted for trainer compatibility")
flags.DEFINE_boolean("use_wandb", True, "Log metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_enum("alignment", "emd", ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"], "Matching modes")
flags.DEFINE_enum("attribute_init", "aligned", ["aligned", "3dgs"], "attribute initialization for emd and random.",)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
EVAL_FLOW_STEPS = [1, 5, 20]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.


@gin.configurable
def set_seed(seed):
    seed_everything(seed)


@gin.configurable
def flow_matching(
    gs_statistics_path="/project/ricky/splatformer-sr-data/gs_statistics.json",
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
        "gs_statistics_path": gs_statistics_path,
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
    quat_direct_mse,
):
    missing_output = [
        key for key in SUPPORTED_GS_KEYS if key not in out_gs
    ]
    missing_target = [
        key for key in SUPPORTED_GS_KEYS if key not in target_gs
    ]
    if missing_output:
        raise ValueError(
            f"GSFlowPredictor must output every Gaussian attribute; "
            f"missing {missing_output}"
        )
    if missing_target:
        raise ValueError(
            f"All-attribute flow target is missing {missing_target}"
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
        loss = loss_utils.feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=True,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=MEANS_LOSS_REDUCTION,
        )
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    if not total_loss.requires_grad:
        raise ValueError(
            "GSFlowPredictor outputs do not receive gradients for the "
            "all-attribute x1 loss"
        )
    return total_loss, losses, weighted_losses


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
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    output_gt=True,
):
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
        out_gs = flow.sample_flow_model(model, source_flow_gs, scene_idx, flow_steps)
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



def init_wandb(output_dir):
    if not FLAGS.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed")
    return wandb.init(project=FLAGS.wandb_project, dir=FLAGS.wandb_dir, name=FLAGS.wandb_name)


def wandb_log(values, step=None):
    if wandb is not None and wandb.run is not None:
        wandb.log(values, step=step)


def build_dataset(scope, alignment=None, **kwargs):
    return SplatFactoSRDataset.from_gin_scope(scope, alignment=alignment, **kwargs)


def build_gaussian_pair(dataset, scene, device):
    input_data = scene["data"][dataset.src_resolution]
    target_data = scene["data"][dataset.tgt_resolution]
    target_gs = gpu_utils.move_to_device(target_data["gs_params"], device)
    if FLAGS.alignment == "fit_lr_to_hr":
        source_gs = gpu_utils.move_to_device(input_data["gs_params"], device)
        target_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
    elif FLAGS.alignment == "fit_hr_to_lr":
        source_gs = gpu_utils.move_to_device(scene[FLAGS.alignment]["tgt_gs"], device)
    else:
        source_gs, target_gs, _ = prepare_alignment(
            dataset=dataset, scene=scene, input_resolution_entry=input_data,
            target_resolution_entry=target_data, target_images=None,
            target_cameras=None, target_gs=target_gs, output_dir="", logger=None,
            device=device, eval_chunk_size=0, alignment=FLAGS.alignment,
            attribute_init=FLAGS.attribute_init, emd_eps=FLAGS.emd_eps,
            emd_iters=FLAGS.emd_iters, input_resolution=dataset.src_resolution,
            target_resolution=dataset.tgt_resolution, write_artifacts=False,
        )
    return target_data, source_gs, target_gs


def evaluate_dataset(model, dataset, output_dir):
    device = next(model.parameters()).device
    results = {flow_steps: [] for flow_steps in EVAL_FLOW_STEPS}
    for scene_idx in tqdm(range(len(dataset.folders)), desc="Evaluating"):
        scene = dataset.load_scene(scene_idx, fit_alignment=FLAGS.alignment)
        target_data, source_gs, target_gs = build_gaussian_pair(dataset, scene, device)
        source_flow_gs = gs_utils.clone_gaussians(source_gs)
        for flow_steps in EVAL_FLOW_STEPS:
            metrics, _ = evaluate_single_scene(
                model=model, input_gs=source_gs, source_flow_gs=source_flow_gs,
                scene_idx=scene["scene_idx"], scene_name=scene["scene_name"],
                eval_images=target_data["images"], eval_cameras=target_data["cameras"],
                image_names=target_data["images_name"],
                output_dir=os.path.join(output_dir, f"flow_steps_{flow_steps:02d}", scene["scene_name"]),
                flow_steps=flow_steps, eval_chunk_size=len(target_data["images"]),
                gt_gs=target_gs, compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer, output_gt=True,
            )
            results[flow_steps].append(metrics)
    reduced = {
        flow_steps: {key: float(np.mean([metrics[key] for metrics in values])) for key in values[0]}
        for flow_steps, values in results.items() if values
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as output_file:
        json.dump(reduced, output_file, indent=2)
    return reduced


@gin.configurable("training")
def training_config(
    output_dir=None, total_steps=gin.REQUIRED, eval_interval=gin.REQUIRED,
    log_interval=gin.REQUIRED, save_interval=gin.REQUIRED,
    log_image_interval=gin.REQUIRED, grad_clip_norm=gin.REQUIRED,
    image_l1_loss_weight=1.0, lpips_loss_weight=0.0, resume_from_step=0,
    enable_amp=False, empty_cache_fre=-1,
):
    return locals()


def training():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    config = training_config(output_dir=FLAGS.output_dir)
    flow_config = flow_matching()
    mix_config = loss_mixing()
    mse_config = feature_mse_loss()
    needs_target_images = mix_config["schedule"] != "fm-only"
    set_seed()
    device = torch.device("cuda")
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger()
    wandb_run = init_wandb(FLAGS.output_dir)
    train_dataset = build_dataset(
        "train_dataset", alignment=FLAGS.alignment, load_src_gs=True,
        load_tgt_gs=True, load_src_images=False,
        load_tgt_images=needs_target_images,
    )
    test_dataset = build_dataset(
        "test_dataset", load_src_gs=True, load_tgt_gs=True,
        load_src_images=True, load_tgt_images=True,
    )
    model = GSFlowPredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)
    velocity_variances = None
    variance_report = {}
    if flow_config["loss_type"] == "velocity":
        raw_variances, velocity_variances = flow.load_aggregate_velocity_variances(
            flow_config["gs_statistics_path"], flow_config["velocity_variance_floor"],
            device=device, dtype=next(model.parameters()).dtype,
        )
        variance_report = {
            key: {"variance": raw_variances[key].detach().cpu().tolist(),
                  "effective_variance": velocity_variances[key].detach().cpu().tolist()}
            for key in SUPPORTED_GS_KEYS
        }
    brief = (
        f"Train SR GSFM scenes={len(train_dataset.folders)} test_scenes={len(test_dataset.folders)}\n"
        f"alignment={FLAGS.alignment} mix_schedule={mix_config['schedule']} target_training_images={needs_target_images}\n"
        f"flow_config={flow_config}\nvelocity_variances={json.dumps(variance_report)}"
    )
    print(brief)
    logger.info(brief)
    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as output_file:
        output_file.write(gin.operative_config_str())
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=config["enable_amp"])
    lpips_function = loss_utils.lpips_loss_fn() if needs_target_images and config["lpips_loss_weight"] > 0 else None
    if not FLAGS.only_eval:
        optimizer.zero_grad(set_to_none=True)
        train_iter = iter(train_dataset)
        progress = tqdm(range(config["resume_from_step"], config["total_steps"]), desc="Training")
        for train_step in progress:
            scene = next(train_iter)
            target_data, source_gs, target_gs = build_gaussian_pair(train_dataset, scene, device)
            source_flow_gs = gs_utils.clone_gaussians(source_gs)
            target_flow_gs = gs_utils.clone_gaussians(target_gs)
            time_value = torch.empty(1, device=device).uniform_(float(flow_config["flow_t_eps"]), 1.0 - float(flow_config["flow_t_eps"]))
            if needs_target_images:
                view_count = min(train_dataset.image_per_scene or len(target_data["images"]), len(target_data["images"]))
                indices = np.random.permutation(len(target_data["images"]))[:view_count]
                train_images = gpu_utils.move_to_device([target_data["images"][index] for index in indices], device)
                train_cameras = dict(target_data["cameras"])
                train_cameras["camera_to_worlds"] = target_data["cameras"]["camera_to_worlds"][indices]
                train_cameras = gpu_utils.move_to_device(train_cameras, device)
            query_gs, flow_noise, gamma, gamma_dot = flow.sample_stochastic_interpolant(source_flow_gs, target_flow_gs, time_value, float(flow_config["flow_noise_std"]))
            with torch.cuda.amp.autocast(enabled=config["enable_amp"]):
                predicted_velocity = model(batch_flow_gs=[query_gs], batch_scene_idx=[scene["scene_idx"]], batch_reference_means=[source_flow_gs["means"]], t=time_value)[0]
                predicted_x1 = flow.predict_x1_from_velocity(model, source_flow_gs, query_gs, predicted_velocity, flow_noise, gamma, gamma_dot, time_value)
                if flow_config["loss_type"] == "velocity":
                    flow_loss, attribute_losses, weighted_attribute_losses = flow.compute_variance_normalized_velocity_loss(
                        pred_vel=predicted_velocity, source_flow_gs=source_flow_gs,
                        target_flow_gs=target_flow_gs, flow_noise=flow_noise,
                        gamma_dot=gamma_dot, velocity_variances=velocity_variances,
                        loss_weights={key: 1.0 for key in SUPPORTED_GS_KEYS},
                    )
                else:
                    flow_loss, attribute_losses, weighted_attribute_losses = compute_all_feature_mse_loss(predicted_x1, target_gs, mse_config["loss_weights"], mse_config["quat_direct_mse"])
                if needs_target_images:
                    predicted_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(predicted_x1, train_cameras)
                    render_l1 = sum((prediction - target[..., :3]).abs().mean() for prediction, target in zip(predicted_images, train_images)) / len(predicted_images)
                    render_lpips = render_l1.new_zeros() if lpips_function is None else sum(lpips_function(prediction.unsqueeze(0), target[..., :3].unsqueeze(0)).mean() for prediction, target in zip(predicted_images, train_images)) / len(predicted_images)
                    render_loss = config["image_l1_loss_weight"] * render_l1 + config["lpips_loss_weight"] * render_lpips
                else:
                    render_loss = flow_loss.new_zeros(())
                flow_weight, render_weight = flow.loss_mix_weights(time_value, mix_config["schedule"])
                total_loss = flow_weight * flow_loss + render_weight * render_loss
            if config["enable_amp"]:
                scaler.scale(total_loss).backward()
                if config["grad_clip_norm"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            progress.set_postfix(loss=f"{total_loss.item():.4f}", scene=scene["scene_name"], lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if train_step % config["log_interval"] == 0:
                values = {
                    "train/total_loss": total_loss.item(), "train/flow_loss": flow_loss.item(),
                    "train/render_loss": render_loss.item(), "train/time": time_value.item(),
                    "train/lr": optimizer.param_groups[0]["lr"], "train/scene_idx": scene["scene_idx"],
                }
                for key, value in attribute_losses.items():
                    values[f"train/{key}_loss"] = value.item()
                    values[f"train/{key}_weighted"] = weighted_attribute_losses[key].item()
                wandb_log(values, step=train_step)
                logger.info("step=%d scene=%s total=%.6f flow=%.6f render=%.6f", train_step, scene["scene_name"], total_loss.item(), flow_loss.item(), render_loss.item())
            if config["empty_cache_fre"] > 0 and (train_step + 1) % config["empty_cache_fre"] == 0:
                torch.cuda.empty_cache()
            final_step = train_step == config["total_steps"] - 1
            if train_step % config["eval_interval"] == 0 or final_step:
                eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir if final_step else "eval", "" if final_step else f"{train_step:08d}")
                metrics = evaluate_dataset(model, test_dataset, eval_dir)
                for flow_steps, step_metrics in metrics.items():
                    wandb_log({f"eval/{flow_steps}/{key}": value for key, value in step_metrics.items()}, step=train_step)
            if (train_step + 1) % config["save_interval"] == 0:
                torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{train_step:08d}.pth"))
    if FLAGS.only_eval:
        evaluate_dataset(model, test_dataset, os.path.join(FLAGS.output_dir, FLAGS.eval_subdir))
    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))
    if wandb_run is not None:
        wandb_run.finish()


def main(argv):
    del argv
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    training()


if __name__ == "__main__":
    app.run(main)

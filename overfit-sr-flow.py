import json
import os
import random

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from gs_flow_model import GSFlowModel
from gs_path import GSPath
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
    total_steps = gin.REQUIRED,
    pretrain_steps = gin.REQUIRED,
    eval_interval = gin.REQUIRED,
    log_interval = gin.REQUIRED,
    save_interval = gin.REQUIRED,
    log_image_interval = gin.REQUIRED,
    grad_clip_norm = gin.REQUIRED,
    x1_loss_mode="point_mse",
    x1_prediction_type="residual",
    x1_optimization_steps=10000,
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
        "x1_loss_mode": x1_loss_mode,
        "x1_prediction_type": x1_prediction_type,
        "x1_optimization_steps": x1_optimization_steps,
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

def x1_pair_stats(pred_x1, target_x1, flow_keys):
    mse = 0.0
    l1 = 0.0
    for key in flow_keys:
        diff = (pred_x1[key] - target_x1[key]).detach().float()
        mse += float((diff * diff).mean().item())
        l1 += float(diff.abs().mean().item())
    denom = max(len(flow_keys), 1)
    return {"mse": mse / denom, "l1": l1 / denom}


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
    input_eval_images=None,
    input_eval_cameras=None,
    input_eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
    gs_path=None,
    flow_keys=None,
    x1_prediction_type="residual",
):
    model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    predicted_keys = list(getattr(model, "output_features", []))

    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")
    if input_eval_images is not None and len(input_eval_images) == 0:
        raise ValueError("Input-resolution evaluation payload has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)
    if input_eval_images is not None:
        if input_eval_chunk_size is None or input_eval_chunk_size <= 0:
            input_eval_chunk_size = len(input_eval_images)
        input_eval_chunk_size = min(input_eval_chunk_size, len(input_eval_images))

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
        if gs_path is None:
            raise ValueError("evaluate_single_scene requires a GSPath instance")
        if flow_keys is None:
            raise ValueError("evaluate_single_scene requires x1 flow keys")
        t0 = torch.zeros(1, device=device)
        out_gs = gs_path.predict_x1(
            model,
            input_gs,
            t0,
            flow_keys,
            prediction_type=x1_prediction_type,
        )

        pred_preview = []
        gt_preview = []
        height, width = eval_images[0].shape[:2]
        output_resolution = f"{int(width)}x{int(height)}"

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
            cv2.imwrite(os.path.join(output_dir, f"scene_output{output_resolution}.png"), pred_grid)

        if len(gt_preview) > 0:
            gt_grid = cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene_gt{output_resolution}.png"), gt_grid)

        if input_eval_images is not None and input_eval_cameras is not None:
            input_preview = []
            input_gt_preview = []
            height, width = input_eval_images[0].shape[:2]
            input_resolution = f"{int(width)}x{int(height)}"
            input_num_views = len(input_eval_images)
            for start in range(0, input_num_views, input_eval_chunk_size):
                end = min(start + input_eval_chunk_size, input_num_views)

                chunk_images = gpu_utils.move_to_device(input_eval_images[start:end], device)
                chunk_cameras = {
                    key: (value[start:end] if key == "camera_to_worlds" else value)
                    for key, value in input_eval_cameras.items()
                }
                chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)

                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                input_gt_imgs = torch.stack(chunk_images, dim=0)

                if input_gt_imgs.shape[-1] == 4:
                    masks = input_gt_imgs[..., 3].unsqueeze(-1)
                    input_imgs = input_imgs * masks
                    input_gt_imgs = (input_gt_imgs[..., :3] * 255).to(torch.uint8)
                    input_imgs = (input_imgs * 255).to(torch.uint8)
                else:
                    input_gt_imgs = (input_gt_imgs * 255).to(torch.uint8)
                    input_imgs = (input_imgs * 255).to(torch.uint8)

                preview_slots = 9 - len(input_preview)
                if preview_slots > 0:
                    input_preview.extend(
                        [im.cpu().numpy().astype(np.uint8) for im in input_imgs[:preview_slots]]
                    )
                    input_gt_preview.extend(
                        [im.cpu().numpy().astype(np.uint8) for im in input_gt_imgs[:preview_slots]]
                    )
                if len(input_preview) >= 9:
                    break

            if len(input_preview) > 0:
                input_grid = cv2.cvtColor(make_grid(input_preview), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, f"scene_input{input_resolution}.png"), input_grid)

            if len(input_gt_preview) > 0:
                input_gt_grid = cv2.cvtColor(make_grid(input_gt_preview), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, f"scene_gt{input_resolution}.png"), input_gt_grid)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, model.sh_degree)
            gs_utils.export_ply_forviewer(
                gs_params=input_gs,
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
            residual_type = "predicted_x1_minus_input"
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

            safe_scene_name = str(scene_name).replace("/", "_").replace("\\", "_")
            scene_stem = f"{int(scene_idx)}_{safe_scene_name}"
            pt_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "residual_type": residual_type,
                "residual_keys": residual_keys,
                "residuals": to_cpu(residuals),
                "input_gs": to_cpu(input_gs),
                "output_gs": to_cpu(out_gs),
                "cameras": to_cpu(eval_cameras),
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


def evalution(
    model,
    input_gs,
    gt_gs,
    eval_payload,
    eval_images,
    eval_cameras,
    eval_chunk_size,
    input_eval_images,
    input_eval_cameras,
    input_eval_chunk_size,
    output_dir,
    logger,
    gs_path,
    flow_keys,
    x1_prediction_type,
    compare_with_input,
    save_viewer,
    save_residuals,
    output_gt=True,
    log_prefix="Eval",
    input_log_prefix=None,
):
    metrics, metrics_input = evaluate_single_scene(
        model=model,
        input_gs=input_gs,
        gt_gs=gt_gs,
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=output_dir,
        eval_chunk_size=eval_chunk_size,
        input_eval_images=input_eval_images,
        input_eval_cameras=input_eval_cameras,
        input_eval_chunk_size=input_eval_chunk_size,
        compare_with_input=compare_with_input,
        save_viewer=save_viewer,
        save_residuals=save_residuals,
        output_gt=output_gt,
        gs_path=gs_path,
        flow_keys=flow_keys,
        x1_prediction_type=x1_prediction_type,
    )
    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{log_prefix}: {metric_str}")
    if compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"{input_log_prefix or log_prefix + ' input'}: {metric_str}")
    return metrics, metrics_input


def training(
    model,
    dataset,
    scene,
    target_factor_entry,
    train_payload,
    batch_gs,
    target_gs,
    eval_payload,
    eval_images,
    eval_cameras,
    eval_chunk_size,
    input_eval_images,
    input_eval_cameras,
    input_eval_chunk_size,
    train_cfg,
    optimizer,
    scheduler,
    scaler,
    logger,
    gs_path: GSPath,
    flow_keys,
    output_dir,
    compare_with_input,
    save_viewer,
    save_residuals,
):
    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]
    x1_loss_mode = train_cfg["x1_loss_mode"]
    x1_prediction_type = train_cfg["x1_prediction_type"]
    x1_optimization_steps = train_cfg["x1_optimization_steps"]
    device = next(model.parameters()).device

    if x1_loss_mode not in ["point_mse", "render_l1"]:
        raise ValueError("x1_loss_mode must be 'point_mse' or 'render_l1'")
    if x1_prediction_type not in ["residual", "velocity_extrapolation"]:
        raise ValueError(
            "x1_prediction_type must be 'residual' or 'velocity_extrapolation'"
        )
    if x1_optimization_steps <= 0:
        raise ValueError("x1_optimization_steps must be positive")

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([train_payload["images"]], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        train_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
        )
        batch_cameras = gpu_utils.move_to_device([train_payload["cameras"]], device)
        batch_images = gpu_utils.move_to_device([train_payload["images"]], device)

        t0 = torch.zeros(1, device=device)
        teacher_x1 = None
        if x1_loss_mode == "point_mse":
            x0, teacher_x1 = gs_path.optimize_target_discrete(
                batch_gs[0],
                batch_images[0],
                batch_cameras[0],
                flow_keys=flow_keys,
                optim_steps=x1_optimization_steps,
                log_prefix=f"optimized_target_discrete/student_step_{step}",
            )
        else:
            x0 = batch_gs[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            raw_prediction = model(
                batch_normalized_gs=[x0],
                timestep=t0,
            )[0]
            pred_x1 = gs_path.extrapolate_x1(
                x0,
                raw_prediction,
                t0,
                flow_keys,
                prediction_type=x1_prediction_type,
            )
            point_loss = torch.zeros((), device=device)
            image_loss = torch.zeros((), device=device)
            if x1_loss_mode == "point_mse":
                point_loss, _ = gs_path.x1_point_mse_loss(
                    pred_x1,
                    teacher_x1,
                    flow_keys,
                )
                total_loss = point_loss
            else:
                image_loss = gs_path.render_l1_loss(
                    pred_x1,
                    batch_images[0],
                    batch_cameras[0],
                )
                total_loss = image_loss

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
                "loss": f"{total_loss.item():.4f}",
                "point": f"{point_loss.item():.4f}",
                "image": f"{image_loss.item():.4f}",
                "mode": x1_loss_mode,
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            endpoint_str = ""
            if teacher_x1 is not None:
                endpoint_stats = x1_pair_stats(pred_x1, teacher_x1, flow_keys)
                endpoint_str = (
                    f" x1_mse={endpoint_stats['mse']:.6f}"
                    f" x1_l1={endpoint_stats['l1']:.6f}"
                )
            logger.info(
                f"step={step} loss_mode={x1_loss_mode} "
                f"prediction_type={x1_prediction_type} total={total_loss.item():.6f} "
                f"point_mse={point_loss.item():.6f} render_l1={image_loss.item():.6f}"
                f"{endpoint_str} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )

        if step % log_image_interval == 0:
            with torch.no_grad():
                preview_gs = gs_path.predict_x1(
                    model,
                    batch_gs[0],
                    t0,
                    flow_keys,
                    prediction_type=x1_prediction_type,
                )
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(preview_gs, batch_cameras[0])
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            evalution(
                model=model,
                input_gs=batch_gs[0],
                gt_gs=target_gs,
                eval_payload=eval_payload,
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                eval_chunk_size=eval_chunk_size,
                input_eval_images=input_eval_images,
                input_eval_cameras=input_eval_cameras,
                input_eval_chunk_size=input_eval_chunk_size,
                output_dir=os.path.join(output_dir, "eval", f"{step:08d}"),
                logger=logger,
                gs_path=gs_path,
                flow_keys=flow_keys,
                x1_prediction_type=x1_prediction_type,
                compare_with_input=compare_with_input,
                save_viewer=save_viewer,
                save_residuals=save_residuals,
                output_gt=(step == 0),
                log_prefix=f"Eval step {step}",
                input_log_prefix=f"Eval input step {step}",
            )

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", "model_last.pth"))
    return evalution(
        model=model,
        input_gs=batch_gs[0],
        gt_gs=target_gs,
        eval_payload=eval_payload,
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        eval_chunk_size=eval_chunk_size,
        input_eval_images=input_eval_images,
        input_eval_cameras=input_eval_cameras,
        input_eval_chunk_size=input_eval_chunk_size,
        output_dir=os.path.join(output_dir, FLAGS.eval_subdir),
        logger=logger,
        gs_path=gs_path,
        flow_keys=flow_keys,
        x1_prediction_type=x1_prediction_type,
        compare_with_input=compare_with_input,
        save_viewer=save_viewer,
        save_residuals=save_residuals,
        output_gt=True,
        log_prefix="Final eval",
        input_log_prefix="Final eval input",
    )


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training_config(output_dir=FLAGS.output_dir)
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset: SplatFactoMultiLevelDataset = None
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
    input_eval_payload = _build_split_payload(
        dataset, scene["idx"], scene["scene_name"], input_factor_entry, split="test"
    )
    if len(input_eval_payload["images"]) == 0:
        input_eval_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], input_factor_entry, split="train"
        )

    batch_gs = gpu_utils.move_to_device([input_factor_entry["gs_params"]], device)
    target_gs = target_factor_entry["gs_params"]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)
    input_eval_images = input_eval_payload["images"]
    input_eval_cameras = input_eval_payload["cameras"]
    input_eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(input_eval_images)
    if input_eval_chunk_size <= 0:
        input_eval_chunk_size = len(input_eval_images)

    model = GSFlowModel().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["enable_amp"])
    gs_path = GSPath(
        path_mode=GSPath.MODE_OPTIMIZED_TARGET_DISCRETE,
        flow_num_timesteps=1,
        flow_segment_steps=train_cfg["x1_optimization_steps"],
        logger=logger,
    )
    flow_keys = gs_path.flow_float_keys(batch_gs[0], None, model.output_features)

    print(
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} "
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"input_factor={INPUT_FACTOR} target_factor={TARGET_FACTOR} "
        f"x1_loss={train_cfg['x1_loss_mode']} "
        f"x1_prediction={train_cfg['x1_prediction_type']} "
        f"x1_optimization_steps={train_cfg['x1_optimization_steps']}"
    )

    training(
        model=model,
        dataset=dataset,
        scene=scene,
        target_factor_entry=target_factor_entry,
        train_payload=train_payload,
        batch_gs=batch_gs,
        target_gs=target_gs,
        eval_payload=eval_payload,
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        eval_chunk_size=eval_chunk_size,
        input_eval_images=input_eval_images,
        input_eval_cameras=input_eval_cameras,
        input_eval_chunk_size=input_eval_chunk_size,
        train_cfg=train_cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger,
        gs_path=gs_path,
        flow_keys=flow_keys,
        output_dir=FLAGS.output_dir,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
    )


app.run(main)

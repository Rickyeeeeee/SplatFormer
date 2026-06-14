import json
import os
import random

import cv2
import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_3DGSoptimizer, build_optimizer, build_scheduler


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

DEFAULT_GS_OPT_LR_DICT = {
    'base': 1e-3,
    'features_dc': 1e-3,
    'features_rest': 5e-5,
    'scales': 1e-3,
    'opacities': 1e-3,
    'quats': 1e-3,
    'means': 1e-3,
}


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
    gs_optim_steps=1000,
    eval_interval = gin.REQUIRED,
    log_interval = gin.REQUIRED,
    save_interval = gin.REQUIRED,
    log_image_interval = gin.REQUIRED,
    grad_clip_norm = gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0.0,
    point_loss_weight=1.0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
):
    return {
        "output_dir": output_dir,
        "total_steps": total_steps,
        "pretrain_steps": pretrain_steps,
        "gs_optim_steps": gs_optim_steps,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "log_image_interval": log_image_interval,
        "grad_clip_norm": grad_clip_norm,
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "point_loss_weight": point_loss_weight,
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


def _gin_query_parameter(name, default):
    try:
        return gin.query_parameter(name)
    except Exception:
        return default


def _make_optimizable_gs(gs_params, device):
    optimizable_gs = {}
    for key, value in gs_params.items():
        optimizable_gs[key] = value.detach().clone().to(device).float().requires_grad_(True)
    return optimizable_gs


def _detach_gs(gs_params):
    return {key: value.detach() for key, value in gs_params.items()}


def _normalize_quats_(gs_params):
    if "quats" not in gs_params:
        return
    with torch.no_grad():
        norms = gs_params["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gs_params["quats"].div_(norms)


def _build_gs_optimizer_and_scheduler(gs_params, total_steps):
    lr_dict = _gin_query_parameter("pretrain/build_optimizer.lr_dict", DEFAULT_GS_OPT_LR_DICT)
    optimizer_type = _gin_query_parameter("pretrain/build_optimizer.optimizer_type", "adam")
    optimizer_params = _gin_query_parameter("pretrain/build_optimizer.optimizer_params", {"eps": 1e-15})
    schedule = _gin_query_parameter("pretrain/build_scheduler.schedule", "constant")
    warmup_step = _gin_query_parameter("pretrain/build_scheduler.warmup_step", 0)

    optimizer = build_3DGSoptimizer(
        gs_params,
        lr_dict=lr_dict,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
    )
    scheduler = build_scheduler(
        optimizer,
        schedule=schedule,
        total_step=max(int(total_steps), 1),
        warmup_step=warmup_step,
    )
    return optimizer, scheduler


def _compute_image_loss(
    gs_params,
    cameras,
    images,
    image_l1_loss_weight,
    lpips_loss_weight,
    lpips_loss_func,
):
    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_params, cameras)
    if len(pred_imgs) == 0:
        raise ValueError("No rendered images produced for GS optimization")

    image_l1 = gs_params["means"].new_tensor(0.0)
    lpips_loss = gs_params["means"].new_tensor(0.0)
    train_psnr = gs_params["means"].new_tensor(0.0)

    for pred_img, gt_img in zip(pred_imgs, images):
        gt_rgb = gt_img[..., :3]
        image_l1 = image_l1 + (pred_img - gt_rgb).abs().mean()
        train_psnr = train_psnr + psnr(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()
        if lpips_loss_func is not None:
            lpips_loss = lpips_loss + lpips_loss_func(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()

    num_images = len(pred_imgs)
    image_l1 = image_l1 / num_images * image_l1_loss_weight
    train_psnr = train_psnr / num_images
    total_loss = image_l1
    if lpips_loss_func is not None:
        lpips_loss = lpips_loss / num_images * lpips_loss_weight
        total_loss = total_loss + lpips_loss

    return total_loss, image_l1, lpips_loss, train_psnr, pred_imgs


def optimize_input_gs_to_images(
    dataset,
    scene,
    input_gs,
    target_factor_entry,
    output_dir,
    device,
    train_cfg,
    logger,
    lpips_loss_func,
):
    gs_optim_steps = int(train_cfg["gs_optim_steps"])
    optimized_gs = _make_optimizable_gs(input_gs, device)
    _normalize_quats_(optimized_gs)

    if gs_optim_steps <= 0:
        logger.info("Skipping GS optimization because training.gs_optim_steps <= 0")
        return _detach_gs(optimized_gs)

    optimizer, scheduler = _build_gs_optimizer_and_scheduler(optimized_gs, gs_optim_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["enable_amp"])
    gs_train_dir = os.path.join(output_dir, "gs_optim")
    os.makedirs(gs_train_dir, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(gs_optim_steps), desc="Optimizing input GS")
    for step in pbar:
        train_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
        )
        cameras = gpu_utils.move_to_device(train_payload["cameras"], device)
        images = gpu_utils.move_to_device(train_payload["images"], device)

        with torch.cuda.amp.autocast(enabled=train_cfg["enable_amp"]):
            total_loss, image_l1, lpips_loss, train_psnr, pred_imgs = _compute_image_loss(
                optimized_gs,
                cameras,
                images,
                train_cfg["image_l1_loss_weight"],
                train_cfg["lpips_loss_weight"],
                lpips_loss_func,
            )

        if train_cfg["enable_amp"]:
            scaler.scale(total_loss).backward()
            if train_cfg["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(optimized_gs.values()), train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if train_cfg["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(list(optimized_gs.values()), train_cfg["grad_clip_norm"])
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        _normalize_quats_(optimized_gs)

        lpips_value = lpips_loss.item() if lpips_loss_func is not None else 0.0
        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{image_l1.item():.4f}",
                "lpips": f"{lpips_value:.4f}",
                "psnr": f"{train_psnr.item():.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if train_cfg["empty_cache_fre"] > 0 and (step + 1) % train_cfg["empty_cache_fre"] == 0:
            torch.cuda.empty_cache()

        if step % train_cfg["log_interval"] == 0:
            log_msg = (
                f"gs_step={step} total={total_loss.item():.6f} "
                f"l1={image_l1.item():.6f} psnr={train_psnr.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
            if lpips_loss_func is not None:
                log_msg += f" lpips={lpips_loss.item():.6f}"
            logger.info(log_msg)

        if step % train_cfg["log_image_interval"] == 0:
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(gs_train_dir, f"{step:08d}_pred.png"), pred_grid)

    detached_gs = _detach_gs(optimized_gs)
    torch.save(_to_cpu(detached_gs), os.path.join(gs_train_dir, "optimized_gs.pt"))
    return detached_gs


def _compute_point_mse_loss(pred_gs, target_gs, loss_keys, point_loss_weight):
    terms = []
    metrics = {}
    for key in loss_keys:
        if key not in pred_gs or key not in target_gs:
            continue
        pred = pred_gs[key]
        target = target_gs[key].to(device=pred.device, dtype=pred.dtype)
        if pred.shape != target.shape:
            raise ValueError(f"Point loss shape mismatch for {key}: pred={pred.shape}, target={target.shape}")
        if pred.numel() == 0:
            continue
        value = F.mse_loss(pred.float(), target.float())
        terms.append(value)
        metrics[f"point_mse/{key}"] = value.detach()

    if len(terms) == 0:
        raise ValueError("No matching non-empty GS fields available for point MSE loss")

    point_mse = torch.stack(terms).mean()
    metrics["point_mse"] = point_mse.detach()
    return point_mse * point_loss_weight, metrics


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
            residual_type = "out_minus_input"
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
                "input_gs": _to_cpu(input_gs),
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


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset: SplatFactoMultiLevelDataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
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

    batch_gs = gpu_utils.move_to_device([input_factor_entry["gs_params"]], device)
    batch_scene_idx = [scene["idx"]]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

    model = FeaturePredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
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
    image_l1_loss_weight = train_cfg["image_l1_loss_weight"]
    lpips_loss_weight = train_cfg["lpips_loss_weight"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if lpips_loss_weight > 0 else None

    print(
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} "
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"input_factor={INPUT_FACTOR} target_factor={TARGET_FACTOR}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([train_payload["images"]], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    optimized_gs = optimize_input_gs_to_images(
        dataset=dataset,
        scene=scene,
        input_gs=batch_gs[0],
        target_factor_entry=target_factor_entry,
        output_dir=FLAGS.output_dir,
        device=device,
        train_cfg=train_cfg,
        logger=logger,
        lpips_loss_func=lpips_loss_func,
    )

    log_cameras = gpu_utils.move_to_device(train_payload["cameras"], device)
    with torch.no_grad():
        optimized_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(optimized_gs, log_cameras)
    optimized_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in optimized_imgs]
    optimized_grid = cv2.cvtColor(make_grid(optimized_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_optimized_gs.png"), optimized_grid)

    point_loss_keys = list(model.output_features)
    point_loss_weight = train_cfg["point_loss_weight"]
    logger.info(
        "Training FeaturePredictor with point MSE against optimized input GS fields: "
        + ", ".join(point_loss_keys)
    )

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps), desc="Training point MSE")
    for step in pbar:
        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_batch_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)
            out_gs = out_batch_gs[0]
            total_loss, point_metrics = _compute_point_mse_loss(
                out_gs,
                optimized_gs,
                point_loss_keys,
                point_loss_weight,
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

        point_mse = point_metrics["point_mse"]
        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.6f}",
                "mse": f"{point_mse.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            field_metrics = " ".join(
                f"{key}={value.item():.6f}"
                for key, value in point_metrics.items()
                if key != "point_mse"
            )
            log_msg = (
                f"step={step} total={total_loss.item():.6f} "
                f"point_mse={point_mse.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {field_metrics}"
            )
            logger.info(log_msg.rstrip())

        if step % log_image_interval == 0:
            model.eval()
            with torch.no_grad():
                preview_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)[0]
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(preview_gs, log_cameras)
            model.train()
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=batch_gs[0],
                gt_gs=target_factor_entry["gs_params"],
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
        gt_gs=target_factor_entry["gs_params"],
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
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")


app.run(main)

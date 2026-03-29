import json
import os
import random

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS import SplatfactoDataset
from dataset.GS_level import SplatfactoLevelDataset
from dataset.Loader import GS_collate_fn, build_testloader, build_trainloader
from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_string("dataset_type", "splatfacto", "Dataset type: splatfacto or level")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


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
    total_steps: gin.REQUIRED = gin.REQUIRED,
    pretrain_steps: gin.REQUIRED = gin.REQUIRED,
    eval_interval: gin.REQUIRED = gin.REQUIRED,
    log_interval: gin.REQUIRED = gin.REQUIRED,
    save_interval: gin.REQUIRED = gin.REQUIRED,
    log_image_interval: gin.REQUIRED = gin.REQUIRED,
    grad_clip_norm: gin.REQUIRED = gin.REQUIRED,
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


def _build_dataset(dataset_type):
    with gin.config_scope("train_dataset"):
        if dataset_type == "level":
            return SplatfactoLevelDataset()
        return SplatfactoDataset()


def _scene_name_from_dataset(dataset, idx):
    nerfstudio_dir = dataset.folders[idx][0]
    return os.path.basename(os.path.dirname(nerfstudio_dir))


def _find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for idx in range(len(dataset.folders)):
        if _scene_name_from_dataset(dataset, idx) == scene_name:
            return idx
    return 0


def _build_split_payload(dataset, scene, split):
    meta = scene["meta"]
    train_imgs_path = scene["train_imgs_path"]
    test_imgs_path = scene["test_imgs_path"]
    train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
    test_imgs_name = [os.path.basename(path) for path in test_imgs_path]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    if split == "test":
        img_paths = test_imgs_path
        images_name = test_imgs_name
        camera_to_worlds = meta["test_camera_to_worlds"]
        images = [dataset.read_image(path, background=background) for path in img_paths]
    else:
        total_train_num = len(meta["train_camera_to_worlds"])
        total_test_num = len(meta["test_camera_to_worlds"])

        sample_test = np.random.rand(dataset.image_per_scene) < dataset.sample_ratio_test
        sample_test_num = min(np.sum(sample_test), total_test_num)
        sample_train_num = dataset.image_per_scene - sample_test_num
        sample_train_num = min(sample_train_num, total_train_num)

        images = []
        images_name = []
        camera_to_worlds = []
        if sample_train_num > 0:
            train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
            images.extend([dataset.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
            images_name.extend([train_imgs_name[i] for i in train_cam_ids])
            camera_to_worlds.append(meta["train_camera_to_worlds"][train_cam_ids])
        if sample_test_num > 0:
            test_cam_ids = np.random.permutation(total_test_num)[:sample_test_num]
            images.extend([dataset.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
            images_name.extend([test_imgs_name[i] for i in test_cam_ids])
            camera_to_worlds.append(meta["test_camera_to_worlds"][test_cam_ids])
        camera_to_worlds = torch.concatenate(camera_to_worlds, axis=0)

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
        "gs_params": scene["gs_params"],
        "images": images,
        "images_name": images_name,
        "cameras": cameras,
        "scene_idx": scene["idx"],
        "scene_name": scene["scene_name"],
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
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
):
    model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    predicted_keys = list(getattr(model, "output_features", []))

    os.makedirs(output_dir, exist_ok=True)
    residual_dir = None
    if save_residuals:
        residual_dir = os.path.join(output_dir, "residuals")
        os.makedirs(residual_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = model(batch_normalized_gs=[input_gs], batch_scene_idx=[scene_idx])[0]

        pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, eval_cameras)
        pred_imgs = torch.stack(pred_imgs, dim=0)
        gt_imgs = torch.stack(eval_images, dim=0)

        if gt_imgs.shape[-1] == 4:
            masks = gt_imgs[..., 3].unsqueeze(-1)
            pred_imgs = pred_imgs * masks
            gt_imgs = (gt_imgs[..., :3] * 255).to(torch.uint8)
            pred_imgs = (pred_imgs * 255).to(torch.uint8)
        else:
            masks = None
            gt_imgs = (gt_imgs * 255).to(torch.uint8)
            pred_imgs = (pred_imgs * 255).to(torch.uint8)

        pred_np = [im.cpu().numpy().astype(np.uint8) for im in pred_imgs]
        pred_grid = cv2.cvtColor(make_grid(pred_np), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt:
            gt_np = [im.cpu().numpy().astype(np.uint8) for im in gt_imgs]
            gt_grid = cv2.cvtColor(make_grid(gt_np), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        metric_computer.update(pred_imgs, gt_imgs, name=f"{scene_idx}")

        pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
        os.makedirs(pred_single_dir, exist_ok=True)
        for name, pred_img in zip(image_names, pred_imgs):
            pred_img = pred_img.cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(pred_single_dir, name), pred_img[:, :, ::-1])

        if compare_with_input:
            input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, eval_cameras)
            input_imgs = torch.stack(input_imgs, dim=0)
            if masks is not None:
                input_imgs = input_imgs * masks
                input_imgs = (input_imgs * 255).to(torch.uint8)
            else:
                input_imgs = (input_imgs * 255).to(torch.uint8)
            metric_computer_input.update(input_imgs, gt_imgs, name=f"{scene_idx}")

            compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
            os.makedirs(compare_dir, exist_ok=True)
            for ii, (gt_img, input_img, pred_img) in enumerate(zip(gt_imgs, input_imgs, pred_imgs)):
                gt_img = gt_img.cpu().numpy().astype(np.uint8)
                input_img = input_img.cpu().numpy().astype(np.uint8)
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1)
                cv2.imwrite(os.path.join(compare_dir, f"{ii:02d}.png"), cmp_img[:, :, ::-1])

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

    dataset: SplatfactoDataset = _build_dataset(FLAGS.dataset_type)
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)

    train_payload = _build_split_payload(dataset, scene, split="train")
    eval_payload = _build_split_payload(dataset, scene, split="test")
    if len(eval_payload["images"]) == 0:
        eval_payload = _build_split_payload(dataset, scene, split="train")

    batch_gs = gpu_utils.move_to_device([scene["gs_params"]], device)
    batch_scene_idx = [scene["idx"]]

    eval_images = gpu_utils.move_to_device(eval_payload["images"], device)
    eval_cameras = gpu_utils.move_to_device(eval_payload["cameras"], device)

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

    logger.info(
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} "
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"gaussians={scene['gs_params']['means'].shape[0]}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([train_payload["images"]], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        train_payload = _build_split_payload(dataset, scene, split="train")
        batch_cameras = gpu_utils.move_to_device([train_payload["cameras"]], device)
        batch_images = gpu_utils.move_to_device([train_payload["images"]], device)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_batch_gs = model(batch_normalized_gs=batch_gs, batch_scene_idx=batch_scene_idx)
            out_gs = out_batch_gs[0]
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, batch_cameras[0])

            image_l1 = 0
            lpips_loss = 0
            train_psnr = 0
            num_images = len(pred_imgs)
            for pred_img, gt_img in zip(pred_imgs, batch_images[0]):
                gt_rgb = gt_img[..., :3]
                image_l1 += (pred_img - gt_rgb).abs().mean()
                train_psnr += psnr(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()
                if lpips_loss_func is not None:
                    lpips_loss += lpips_loss_func(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()

            image_l1 = image_l1 / num_images * image_l1_loss_weight
            train_psnr = train_psnr / num_images
            total_loss = image_l1
            if lpips_loss_func is not None:
                lpips_loss = lpips_loss / num_images * lpips_loss_weight
                total_loss = total_loss + lpips_loss

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

        lpips_value = lpips_loss.item() if lpips_loss_func is not None else 0.0
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "l1": f"{image_l1.item():.4f}",
            "lpips": f"{lpips_value:.4f}",
            "psnr": f"{train_psnr.item():.2f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            log_msg = (
                f"step={step} total={total_loss.item():.6f} "
                f"l1={image_l1.item():.6f} psnr={train_psnr.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
            if lpips_loss_func is not None:
                log_msg += f" lpips={lpips_loss.item():.6f}"
            logger.info(log_msg)

        if step % log_image_interval == 0:
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=batch_gs[0],
                scene_idx=eval_payload["scene_idx"],
                scene_name=eval_payload["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=eval_payload["images_name"],
                output_dir=eval_dir,
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
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=final_eval_dir,
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

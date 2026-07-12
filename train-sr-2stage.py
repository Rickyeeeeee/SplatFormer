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

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_predictor import ALL_FEATURES, FeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_densify_utils import build_densified_input_gs, convert_gs_to_target_frame


flags.DEFINE_string("output_dir", "output_sr_2stage", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("compare_with_input", True, "Compare predictions with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("use_wandb", True, "Log training and evaluation metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as the model-input source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as training target")
flags.DEFINE_enum(
    "means_source",
    "high_res",
    ["high_res", "low_res", "predicted", "splatformer"],
    "Source/model path: high-res target means, unchanged low-res means, residual means predictor, "
    "or direct full-feature SplatFormer.",
)
flags.DEFINE_enum("alignment", "emd", ["emd", "nearest", "none"], "Interpolated-to-target alignment method")
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

WANDB_EVAL_IMAGE_SCENE = "3e288ee8aced4a0797e66d53536112b1"
MEANS_FEATURES = ["means"]
ATTRIBUTE_FEATURES = ["features_dc", "features_rest", "opacities", "scales", "quats"]
FULL_FEATURES = list(ALL_FEATURES)


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


def _default_wandb_name(output_dir):
    output_dir = output_dir.rstrip("/")
    parts = Path(output_dir).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else output_dir


def _init_wandb(output_dir):
    if not FLAGS.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

    return wandb.init(
        project=FLAGS.wandb_project,
        dir=FLAGS.wandb_dir,
        name=FLAGS.wandb_name or _default_wandb_name(output_dir),
        config={
            "output_dir": output_dir,
            "eval_subdir": FLAGS.eval_subdir,
            "compare_with_input": FLAGS.compare_with_input,
            "save_viewer": FLAGS.save_viewer,
            "input_factor": FLAGS.input_factor,
            "target_factor": FLAGS.target_factor,
            "means_source": FLAGS.means_source,
            "alignment": FLAGS.alignment,
            "attribute_init": FLAGS.attribute_init,
            "gin_config": gin.operative_config_str(),
        },
    )


def _wandb_log(data, step=None):
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def _build_dataset(scope):
    with gin.config_scope(scope):
        dataset = SplatFactoMultiLevelDataset()
    required_factors = {FLAGS.input_factor, FLAGS.target_factor}
    missing = sorted(required_factors - set(dataset.factors))
    if missing:
        raise ValueError(
            f"{scope} dataset factors {sorted(dataset.factors)} do not include required factors {missing}"
        )
    return dataset


def _normalize_skipped_scene(split, skipped_scene):
    skip_reason = skipped_scene.get("skip_reason", skipped_scene.get("reason"))
    exception_reason = skipped_scene.get("exception_reason")
    return {
        "split": split,
        "scene_idx": skipped_scene.get("scene_idx"),
        "scene_name": skipped_scene.get("scene_name"),
        "skip_reason": skip_reason,
        "exception_reason": exception_reason,
        "exception_type": skipped_scene.get("exception_type"),
        "nerfstudio_dir": skipped_scene.get("nerfstudio_dir"),
        "colmap_dir": skipped_scene.get("colmap_dir"),
    }


def _write_skipped_scenes(output_dir, datasets_by_split, extra_skipped_scenes=None):
    skipped_scenes = []
    for split, dataset in datasets_by_split.items():
        for skipped_scene in getattr(dataset, "skipped_scenes", []):
            skipped_scenes.append(_normalize_skipped_scene(split, skipped_scene))

    if extra_skipped_scenes is not None:
        skipped_scenes.extend(extra_skipped_scenes)

    with open(os.path.join(output_dir, "skipped_scenes.json"), "w") as f:
        json.dump(skipped_scenes, f, indent=2)
    return skipped_scenes


def _next_train_batch(train_iter, train_dataset):
    try:
        return train_iter, next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataset)
        return train_iter, next(train_iter)


def _prepare_model_input(input_factor_entry, target_factor_entry, device):
    if FLAGS.means_source == "low_res":
        input_gs = gpu_utils.move_to_device(input_factor_entry["gs_params"], device)
        return convert_gs_to_target_frame(
            input_gs,
            input_factor_entry["scaler"],
            target_factor_entry["scaler"],
        )
    return build_densified_input_gs(
        input_factor_dict=input_factor_entry,
        target_factor_dict=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
    )


def _print_densify_scene_context(prefix, scene_idx, scene_name, input_factor_entry, target_factor_entry):
    input_count = input_factor_entry["gs_params"]["means"].shape[0]
    target_count = target_factor_entry["gs_params"]["means"].shape[0]
    print(
        f"[{prefix}] scene_idx={scene_idx} scene_name={scene_name} "
        f"input_factor={FLAGS.input_factor} input_gaussians={input_count} "
        f"target_factor={FLAGS.target_factor} target_gaussians={target_count}",
        flush=True,
    )


def _clone_gs(gs):
    return {key: value.clone() for key, value in gs.items()}


def _replace_means(gs, means):
    out_gs = _clone_gs(gs)
    out_gs["means"] = means.to(device=gs["means"].device, dtype=gs["means"].dtype)
    return out_gs


def _bind_feature_predictor(output_features, output_features_type="res", input_features=None):
    with gin.unlock_config():
        if input_features is not None:
            gin.bind_parameter("FeaturePredictor.input_features", list(input_features))
        gin.bind_parameter("FeaturePredictor.output_features", list(output_features))
        gin.bind_parameter("FeaturePredictor.output_features_type", output_features_type)


def _load_resume_checkpoint(model, checkpoint_key, logger):
    if model.resume_ckpt is None:
        return
    checkpoint = torch.load(model.resume_ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    logger.info(f"Loaded {checkpoint_key} checkpoint from {model.resume_ckpt}")


def _build_feature_predictor(output_features, device, logger, checkpoint_key, input_features=None):
    _bind_feature_predictor(output_features, output_features_type="res", input_features=input_features)
    model = FeaturePredictor().to(device)
    _load_resume_checkpoint(model, checkpoint_key, logger)
    return model


def _stage2_input_gs(input_gs, target_gs, means_model, means_source, scene_idx):
    if means_source == "high_res":
        return _replace_means(input_gs, target_gs["means"])
    if means_source == "low_res":
        return input_gs
    if means_source == "predicted":
        if means_model is None:
            raise ValueError("means_model is required when --means_source=predicted")
        return means_model(batch_normalized_gs=[input_gs], batch_scene_idx=[scene_idx])[0]
    if means_source == "splatformer":
        return input_gs
    raise ValueError(f"Unsupported means_source={means_source}")


def _forward_two_stage(
    attribute_model,
    input_gs,
    scene_idx,
    target_gs,
    means_source,
    means_model=None,
    return_stage1=False,
):
    stage1_gs = _stage2_input_gs(input_gs, target_gs, means_model, means_source, scene_idx)
    stage2_gs = attribute_model(batch_normalized_gs=[stage1_gs], batch_scene_idx=[scene_idx])[0]
    if return_stage1:
        return stage2_gs, stage1_gs
    return stage2_gs


def _active_models(attribute_model, means_model):
    models = [attribute_model]
    if means_model is not None:
        models.insert(0, means_model)
    return models


def _trainable_parameters(models):
    return [param for model in models for param in model.parameters() if param.requires_grad]


def _checkpoint_payload(attribute_model, means_model, means_source):
    payload = {
        "means_source": means_source,
        "attribute_model": attribute_model.state_dict(),
    }
    if means_model is not None:
        payload["means_model"] = means_model.state_dict()
    return payload


def _compute_render_loss(pred_imgs, batch_images, lpips_loss_func, image_l1_loss_weight, lpips_loss_weight):
    image_l1 = 0
    lpips_loss = 0
    train_psnr = 0
    num_images = len(pred_imgs)
    if num_images == 0:
        raise ValueError("Cannot compute render loss on zero images")

    for pred_img, gt_img in zip(pred_imgs, batch_images):
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
    return total_loss, image_l1, lpips_loss, train_psnr


def _image_png_name(image_names, image_id):
    if image_id < len(image_names):
        image_name = os.path.basename(str(image_names[image_id]))
        stem, ext = os.path.splitext(image_name)
        if ext.lower() == ".png":
            return image_name
        if stem:
            return f"{stem}.png"
    return f"{image_id:04d}.png"


def _metric_counts(metric_computer):
    return {metric: len(values) for metric, values in metric_computer.results.items()}


def _append_image_metric_records(records, metric_computer, previous_counts, start, image_names):
    metric_values = {}
    for metric, values in metric_computer.results.items():
        new_values = values[previous_counts[metric]:]
        metric_values[metric] = torch.cat([value.reshape(-1) for value in new_values]).detach().cpu().tolist()

    num_records = len(metric_values["psnr"])
    for offset in range(num_records):
        image_id = start + offset
        records.append(
            {
                "image_id": image_id,
                "image_name": _image_png_name(image_names, image_id),
                "psnr": float(metric_values["psnr"][offset]),
                "ssim": float(metric_values["ssim"][offset]),
                "lpips": float(metric_values["lpips"][offset]),
            }
        )


def _models_train(attribute_model, means_model):
    attribute_model.train()
    if means_model is not None:
        means_model.train()


def _models_eval(attribute_model, means_model):
    attribute_model.eval()
    if means_model is not None:
        means_model.eval()


def evaluate_single_scene(
    attribute_model,
    input_gs,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=True,
    save_viewer=False,
    output_gt=True,
    wandb_step=None,
    target_gs_for_means=None,
    low_res_gt_gs=None,
    evaluate_baselines=False,
    means_source="high_res",
    means_model=None,
):
    if target_gs_for_means is None:
        raise ValueError("target_gs_for_means is required for two-stage evaluation")
    _models_eval(attribute_model, means_model)
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if evaluate_baselines else None
    metric_computer_gt_low_res = MetricComputer() if evaluate_baselines else None
    metric_computer_gt_high_res = MetricComputer() if evaluate_baselines else None
    device = next(attribute_model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)
    pred_single_dir = os.path.join(output_dir, "pred")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input and evaluate_baselines:
        compare_dir = os.path.join(output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)

    image_metrics = []
    image_metrics_input = []
    image_metrics_gt_low_res = []
    image_metrics_gt_high_res = []

    with torch.no_grad():
        input_gs_device = gpu_utils.move_to_device(input_gs, device)
        target_gs_device = gpu_utils.move_to_device(target_gs_for_means, device)
        gt_gs_device = (
            gpu_utils.move_to_device(gt_gs, device) if evaluate_baselines or save_viewer else None
        )
        low_res_gt_gs_device = (
            gpu_utils.move_to_device(low_res_gt_gs, device) if evaluate_baselines else None
        )
        out_gs, stage1_gs = _forward_two_stage(
            attribute_model=attribute_model,
            input_gs=input_gs_device,
            scene_idx=scene_idx,
            target_gs=target_gs_device,
            means_source=means_source,
            means_model=means_model,
            return_stage1=True,
        )

        pred_preview = []
        gt_preview = []
        compare_preview = []

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

            metric_counts = _metric_counts(metric_computer)
            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)
            _append_image_metric_records(image_metrics, metric_computer, metric_counts, start, image_names)

            if evaluate_baselines:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs_device, chunk_cameras)
                low_res_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(low_res_gt_gs_device, chunk_cameras)
                gt_high_res_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gt_gs_device, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                low_res_imgs = torch.stack(low_res_imgs, dim=0)
                gt_high_res_imgs = torch.stack(gt_high_res_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                    low_res_imgs = low_res_imgs * masks
                    gt_high_res_imgs = gt_high_res_imgs * masks
                input_imgs = (input_imgs * 255).to(torch.uint8)
                low_res_imgs = (low_res_imgs * 255).to(torch.uint8)
                gt_high_res_imgs = (gt_high_res_imgs * 255).to(torch.uint8)

                input_counts = _metric_counts(metric_computer_input)
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)
                _append_image_metric_records(
                    image_metrics_input, metric_computer_input, input_counts, start, image_names
                )
                low_res_counts = _metric_counts(metric_computer_gt_low_res)
                metric_computer_gt_low_res.update(low_res_imgs, gt_imgs, name=chunk_name)
                _append_image_metric_records(
                    image_metrics_gt_low_res, metric_computer_gt_low_res,
                    low_res_counts, start, image_names
                )
                high_res_counts = _metric_counts(metric_computer_gt_high_res)
                metric_computer_gt_high_res.update(gt_high_res_imgs, gt_imgs, name=chunk_name)
                _append_image_metric_records(
                    image_metrics_gt_high_res, metric_computer_gt_high_res,
                    high_res_counts, start, image_names
                )

            for global_idx, pred_img in enumerate(pred_imgs, start=start):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(pred_single_dir, _image_png_name(image_names, global_idx)), pred_img[:, :, ::-1])

            if compare_with_input and evaluate_baselines:
                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    gt_img = gt_img.cpu().numpy().astype(np.uint8)
                    input_img = input_img.cpu().numpy().astype(np.uint8)
                    pred_img = pred_img.cpu().numpy().astype(np.uint8)
                    cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1)
                    if len(compare_preview) < 9:
                        compare_preview.append(cmp_img)
                    cv2.imwrite(
                        os.path.join(compare_dir, _image_png_name(image_names, global_idx)),
                        cmp_img[:, :, ::-1],
                    )

        if len(pred_preview) > 0:
            pred_grid_rgb = gs_utils.make_grid(pred_preview)
            pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid_rgb = gs_utils.make_grid(gt_preview)
            gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        if wandb is not None and wandb.run is not None and scene_name == WANDB_EVAL_IMAGE_SCENE:
            wandb_images = {}
            if len(pred_preview) > 0:
                wandb_images[f"eval_images/{scene_name}/pred_grid"] = wandb.Image(
                    pred_grid_rgb,
                    caption=f"{scene_name} prediction",
                )
            if output_gt and len(gt_preview) > 0:
                wandb_images[f"eval_images/{scene_name}/gt_grid"] = wandb.Image(
                    gt_grid_rgb,
                    caption=f"{scene_name} ground truth",
                )
            if compare_with_input and evaluate_baselines and len(compare_preview) > 0:
                compare_grid = gs_utils.make_grid(compare_preview)
                wandb_images[f"eval_images/{scene_name}/compare_grid"] = wandb.Image(
                    compare_grid,
                    caption=f"{scene_name} GT | input | pred",
                )
            if wandb_images:
                _wandb_log(wandb_images, step=wandb_step)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, attribute_model.sh_degree)
            gs_utils.export_ply_forviewer(
                gs_params=input_gs_device,
                filename=os.path.join(viewerdir, "point_cloud/00_input_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=stage1_gs,
                filename=os.path.join(viewerdir, "point_cloud/01_stage1_output_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(viewerdir, "point_cloud/02_stage2_output_gs.ply"),
            )
            if gt_gs_device is not None:
                gs_utils.export_ply_forviewer(
                    gs_params=gt_gs_device,
                    filename=os.path.join(viewerdir, "point_cloud/03_gt_gs.ply"),
                )

            gs_dir = os.path.join(output_dir, "gs")
            os.makedirs(gs_dir, exist_ok=True)
            if low_res_gt_gs is None:
                raise ValueError("low_res_gt_gs is required when save_viewer=True")
            if low_res_gt_gs_device is None:
                low_res_gt_gs_device = gpu_utils.move_to_device(low_res_gt_gs, device)
            high_res_gt_gs_device = gpu_utils.move_to_device(gt_gs if gt_gs is not None else target_gs_for_means, device)
            gs_utils.export_ply_forviewer(
                gs_params=low_res_gt_gs_device,
                filename=os.path.join(gs_dir, "low_res_gt_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=high_res_gt_gs_device,
                filename=os.path.join(gs_dir, "high_res_gt_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=input_gs_device,
                filename=os.path.join(gs_dir, "input_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(gs_dir, "output_gs.ply"),
            )

    metrics = metric_computer.finalize()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(image_metrics, f, indent=2)
    if evaluate_baselines:
        metrics_input = metric_computer_input.finalize()
        with open(os.path.join(output_dir, "metrics_input.json"), "w") as f:
            json.dump(image_metrics_input, f, indent=2)
        metrics_gt_low_res = metric_computer_gt_low_res.finalize()
        with open(os.path.join(output_dir, "metrics_gt_low_res.json"), "w") as f:
            json.dump(image_metrics_gt_low_res, f, indent=2)
        metrics_gt_high_res = metric_computer_gt_high_res.finalize()
        with open(os.path.join(output_dir, "metrics_gt_high_res.json"), "w") as f:
            json.dump(image_metrics_gt_high_res, f, indent=2)
    else:
        metrics_input = {}
        metrics_gt_low_res = {}
        metrics_gt_high_res = {}

    _models_train(attribute_model, means_model)
    return metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res


def evaluate_dataset(
    attribute_model,
    means_model,
    dataset,
    output_dir,
    compare_with_input=True,
    save_viewer=False,
    output_gt=False,
    wandb_step=None,
    evaluate_baselines=False,
):
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    all_metrics_input = []
    all_metrics_gt_low_res = []
    all_metrics_gt_high_res = []
    scene_average_metrics = []
    eval_skipped_scenes = []
    _write_skipped_scenes(output_dir, {"test": dataset}, eval_skipped_scenes)
    logger = ProcessSafeLogger(os.path.join(output_dir, "eval.log")).get_logger()
    device = next(attribute_model.parameters()).device

    for scene_idx in tqdm(range(len(dataset.folders)), desc="Evaluating"):
        scene_info = dataset.folders[scene_idx]
        scene_name = scene_info["scene_name"]
        try:
            scene = dataset.load_scene(scene_idx)
            input_factor_entry = scene["factor_data"][FLAGS.input_factor]
            target_factor_entry = scene["factor_data"][FLAGS.target_factor]
            eval_images, image_names, eval_cameras = dataset.load_factor_views(target_factor_entry)

            target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
            _print_densify_scene_context(
                "eval_input",
                scene["idx"],
                scene["scene_name"],
                input_factor_entry,
                target_factor_entry,
            )
            input_gs = _prepare_model_input(input_factor_entry, target_factor_entry, device)
            low_res_gt_gs = convert_gs_to_target_frame(
                gpu_utils.move_to_device(input_factor_entry["gs_params"], device),
                input_factor_entry["scaler"],
                target_factor_entry["scaler"],
            )

            scene_output_dir = os.path.join(output_dir, scene["scene_name"])
            metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_single_scene(
                attribute_model=attribute_model,
                means_model=means_model,
                input_gs=input_gs,
                gt_gs=target_gs,
                scene_idx=scene["idx"],
                scene_name=scene["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=image_names,
                output_dir=scene_output_dir,
                eval_chunk_size=len(eval_images),
                compare_with_input=compare_with_input,
                save_viewer=save_viewer,
                output_gt=output_gt,
                wandb_step=wandb_step,
                target_gs_for_means=target_gs,
                low_res_gt_gs=low_res_gt_gs,
                evaluate_baselines=evaluate_baselines,
                means_source=FLAGS.means_source,
            )
        except Exception as exc:
            _models_train(attribute_model, means_model)
            skipped_scene = _normalize_skipped_scene(
                "test",
                {
                    "scene_idx": scene_idx,
                    "scene_name": scene_name,
                    "skip_reason": "eval_exception",
                    "exception_reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "nerfstudio_dir": scene_info.get("factor_paths", {})
                    .get(FLAGS.target_factor, {})
                    .get("nerfstudio_dir"),
                    "colmap_dir": scene_info.get("colmap_dir"),
                },
            )
            eval_skipped_scenes.append(skipped_scene)
            _write_skipped_scenes(output_dir, {"test": dataset}, eval_skipped_scenes)
            logger.exception(f"Skipping eval scene {scene_name} after exception")
            continue

        all_metrics.append(metrics)
        if evaluate_baselines:
            all_metrics_gt_low_res.append(metrics_gt_low_res)
            all_metrics_gt_high_res.append(metrics_gt_high_res)
            all_metrics_input.append(metrics_input)
        scene_metrics = {
            "scene_idx": scene["idx"],
            "scene_name": scene["scene_name"],
            "output_gs": metrics,
        }
        if evaluate_baselines:
            scene_metrics.update({
                "gt_low_res_gs": metrics_gt_low_res,
                "input_gs": metrics_input,
                "gt_high_res_gs": metrics_gt_high_res,
            })
        scene_average_metrics.append(scene_metrics)
        logger.info(
            f"Scene {scene['scene_name']}: "
            + " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        )
        if evaluate_baselines:
            logger.info(f"Scene {scene['scene_name']} GT low-res GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()]
            ))
            logger.info(f"Scene {scene['scene_name']} input GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_input.items()]
            ))
            logger.info(f"Scene {scene['scene_name']} GT high-res GS: " + " ".join(
                [f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()]
            ))

    reduced_metrics = {}
    if len(all_metrics) > 0:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            reduced_metrics[key] = float(np.mean([metrics[key] for metrics in all_metrics]))

    reduced_metrics_input = {}
    if evaluate_baselines and len(all_metrics_input) > 0:
        metric_keys = all_metrics_input[0].keys()
        for key in metric_keys:
            reduced_metrics_input[key] = float(np.mean([metrics[key] for metrics in all_metrics_input]))

    reduced_metrics_gt_low_res = {}
    if evaluate_baselines and len(all_metrics_gt_low_res) > 0:
        metric_keys = all_metrics_gt_low_res[0].keys()
        for key in metric_keys:
            reduced_metrics_gt_low_res[key] = float(np.mean(
                [metrics[key] for metrics in all_metrics_gt_low_res]
            ))

    reduced_metrics_gt_high_res = {}
    if evaluate_baselines and len(all_metrics_gt_high_res) > 0:
        metric_keys = all_metrics_gt_high_res[0].keys()
        for key in metric_keys:
            reduced_metrics_gt_high_res[key] = float(np.mean(
                [metrics[key] for metrics in all_metrics_gt_high_res]
            ))

    with open(os.path.join(output_dir, "scene_average_metrics.json"), "w") as f:
        json.dump(scene_average_metrics, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(reduced_metrics, f, indent=2)
    if evaluate_baselines:
        with open(os.path.join(output_dir, "metrics_input.json"), "w") as f:
            json.dump(reduced_metrics_input, f, indent=2)
        with open(os.path.join(output_dir, "metrics_gt_low_res.json"), "w") as f:
            json.dump(reduced_metrics_gt_low_res, f, indent=2)
        with open(os.path.join(output_dir, "metrics_gt_high_res.json"), "w") as f:
            json.dump(reduced_metrics_gt_high_res, f, indent=2)

    return (
        reduced_metrics, reduced_metrics_input,
        reduced_metrics_gt_low_res, reduced_metrics_gt_high_res,
    )

def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.bind_parameter("training.output_dir", FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    set_seed()

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger()
    wandb_run = _init_wandb(FLAGS.output_dir)
    device = torch.device("cuda")

    train_dataset = _build_dataset("train_dataset")
    test_dataset = _build_dataset("test_dataset")
    skipped_scenes = _write_skipped_scenes(
        FLAGS.output_dir,
        {"train": train_dataset, "test": test_dataset},
    )
    if skipped_scenes:
        logger.info(f"Saved {len(skipped_scenes)} skipped scenes to skipped_scenes.json")

    means_model = None
    if FLAGS.means_source == "predicted":
        means_model = _build_feature_predictor(MEANS_FEATURES, device, logger, "means_model")
    if FLAGS.means_source in ["splatformer", "low_res"]:
        attribute_model = _build_feature_predictor(
            FULL_FEATURES,
            device,
            logger,
            "attribute_model",
            input_features=FULL_FEATURES,
        )
    else:
        attribute_model = _build_feature_predictor(ATTRIBUTE_FEATURES, device, logger, "attribute_model")

    if FLAGS.only_eval:
        _models_eval(attribute_model, means_model)
    else:
        _models_train(attribute_model, means_model)

    active_models = _active_models(attribute_model, means_model)
    optimizers = []
    schedulers = []
    with gin.config_scope("train2D"):
        if means_model is not None:
            means_optimizer = build_optimizer(means_model)
            means_scheduler = build_scheduler(means_optimizer)
            optimizers.append(means_optimizer)
            schedulers.append(means_scheduler)
        attribute_optimizer = build_optimizer(attribute_model)
        attribute_scheduler = build_scheduler(attribute_optimizer)
        optimizers.append(attribute_optimizer)
        schedulers.append(attribute_scheduler)

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    training_brief = (
        f"Train SR 2Stage input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor}\n"
        f"train_scenes={len(train_dataset.folders)} test_scenes={len(test_dataset.folders)}\n"
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init}\n"
        f"means_source={FLAGS.means_source}\n"
        f"means_model_input_features={','.join(means_model.input_features) if means_model is not None else 'none'}\n"
        f"means_model_output_features={','.join(means_model.output_features) if means_model is not None else 'none'}\n"
        f"attribute_model_input_features={','.join(attribute_model.input_features)}\n"
        f"attribute_model_output_features={','.join(attribute_model.output_features)}"
    )
    print(training_brief)
    logger.info(training_brief)

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
    _ = train_cfg["pretrain_steps"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if lpips_loss_weight > 0 else None

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    if not FLAGS.only_eval:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        train_iter = iter(train_dataset)
        pbar = tqdm(range(resume_from_step, total_steps), desc="Training")
        for step in pbar:
            train_iter, batch = _next_train_batch(train_iter, train_dataset)
            input_factor_entry = batch["multilevel"][FLAGS.input_factor]
            target_factor_entry = batch["multilevel"][FLAGS.target_factor]

            target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
            # _print_densify_scene_context(
            #     "train_densify",
            #     batch["scene_idx"],
            #     batch["scene_name"],
            #     input_factor_entry,
            #     target_factor_entry,
            # )
            model_input_gs = _prepare_model_input(input_factor_entry, target_factor_entry, device)
            batch_scene_idx = [batch["scene_idx"]]
            batch_cameras = gpu_utils.move_to_device(target_factor_entry["cameras"], device)
            batch_images = gpu_utils.move_to_device(target_factor_entry["images"], device)

            with torch.cuda.amp.autocast(enabled=enable_amp):
                out_gs = _forward_two_stage(
                    attribute_model=attribute_model,
                    input_gs=model_input_gs,
                    scene_idx=batch_scene_idx[0],
                    target_gs=target_gs,
                    means_source=FLAGS.means_source,
                    means_model=means_model,
                )
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, batch_cameras)
                total_loss, image_l1, lpips_loss, train_psnr = _compute_render_loss(
                    pred_imgs,
                    batch_images,
                    lpips_loss_func,
                    image_l1_loss_weight,
                    lpips_loss_weight,
                )

            if enable_amp:
                scaler.scale(total_loss).backward()
                if grad_clip_norm > 0:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(_trainable_parameters(active_models), grad_clip_norm)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(_trainable_parameters(active_models), grad_clip_norm)
                for optimizer in optimizers:
                    optimizer.step()

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            lpips_value = lpips_loss.item() if lpips_loss_func is not None else 0.0
            postfix = {
                "scene": batch["scene_name"],
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{image_l1.item():.4f}",
                "lpips": f"{lpips_value:.4f}",
                "psnr": f"{train_psnr.item():.2f}",
                "attr_lr": f"{attribute_optimizer.param_groups[0]['lr']:.2e}",
            }
            if means_model is not None:
                postfix["means_lr"] = f"{means_optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_postfix(postfix)

            if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
                torch.cuda.empty_cache()

            if step % log_interval == 0:
                train_log = {
                    "train/total_loss": total_loss.item(),
                    "train/image_l1": image_l1.item(),
                    "train/lpips": lpips_value,
                    "train/psnr": train_psnr.item(),
                    "train/attr_lr": attribute_optimizer.param_groups[0]["lr"],
                    "train/scene_idx": batch["scene_idx"],
                    "train/input_gaussians": input_factor_entry["gs_params"]["means"].shape[0],
                    "train/model_input_gaussians": model_input_gs["means"].shape[0],
                    "train/target_gaussians": target_factor_entry["gs_params"]["means"].shape[0],
                    "train/views": len(batch_images),
                }
                if means_model is not None:
                    train_log["train/means_lr"] = means_optimizer.param_groups[0]["lr"]
                if FLAGS.means_source != "low_res":
                    train_log["train/densified_gaussians"] = model_input_gs["means"].shape[0]
                _wandb_log(train_log, step=step)

                log_msg = (
                    f"step={step} scene={batch['scene_name']} total={total_loss.item():.6f} "
                    f"l1={image_l1.item():.6f} psnr={train_psnr.item():.4f} "
                    f"attr_lr={attribute_optimizer.param_groups[0]['lr']:.8f}"
                )
                if means_model is not None:
                    log_msg += f" means_lr={means_optimizer.param_groups[0]['lr']:.8f}"
                if lpips_loss_func is not None:
                    log_msg += f" lpips={lpips_loss.item():.6f}"
                logger.info(log_msg)

            if step % log_image_interval == 0:
                with torch.no_grad():
                    log_out_gs = _forward_two_stage(
                        attribute_model=attribute_model,
                        input_gs=model_input_gs,
                        scene_idx=batch_scene_idx[0],
                        target_gs=target_gs,
                        means_source=FLAGS.means_source,
                        means_model=means_model,
                    )
                    log_pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(log_out_gs, batch_cameras)

                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in log_pred_imgs]
                pred_grid_rgb = gs_utils.make_grid(pred_imgs_uint8)
                pred_grid = cv2.cvtColor(pred_grid_rgb, cv2.COLOR_RGB2BGR)
                pred_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png")
                cv2.imwrite(pred_path, pred_grid)

                gt_imgs_uint8 = [
                    (img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in batch_images
                ]
                gt_grid_rgb = gs_utils.make_grid(gt_imgs_uint8)
                gt_grid = cv2.cvtColor(gt_grid_rgb, cv2.COLOR_RGB2BGR)
                gt_path = os.path.join(FLAGS.output_dir, "train", f"{step:08d}_gt.png")
                cv2.imwrite(gt_path, gt_grid)

                if wandb is not None and wandb.run is not None:
                    _wandb_log(
                        {
                            "train/pred_grid": wandb.Image(pred_grid_rgb, caption=f"step={step} pred"),
                            "train/gt_grid": wandb.Image(gt_grid_rgb, caption=f"step={step} gt"),
                        },
                        step=step,
                    )

            if step % eval_interval == 0:
                eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
                metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_dataset(
                    attribute_model=attribute_model,
                    means_model=means_model,
                    dataset=test_dataset,
                    output_dir=eval_dir,
                    compare_with_input=FLAGS.compare_with_input,
                    save_viewer=FLAGS.save_viewer,
                    output_gt=(step == 0),
                    wandb_step=step,
                    evaluate_baselines=(step == 0),
                )
                metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                logger.info(f"Eval step {step}: {metric_str}")
                _wandb_log({f"eval/{key}": value for key, value in metrics.items()}, step=step)
                if step == 0:
                    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_input.items()])
                    logger.info(f"Eval step {step} input: {metric_str}")
                    _wandb_log({f"eval_input/{key}": value for key, value in metrics_input.items()}, step=step)
                    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()])
                    logger.info(f"Eval step {step} GT low-res GS: {metric_str}")
                    _wandb_log({f"eval_gt_low_res/{key}": value for key, value in metrics_gt_low_res.items()}, step=step)
                    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()])
                    logger.info(f"Eval step {step} GT high-res GS: {metric_str}")
                    _wandb_log({f"eval_gt_high_res/{key}": value for key, value in metrics_gt_high_res.items()}, step=step)
                _models_train(attribute_model, means_model)

            if (step + 1) % save_interval == 0:
                ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth")
                torch.save(_checkpoint_payload(attribute_model, means_model, FLAGS.means_source), ckpt_path)
                logger.info(f"Saved model checkpoint to {ckpt_path}")

        last_ckpt_path = os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth")
        torch.save(_checkpoint_payload(attribute_model, means_model, FLAGS.means_source), last_ckpt_path)
        logger.info(f"Saved model checkpoint to {last_ckpt_path}")

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input, metrics_gt_low_res, metrics_gt_high_res = evaluate_dataset(
        attribute_model=attribute_model,
        means_model=means_model,
        dataset=test_dataset,
        output_dir=final_eval_dir,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        output_gt=True,
        evaluate_baselines=True,
    )
    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    _wandb_log({f"final_eval/{key}": value for key, value in metrics.items()})
    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_input.items()])
    logger.info(f"Final eval input: {metric_str}")
    _wandb_log({f"final_eval_input/{key}": value for key, value in metrics_input.items()})
    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_low_res.items()])
    logger.info(f"Final eval GT low-res GS: {metric_str}")
    _wandb_log({f"final_eval_gt_low_res/{key}": value for key, value in metrics_gt_low_res.items()})
    metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics_gt_high_res.items()])
    logger.info(f"Final eval GT high-res GS: {metric_str}")
    _wandb_log({f"final_eval_gt_high_res/{key}": value for key, value in metrics_gt_high_res.items()})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)

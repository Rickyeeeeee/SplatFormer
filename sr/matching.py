"""Fixed-cardinality gsplat fitting for identity-indexed SR pairs."""

import csv
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import gpu_utils, gs_utils, loss_utils
from utils.gs_utils import make_grid
from utils.metrics import psnr, render_gs_average_metrics


MATCHING_CACHE_VERSION = 1


class MatchingCache:
    def __init__(
        self,
        root,
        scene_name,
        input_resolution,
        target_resolution,
        source_gs,
        matching_config,
    ):
        normalized_scene_name = os.path.basename(os.path.normpath(scene_name))
        if not normalized_scene_name or normalized_scene_name in {".", os.pardir}:
            raise ValueError(f"Invalid scene name for matching cache: {scene_name!r}")

        self.cache_dir = os.path.join(
            root,
            normalized_scene_name,
            f"ir{int(input_resolution)}_tr{int(target_resolution)}",
        )
        self.checkpoint_path = os.path.join(
            self.cache_dir, "matching_target.pt"
        )
        self.metadata = {
            "version": MATCHING_CACHE_VERSION,
            "scene_name": scene_name,
            "input_resolution": int(input_resolution),
            "target_resolution": int(target_resolution),
            "source_attributes": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(source_gs.items())
            },
            "matching_config": dict(matching_config),
        }

    @contextmanager
    def lock(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        lock_path = os.path.join(self.cache_dir, ".matching.lock")
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def validate(self, payload, source_gs):
        if not isinstance(payload, dict):
            return None, "payload is not a dictionary"
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None, "missing metadata"
        required_metadata = (
            "version",
            "scene_name",
            "input_resolution",
            "target_resolution",
            "source_attributes",
        )
        for key in required_metadata:
            if metadata.get(key) != self.metadata[key]:
                return None, f"metadata mismatch for {key}"

        target_gs = payload.get("target_gs")
        if not isinstance(target_gs, dict) or set(target_gs) != set(source_gs):
            return None, "target attributes do not match source attributes"
        for key, source_value in source_gs.items():
            target_value = target_gs[key]
            if not torch.is_tensor(target_value):
                return None, f"cached target {key} is not a tensor"
            if (
                target_value.shape != source_value.shape
                or target_value.dtype != source_value.dtype
            ):
                return None, f"cached target {key} shape or dtype mismatch"
        return target_gs, None

    def load(self, source_gs):
        if not os.path.isfile(self.checkpoint_path):
            return None, "checkpoint not found"
        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            cached_target, reason = self.validate(payload, source_gs)
        except Exception as exc:
            return None, f"failed to load checkpoint: {exc}"
        if cached_target is None:
            return None, reason

        device = source_gs["means"].device
        target_gs = {
            key: value.detach().clone().to(
                device=device, dtype=source_gs[key].dtype
            )
            for key, value in cached_target.items()
        }
        return target_gs, "compatible checkpoint"

    def save(self, target_gs):
        payload = {
            "metadata": self.metadata,
            "target_gs": {
                key: value.detach().cpu().clone()
                for key, value in target_gs.items()
            },
        }
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".matching_target_",
            suffix=".pt",
            dir=self.cache_dir,
        )
        os.close(file_descriptor)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, self.checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def build_matching_source(input_resolution_dict, target_resolution_dict, device):
    input_gs = gpu_utils.move_to_device(
        input_resolution_dict["gs_params"], device
    )
    source_gs = gs_utils.convert_gaussian_frame(
        input_gs,
        input_resolution_dict["scaler"],
        target_resolution_dict["scaler"],
    )
    if source_gs["means"].shape[0] != input_gs["means"].shape[0]:
        raise RuntimeError("Matching source construction changed the Gaussian count")
    return source_gs


def make_trainable_gaussians(source_gs):
    return torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(value.detach().clone())
            for key, value in source_gs.items()
        }
    )


def detach_matching_target(trainable_gs, source_gs):
    target_gs = {
        key: value.detach().clone() for key, value in trainable_gs.items()
    }
    if set(target_gs) != set(source_gs):
        raise RuntimeError("Matching fit changed the Gaussian attribute set")
    for key, source_value in source_gs.items():
        if target_gs[key].shape != source_value.shape:
            raise RuntimeError(
                f"Matching fit changed shape for {key}: "
                f"{tuple(source_value.shape)} -> {tuple(target_gs[key].shape)}"
            )
    return target_gs


def get_or_fit_matching_target(
    source_gs,
    target_images,
    target_cameras,
    pre_matching_root,
    scene_name,
    input_resolution,
    target_resolution,
    logger,
    config,
    optimizer_factory,
    force_pre_matching=False,
):
    cache = MatchingCache(
        pre_matching_root,
        scene_name,
        input_resolution,
        target_resolution,
        source_gs,
        config,
    )
    if force_pre_matching:
        logger.info(
            "Pre-matching forced rematch (persistent cache left unchanged): %s",
            cache.checkpoint_path,
        )
        with tempfile.TemporaryDirectory(
            prefix="splatformer_matching_fit_"
        ) as fit_output_dir:
            target_gs = fit_matching_target(
                source_gs,
                target_images,
                target_cameras,
                fit_output_dir,
                logger,
                config,
                optimizer_factory,
            )
        return target_gs, {
            "status": "refit_uncached",
            "cache_dir": cache.cache_dir,
            "checkpoint_path": cache.checkpoint_path,
            "reason": "forced rematch; persistent cache unchanged",
        }

    with cache.lock():
        target_gs, cache_reason = cache.load(source_gs)
        if target_gs is not None:
            logger.info("Pre-matching cache hit: %s", cache.checkpoint_path)
            return target_gs, {
                "status": "hit",
                "cache_dir": cache.cache_dir,
                "checkpoint_path": cache.checkpoint_path,
                "reason": cache_reason,
            }

        logger.info(
            "Pre-matching cache miss (%s): %s",
            cache_reason,
            cache.checkpoint_path,
        )
        target_gs = fit_matching_target(
            source_gs,
            target_images,
            target_cameras,
            cache.cache_dir,
            logger,
            config,
            optimizer_factory,
        )
        cache.save(target_gs)
        logger.info("Pre-matching cache saved: %s", cache.checkpoint_path)
        return target_gs, {
            "status": "refit",
            "cache_dir": cache.cache_dir,
            "checkpoint_path": cache.checkpoint_path,
            "reason": cache_reason,
        }


def compute_matching_render_loss(
    pred_imgs,
    target_images,
    lpips_loss_func,
    image_l1_loss_weight,
    lpips_loss_weight,
):
    if not pred_imgs:
        raise ValueError("Matching fit received zero rendered views")
    if len(pred_imgs) != len(target_images):
        raise ValueError(
            f"Matching fit rendered {len(pred_imgs)} views for "
            f"{len(target_images)} targets"
        )

    image_l1 = pred_imgs[0].new_zeros(())
    lpips_loss = pred_imgs[0].new_zeros(())
    train_psnr = pred_imgs[0].new_zeros(())
    for pred_img, gt_img in zip(pred_imgs, target_images):
        gt_rgb = gt_img[..., :3]
        image_l1 = image_l1 + (pred_img - gt_rgb).abs().mean()
        train_psnr = train_psnr + psnr(
            pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)
        ).mean()
        if lpips_loss_func is not None:
            lpips_loss = lpips_loss + lpips_loss_func(
                pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)
            ).mean()

    image_l1 = image_l1 / len(pred_imgs) * image_l1_loss_weight
    train_psnr = train_psnr / len(pred_imgs)
    if lpips_loss_func is not None:
        lpips_loss = lpips_loss / len(pred_imgs) * lpips_loss_weight
    return image_l1 + lpips_loss, image_l1, lpips_loss, train_psnr


def write_matching_preview(pred_imgs, output_path):
    images = [
        (image.clamp(0.0, 1.0) * 255)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
        for image in pred_imgs
    ]
    cv2.imwrite(
        output_path,
        cv2.cvtColor(make_grid(images), cv2.COLOR_RGB2BGR),
    )


def save_matching_artifacts(
    output_dir,
    source_gs,
    target_gs,
    images,
    cameras,
    chunk_size,
    device,
):
    matching_dir = os.path.join(output_dir, "matching_init")
    os.makedirs(matching_dir, exist_ok=True)
    stages = {
        "00_input_low_res_gs.ply": source_gs,
        "01_fitted_target_gs.ply": target_gs,
    }
    for name, gs in stages.items():
        gs_utils.export_ply_forviewer(gs, os.path.join(matching_dir, name))
    metrics = {
        name: render_gs_average_metrics(
            gs, images, cameras, chunk_size, device
        )
        for name, gs in stages.items()
    }
    with open(
        os.path.join(matching_dir, "render_metrics.json"), "w"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def fit_matching_target(
    source_gs,
    target_images,
    target_cameras,
    output_dir,
    logger,
    config,
    optimizer_factory,
):
    if config["total_steps"] <= 0:
        raise ValueError(
            f"matching_fit.total_steps must be positive, "
            f"got {config['total_steps']}"
        )
    if config["image_l1_loss_weight"] < 0 or config["lpips_loss_weight"] < 0:
        raise ValueError("matching fitting loss weights must be non-negative")
    if (
        config["image_l1_loss_weight"] == 0
        and config["lpips_loss_weight"] == 0
    ):
        raise ValueError("matching fitting needs a non-zero L1 or LPIPS weight")
    if not target_images:
        raise ValueError("Matching fit requires at least one target-factor image")

    device = source_gs["means"].device
    trainable_gs = make_trainable_gaussians(source_gs)
    optimizer, scheduler = optimizer_factory(trainable_gs)

    target_images = gpu_utils.move_to_device(target_images, device)
    target_cameras = gpu_utils.move_to_device(target_cameras, device)
    target_view_count = len(target_images)
    image_per_step = int(config["image_per_step"])
    if image_per_step < 0:
        raise ValueError(
            f"matching_fit.image_per_step must be >= 0, got {image_per_step}"
        )
    image_per_step = (
        target_view_count
        if image_per_step == 0
        else min(image_per_step, target_view_count)
    )

    matching_train_dir = os.path.join(output_dir, "matching_train")
    os.makedirs(matching_train_dir, exist_ok=True)
    lpips_loss_func = (
        loss_utils.lpips_loss_fn()
        if config["lpips_loss_weight"] > 0
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["enable_amp"])
    optimizer.zero_grad(set_to_none=True)

    loss_path = os.path.join(matching_train_dir, "loss.csv")
    with open(loss_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "total_loss", "l1", "lpips", "psnr", "lr"],
        )
        writer.writeheader()
        progress = tqdm(range(config["total_steps"]), desc="matching fit")
        for step in progress:
            if image_per_step == target_view_count:
                batch_images = target_images
                batch_cameras = target_cameras
            else:
                view_indices = torch.randperm(
                    target_view_count, device=device
                )[:image_per_step]
                batch_images = [
                    target_images[index] for index in view_indices.tolist()
                ]
                batch_cameras = {
                    key: (
                        value[view_indices]
                        if key == "camera_to_worlds"
                        else value
                    )
                    for key, value in target_cameras.items()
                }

            with torch.cuda.amp.autocast(enabled=config["enable_amp"]):
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(
                    trainable_gs, batch_cameras
                )
                total_loss, image_l1, lpips_value, train_psnr = (
                    compute_matching_render_loss(
                        pred_imgs,
                        batch_images,
                        lpips_loss_func,
                        config["image_l1_loss_weight"],
                        config["lpips_loss_weight"],
                    )
                )
            if config["enable_amp"]:
                scaler.scale(total_loss).backward()
                if config["grad_clip_norm"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_gs.parameters(), config["grad_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_gs.parameters(), config["grad_clip_norm"]
                    )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            values = {
                "step": step,
                "total_loss": total_loss.item(),
                "l1": image_l1.item(),
                "lpips": lpips_value.item(),
                "psnr": train_psnr.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(values)
            csv_file.flush()
            progress.set_postfix(
                loss=f"{values['total_loss']:.4f}",
                psnr=f"{values['psnr']:.2f}",
            )
            if step % config["log_interval"] == 0:
                logger.info(
                    "matching step=%d total=%.6f l1=%.6f "
                    "lpips=%.6f psnr=%.4f lr=%.8f",
                    step,
                    values["total_loss"],
                    values["l1"],
                    values["lpips"],
                    values["psnr"],
                    values["lr"],
                )
            if step % config["preview_interval"] == 0:
                write_matching_preview(
                    pred_imgs,
                    os.path.join(
                        matching_train_dir, f"{step:08d}_pred.png"
                    ),
                )
            if (
                config["empty_cache_fre"] > 0
                and (step + 1) % config["empty_cache_fre"] == 0
            ):
                torch.cuda.empty_cache()
    return detach_matching_target(trainable_gs, source_gs)

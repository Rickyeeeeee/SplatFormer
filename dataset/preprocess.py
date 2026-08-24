#!/usr/bin/env python3
"""Validate, evaluate, select, pre-fit, and analyze native-resolution GS scenes."""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from utils.transform_utils import MinMaxScaler


DEFAULT_RESOLUTIONS = (512, 128)
DEFAULT_EXPECTED_IMAGES = 128
CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"
CHECKPOINT_PATTERN = re.compile(r"^step-(\d+)\.ckpt$")
REQUIRED_CAMERA_KEYS = (
    "train_camera_to_worlds",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
)
REQUIRED_GS_KEYS = {"means", "scales", "quats", "features_dc", "opacities"}
STRUCTURE_COLUMNS = ["scene", "status", "reason", "detail"]
METRIC_COLUMNS = [
    "scene",
    "resolution",
    "status",
    "reason",
    "checkpoint_path",
    "image_count",
    "gs_count",
    "psnr",
    "ssim",
    "lpips",
]
SELECTION_COLUMNS = [
    "scene",
    "status",
    "reason",
    "observed_min_psnr",
    "observed_min_gs_count",
    "required_min_psnr",
    "required_min_gs_count",
]
PRETRAIN_COLUMNS = [
    "scene",
    "direction",
    "source_resolution",
    "target_resolution",
    "status",
    "cache_status",
    "cache_path",
    "reason",
]
STATISTICS_PARAMETERS = (
    "means",
    "scales",
    "opacities",
    "quats",
    "features_dc",
    "features_rest",
)
STATISTICS_SPACES = ("checkpoint", "processed")
STATISTICS_REPRESENTATIONS = ("raw", "post_activation")
STATISTICS_RECORD_VERSION = 1
MATCHING_LR_DICT = {
    "base": 1e-3,
    "means": 1.6e-4,
    "features_dc": 2.5e-3,
    "features_rest": 1.25e-4,
    "opacities": 5e-2,
    "scales": 5e-3,
    "quats": 1e-3,
}


def build_matching_source(*args, **kwargs):
    from utils.sr_matching_utils import build_matching_source as implementation

    return implementation(*args, **kwargs)


def get_or_fit_matching_target(*args, **kwargs):
    from utils.sr_matching_utils import get_or_fit_matching_target as implementation

    return implementation(*args, **kwargs)


def _path_or_default(value: Optional[Path], dataset_root: Path, name: str) -> Path:
    return Path(value) if value is not None else dataset_root / name


def structure_report_path(args) -> Path:
    return _path_or_default(args.structure_report, args.dataset_root, "structure_report.csv")


def structural_scene_list_path(args) -> Path:
    return _path_or_default(
        args.structural_scene_list,
        args.dataset_root,
        "structurally_valid_scenes.txt",
    )


def metrics_path(args) -> Path:
    return _path_or_default(args.metrics_csv, args.dataset_root, "scene_metrics.csv")


def selection_report_path(args) -> Path:
    return _path_or_default(args.selection_report, args.dataset_root, "selection_report.csv")


def valid_scene_list_path(args) -> Path:
    return _path_or_default(args.valid_scene_list, args.dataset_root, "valid_scenes.txt")


def pretrain_status_path(args) -> Path:
    return _path_or_default(args.pretrain_status, args.dataset_root, "pretrain_status.csv")


def cache_root_path(args) -> Path:
    return _path_or_default(args.cache_root, args.dataset_root, "pretrained_gaussians")


def statistics_json_path(args) -> Path:
    return _path_or_default(args.statistics_json, args.dataset_root, "gs_statistics.json")


def statistics_records_path(args) -> Path:
    return _path_or_default(
        args.statistics_records,
        args.dataset_root,
        "gs_statistics_scenes.jsonl",
    )


def atomic_write_scene_list(path: Path, scenes: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for scene in scenes:
                handle.write("%s\n" % scene)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_csv_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())


def read_scene_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def list_pngs(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )


def latest_checkpoint(nerfstudio_dir: Path) -> Path:
    model_dir = nerfstudio_dir / "nerfstudio_models"
    candidates = []
    if model_dir.is_dir():
        for path in model_dir.iterdir():
            match = CHECKPOINT_PATTERN.match(path.name)
            if path.is_file() and match is not None:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No numeric step-*.ckpt in %s" % model_dir)
    return max(candidates, key=lambda item: item[0])[1]


def scene_resolution_paths(
    dataset_root: Path, scene: str, resolution: int
) -> Dict[str, Path]:
    resolution_root = dataset_root / str(resolution)
    colmap_dir = resolution_root / "colmap" / scene
    nerfstudio_dir = resolution_root / "nerfstudio" / scene / "splatfacto"
    return {
        "colmap_dir": colmap_dir,
        "image_dir": colmap_dir / "images",
        "nerfstudio_dir": nerfstudio_dir,
        "camera_path": nerfstudio_dir / CAMERA_METADATA_NAME,
    }


def discover_scenes(dataset_root: Path, resolutions: Sequence[int]) -> List[str]:
    scenes = set()
    for resolution in resolutions:
        resolution_root = dataset_root / str(resolution)
        for folder_name in ("colmap", "nerfstudio"):
            folder = resolution_root / folder_name
            if folder.is_dir():
                scenes.update(path.name for path in folder.iterdir() if path.is_dir())
    return sorted(scenes)


def _camera_pose_count(camera_path: Path) -> int:
    with camera_path.open("rb") as handle:
        metadata = pickle.load(handle)
    missing = [key for key in REQUIRED_CAMERA_KEYS if key not in metadata]
    if missing:
        raise KeyError("missing camera keys %s" % ",".join(missing))
    return len(metadata["train_camera_to_worlds"])


def validate_scene_structure(
    dataset_root: Path,
    scene: str,
    resolutions: Sequence[int],
    expected_images: int,
) -> Tuple[bool, str, str]:
    reference_names = None
    for resolution in resolutions:
        paths = scene_resolution_paths(dataset_root, scene, resolution)
        if not paths["colmap_dir"].is_dir():
            return False, "missing_colmap_scene", "%s:%s" % (resolution, paths["colmap_dir"])
        if not paths["image_dir"].is_dir():
            return False, "missing_image_dir", "%s:%s" % (resolution, paths["image_dir"])

        image_paths = list_pngs(paths["image_dir"])
        if len(image_paths) != expected_images:
            return (
                False,
                "bad_image_count",
                "%s:got=%d expected=%d" % (resolution, len(image_paths), expected_images),
            )
        if not paths["nerfstudio_dir"].is_dir():
            return (
                False,
                "missing_nerfstudio_scene",
                "%s:%s" % (resolution, paths["nerfstudio_dir"]),
            )
        if not paths["camera_path"].is_file():
            return (
                False,
                "missing_camera_metadata",
                "%s:%s" % (resolution, paths["camera_path"]),
            )
        try:
            pose_count = _camera_pose_count(paths["camera_path"])
        except Exception as exc:
            return (
                False,
                "invalid_camera_metadata",
                "%s:%s:%s" % (resolution, type(exc).__name__, exc),
            )
        if pose_count != len(image_paths):
            return (
                False,
                "image_pose_count_mismatch",
                "%s:images=%d poses=%d" % (resolution, len(image_paths), pose_count),
            )
        try:
            latest_checkpoint(paths["nerfstudio_dir"])
        except FileNotFoundError as exc:
            return False, "missing_checkpoint", "%s:%s" % (resolution, exc)

        image_names = [path.name for path in image_paths]
        if reference_names is None:
            reference_names = image_names
        elif image_names != reference_names:
            return (
                False,
                "cross_resolution_image_mismatch",
                "%s_vs_%s" % (resolutions[0], resolution),
            )
    return True, "valid", ""


def run_validate(args) -> List[str]:
    candidate_scenes = discover_scenes(args.dataset_root, args.resolutions)
    if args.candidate_scene_list is not None:
        requested = set(read_scene_list(args.candidate_scene_list))
        candidate_scenes = [scene for scene in candidate_scenes if scene in requested]
    if args.max_scenes is not None:
        candidate_scenes = candidate_scenes[: args.max_scenes]

    rows = []
    valid_scenes = []
    for scene in tqdm(candidate_scenes, desc="validate"):
        valid, reason, detail = validate_scene_structure(
            args.dataset_root,
            scene,
            args.resolutions,
            args.expected_images,
        )
        rows.append(
            {
                "scene": scene,
                "status": "valid" if valid else "invalid",
                "reason": reason,
                "detail": detail,
            }
        )
        if valid:
            valid_scenes.append(scene)

    atomic_write_csv(structure_report_path(args), STRUCTURE_COLUMNS, rows)
    atomic_write_scene_list(structural_scene_list_path(args), valid_scenes)
    print(
        "Validated %d scenes: %d valid, %d invalid"
        % (len(candidate_scenes), len(valid_scenes), len(candidate_scenes) - len(valid_scenes))
    )
    print("Structure report: %s" % structure_report_path(args))
    print("Structurally valid scenes: %s" % structural_scene_list_path(args))
    return valid_scenes


def require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("Evaluation and pretraining require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.index is not None:
        torch.cuda.set_device(device)
    return device


def read_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0
    if image.ndim != 3:
        raise ValueError("Expected HWC image at %s, got %s" % (path, image.shape))
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha
    elif image.shape[2] != 3:
        raise ValueError("Expected RGB/RGBA image at %s" % path)
    return torch.from_numpy(image)


def _reduce_nan_mask(value: torch.Tensor) -> torch.Tensor:
    has_nan = torch.isnan(value)
    for dim in range(value.ndim - 1, 0, -1):
        has_nan = has_nan.any(dim=dim)
    return has_nan


def load_gaussian_params(
    checkpoint_path: Path, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], MinMaxScaler, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    gs_params = {
        key.replace("_model.gauss_params.", ""): value
        for key, value in checkpoint.items()
        if "gauss_params" in key
    }
    missing = sorted(REQUIRED_GS_KEYS - set(gs_params))
    if missing:
        raise KeyError("Checkpoint %s is missing GS keys %s" % (checkpoint_path, missing))

    gaussian_count = gs_params["means"].shape[0]
    select = torch.ones(gaussian_count, dtype=torch.bool)
    for value in gs_params.values():
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == gaussian_count
        ):
            select &= ~_reduce_nan_mask(value)

    filtered = {}
    for key, value in gs_params.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == gaussian_count:
            filtered[key] = value[select]
        else:
            filtered[key] = value

    if filtered["means"].shape[0] == 0:
        raise ValueError("Checkpoint %s has zero finite Gaussians" % checkpoint_path)

    scaler = MinMaxScaler()
    filtered["means"] = scaler.fit_transform(filtered["means"])
    filtered["scales"] = filtered["scales"] + torch.log(scaler.scale_)

    valid_mask = ~torch.isinf(filtered["scales"]).any(dim=1)
    valid_mask &= torch.all(
        (filtered["means"] >= 0) & (filtered["means"] <= 1), dim=1
    )
    for key, value in list(filtered.items()):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == valid_mask.shape[0]:
            filtered[key] = value[valid_mask].to(device)
        elif torch.is_tensor(value):
            filtered[key] = value.to(device)

    valid_count = int(filtered["means"].shape[0])
    if valid_count == 0:
        raise ValueError("Checkpoint %s has zero valid Gaussians after filtering" % checkpoint_path)
    return filtered, scaler, valid_count


def load_cameras(
    camera_path: Path, scaler: MinMaxScaler, device: torch.device
) -> Dict[str, torch.Tensor]:
    with camera_path.open("rb") as handle:
        metadata = pickle.load(handle)
    missing = [key for key in REQUIRED_CAMERA_KEYS if key not in metadata]
    if missing:
        raise KeyError("Camera metadata %s is missing %s" % (camera_path, missing))

    camera_to_worlds = torch.as_tensor(
        metadata["train_camera_to_worlds"], dtype=torch.float32
    ).clone()
    camera_to_worlds[:, :3, -1] = scaler.transform(camera_to_worlds[:, :3, -1])
    camera_to_worlds = camera_to_worlds.to(device)
    return {
        "camera_to_worlds": camera_to_worlds,
        "fx": torch.as_tensor(metadata["fx"], dtype=torch.float32, device=device),
        "fy": torch.as_tensor(metadata["fy"], dtype=torch.float32, device=device),
        "cx": torch.as_tensor(metadata["cx"], dtype=torch.float32, device=device),
        "cy": torch.as_tensor(metadata["cy"], dtype=torch.float32, device=device),
        "width": torch.as_tensor(metadata["width"], dtype=torch.float32, device=device),
        "height": torch.as_tensor(metadata["height"], dtype=torch.float32, device=device),
        "background_color": torch.zeros(3, dtype=torch.float32, device=device),
    }


def load_resolution_bundle(
    dataset_root: Path,
    scene: str,
    resolution: int,
    device: torch.device,
) -> Dict:
    paths = scene_resolution_paths(dataset_root, scene, resolution)
    checkpoint_path = latest_checkpoint(paths["nerfstudio_dir"])
    gs_params, scaler, gs_count = load_gaussian_params(checkpoint_path, device)
    cameras = load_cameras(paths["camera_path"], scaler, device)
    image_paths = list_pngs(paths["image_dir"])
    if len(image_paths) != len(cameras["camera_to_worlds"]):
        raise ValueError(
            "Scene %s resolution %d image/pose count mismatch" % (scene, resolution)
        )
    return {
        "gs_params": gs_params,
        "scaler": scaler,
        "gs_count": gs_count,
        "cameras": cameras,
        "image_paths": image_paths,
        "checkpoint_path": checkpoint_path,
    }


def build_lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS is required for evaluation and pretraining") from exc
    model = lpips.LPIPS(net="vgg", verbose=False).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def evaluate_bundle(
    bundle: Dict,
    device: torch.device,
    chunk_size: int,
    lpips_model,
) -> Dict[str, float]:
    from utils import gs_utils
    from utils.metrics import psnr, ssim

    image_paths = bundle["image_paths"]
    view_count = len(image_paths)
    if view_count == 0:
        raise ValueError("Cannot evaluate a resolution with zero images")
    chunk_size = min(max(1, int(chunk_size)), view_count)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for start in range(0, view_count, chunk_size):
            end = min(start + chunk_size, view_count)
            ground_truth = torch.stack(
                [read_image(path) for path in image_paths[start:end]], dim=0
            ).to(device)
            chunk_cameras = {
                key: (
                    value[start:end]
                    if key == "camera_to_worlds"
                    else value
                )
                for key, value in bundle["cameras"].items()
            }
            predictions, _ = gs_utils.rasterize_gaussians_to_multiimgs(
                bundle["gs_params"], chunk_cameras
            )
            predictions = torch.stack(predictions, dim=0).clamp(0.0, 1.0)
            ground_truth = ground_truth[..., :3].clamp(0.0, 1.0)
            batch_count = end - start

            totals["psnr"] += float(
                psnr(predictions, ground_truth).reshape(-1).sum().item()
            )
            totals["ssim"] += float(
                ssim(
                    predictions.permute(0, 3, 1, 2),
                    ground_truth.permute(0, 3, 1, 2),
                    window_size=11,
                    size_average=False,
                )
                .reshape(-1)
                .sum()
                .item()
            )
            lpips_values = lpips_model(
                predictions.permute(0, 3, 1, 2),
                ground_truth.permute(0, 3, 1, 2),
                normalize=True,
            )
            totals["lpips"] += float(lpips_values.reshape(batch_count, -1).mean(dim=1).sum().item())

    return {key: value / view_count for key, value in totals.items()}


def _read_latest_rows(path: Path, key_fields: Sequence[str]) -> Dict[Tuple, Dict[str, str]]:
    latest = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row.get(field, "") for field in key_fields)
            if all(key):
                latest[key] = row
    return latest


def _metric_scene_state(
    latest_rows: Dict[Tuple, Dict[str, str]],
    scene: str,
    resolutions: Sequence[int],
) -> str:
    rows = [
        latest_rows.get((scene, str(resolution))) for resolution in resolutions
    ]
    if all(row is not None and row.get("status") == "ok" for row in rows):
        return "complete"
    if any(row is not None and row.get("status") == "error" for row in rows):
        return "error"
    return "pending"


def run_evaluate(args) -> None:
    device = require_cuda(args.device)
    scene_list = (
        getattr(args, "scene_list", None)
        if getattr(args, "scene_list", None) is not None
        else structural_scene_list_path(args)
    )
    scenes = read_scene_list(scene_list)
    metric_file = metrics_path(args)
    latest_rows = _read_latest_rows(metric_file, ("scene", "resolution"))
    lpips_model = build_lpips_model(device)

    attempted = 0
    for scene in tqdm(scenes, desc="evaluate"):
        state = _metric_scene_state(latest_rows, scene, args.resolutions)
        if state == "complete" or (state == "error" and not args.retry_errors):
            continue
        if args.max_scenes is not None and attempted >= args.max_scenes:
            break
        attempted += 1
        scene_rows = []
        for resolution in args.resolutions:
            try:
                bundle = load_resolution_bundle(
                    args.dataset_root, scene, resolution, device
                )
                values = evaluate_bundle(
                    bundle,
                    device,
                    args.render_chunk_size,
                    lpips_model,
                )
                row = {
                    "scene": scene,
                    "resolution": resolution,
                    "status": "ok",
                    "reason": "",
                    "checkpoint_path": bundle["checkpoint_path"],
                    "image_count": len(bundle["image_paths"]),
                    "gs_count": bundle["gs_count"],
                    "psnr": "%.8f" % values["psnr"],
                    "ssim": "%.8f" % values["ssim"],
                    "lpips": "%.8f" % values["lpips"],
                }
            except Exception as exc:
                row = {
                    "scene": scene,
                    "resolution": resolution,
                    "status": "error",
                    "reason": "%s:%s" % (type(exc).__name__, exc),
                }
            scene_rows.append(row)
            latest_rows[(scene, str(resolution))] = {
                key: str(value) for key, value in row.items()
            }
            torch.cuda.empty_cache()
        append_csv_rows(metric_file, METRIC_COLUMNS, scene_rows)

    print("Metrics CSV: %s" % metric_file)


def _safe_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_select(args) -> List[str]:
    metric_file = metrics_path(args)
    latest_rows = _read_latest_rows(metric_file, ("scene", "resolution"))
    scene_list = (
        getattr(args, "scene_list", None)
        if getattr(args, "scene_list", None) is not None
        else structural_scene_list_path(args)
    )
    scenes = read_scene_list(scene_list)

    selected = []
    report_rows = []
    for scene in scenes:
        reasons = []
        observed_psnr = []
        observed_counts = []
        for resolution in args.resolutions:
            row = latest_rows.get((scene, str(resolution)))
            if row is None:
                reasons.append("missing_metrics:%d" % resolution)
                continue
            if row.get("status") != "ok":
                reasons.append("evaluation_error:%d" % resolution)
                continue

            row_psnr = _safe_float(row.get("psnr"))
            row_count = _safe_int(row.get("gs_count"))
            if row_psnr is None:
                reasons.append("invalid_psnr:%d" % resolution)
            else:
                observed_psnr.append(row_psnr)
                if row_psnr < args.min_psnr:
                    reasons.append("psnr_below_threshold:%d" % resolution)
            if row_count is None:
                reasons.append("invalid_gs_count:%d" % resolution)
            else:
                observed_counts.append(row_count)
                if row_count < args.min_gs_count:
                    reasons.append("gs_count_below_threshold:%d" % resolution)

        if not reasons:
            selected.append(scene)
        report_rows.append(
            {
                "scene": scene,
                "status": "selected" if not reasons else "rejected",
                "reason": ";".join(reasons),
                "observed_min_psnr": min(observed_psnr) if observed_psnr else "",
                "observed_min_gs_count": min(observed_counts) if observed_counts else "",
                "required_min_psnr": args.min_psnr,
                "required_min_gs_count": args.min_gs_count,
            }
        )

    selected = sorted(set(selected))
    atomic_write_csv(selection_report_path(args), SELECTION_COLUMNS, report_rows)
    atomic_write_scene_list(valid_scene_list_path(args), selected)
    print("Selected %d of %d scenes" % (len(selected), len(scenes)))
    print("Selection report: %s" % selection_report_path(args))
    print("GS_SR scene list: %s" % valid_scene_list_path(args))
    return selected


def _bind_matching_optimizer(total_steps: int) -> None:
    from utils import optimizers  # noqa: F401

    bindings = {
        "matching_fit/build_3DGSoptimizer.lr_dict": MATCHING_LR_DICT,
        "matching_fit/build_3DGSoptimizer.optimizer_type": "adam",
        "matching_fit/build_3DGSoptimizer.optimizer_params": {"eps": 1e-15},
        "matching_fit/build_scheduler.total_step": total_steps,
        "matching_fit/build_scheduler.schedule": "constant",
    }
    for selector, value in bindings.items():
        gin.bind_parameter(selector, value)


def matching_config_from_args(args) -> Dict:
    return {
        "total_steps": args.matching_steps,
        "image_per_step": args.matching_images_per_step,
        "log_interval": args.matching_log_interval,
        "preview_interval": args.matching_preview_interval,
        "grad_clip_norm": 0.0,
        "image_l1_loss_weight": args.matching_l1_weight,
        "lpips_loss_weight": args.matching_lpips_weight,
        "enable_amp": args.amp,
        "empty_cache_fre": -1,
    }


def requested_directions(direction: str) -> List[str]:
    if direction == "none":
        return []
    if direction == "both":
        return ["lr_to_hr", "hr_to_lr"]
    return [direction]


def _direction_resolutions(
    direction: str, resolutions: Sequence[int]
) -> Tuple[int, int]:
    low_resolution = min(resolutions)
    high_resolution = max(resolutions)
    if low_resolution == high_resolution:
        raise ValueError("Pretraining requires at least two distinct resolutions")
    if direction == "lr_to_hr":
        return low_resolution, high_resolution
    if direction == "hr_to_lr":
        return high_resolution, low_resolution
    raise ValueError("Unsupported pretraining direction %s" % direction)


def _pretrain_state(
    latest_rows: Dict[Tuple, Dict[str, str]], scene: str, direction: str
) -> str:
    row = latest_rows.get((scene, direction))
    if row is None:
        return "pending"
    return row.get("status", "error")


def run_pretrain(args) -> None:
    directions = requested_directions(args.direction)
    if not directions:
        print("Pretraining disabled")
        return

    device = require_cuda(args.device)
    scene_list = (
        getattr(args, "scene_list", None)
        if getattr(args, "scene_list", None) is not None
        else valid_scene_list_path(args)
    )
    scenes = read_scene_list(scene_list)
    status_file = pretrain_status_path(args)
    latest_rows = _read_latest_rows(status_file, ("scene", "direction"))
    cache_root = cache_root_path(args)
    cache_root.mkdir(parents=True, exist_ok=True)
    _bind_matching_optimizer(args.matching_steps)
    config = matching_config_from_args(args)
    logger = logging.getLogger("gs_sr_preprocess")
    attempted_scenes = 0

    for scene in tqdm(scenes, desc="pretrain"):
        pending_directions = []
        for direction in directions:
            state = _pretrain_state(latest_rows, scene, direction)
            if args.force_refit or state == "pending" or (
                state == "error" and args.retry_errors
            ):
                pending_directions.append(direction)
        if not pending_directions:
            continue
        if args.max_scenes is not None and attempted_scenes >= args.max_scenes:
            break
        attempted_scenes += 1

        for direction in pending_directions:
            source_resolution, target_resolution = _direction_resolutions(
                direction, args.resolutions
            )
            try:
                source_bundle = load_resolution_bundle(
                    args.dataset_root, scene, source_resolution, device
                )
                target_bundle = load_resolution_bundle(
                    args.dataset_root, scene, target_resolution, device
                )
                source_entry = {
                    "gs_params": source_bundle["gs_params"],
                    "scaler": source_bundle["scaler"],
                }
                target_entry = {
                    "gs_params": target_bundle["gs_params"],
                    "scaler": target_bundle["scaler"],
                }
                source_gs = build_matching_source(source_entry, target_entry, device)
                target_images = [
                    read_image(path) for path in target_bundle["image_paths"]
                ]
                fitted_target, cache = get_or_fit_matching_target(
                    source_gs=source_gs,
                    target_images=target_images,
                    target_cameras=target_bundle["cameras"],
                    pre_matching_root=str(cache_root),
                    scene_name=scene,
                    input_resolution=source_resolution,
                    target_resolution=target_resolution,
                    logger=logger,
                    config=config,
                    force_pre_matching=args.force_refit,
                )
                del fitted_target
                row = {
                    "scene": scene,
                    "direction": direction,
                    "source_resolution": source_resolution,
                    "target_resolution": target_resolution,
                    "status": "ok",
                    "cache_status": cache["status"],
                    "cache_path": cache["checkpoint_path"],
                    "reason": "",
                }
            except Exception as exc:
                row = {
                    "scene": scene,
                    "direction": direction,
                    "source_resolution": source_resolution,
                    "target_resolution": target_resolution,
                    "status": "error",
                    "cache_status": "",
                    "cache_path": "",
                    "reason": "%s:%s" % (type(exc).__name__, exc),
                }
                logger.exception("Pretraining failed for %s %s", scene, direction)
            append_csv_rows(status_file, PRETRAIN_COLUMNS, [row])
            latest_rows[(scene, direction)] = {
                key: str(value) for key, value in row.items()
            }
            torch.cuda.empty_cache()

    print("Pretraining status: %s" % status_file)
    print("Pretraining cache root: %s" % cache_root)


def _scaler_tensor(scaler: MinMaxScaler, name: str, value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        getattr(scaler, name),
        dtype=value.dtype,
        device=value.device,
    )


def convert_gaussian_frame(
    gs_params: Mapping[str, torch.Tensor],
    input_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
) -> Dict[str, torch.Tensor]:
    """Convert processed Gaussian tensors between scene normalization frames."""
    means = gs_params["means"]
    input_scale = _scaler_tensor(input_scaler, "scale_", means)
    input_trans = _scaler_tensor(input_scaler, "trans_", means)
    target_scale = _scaler_tensor(target_scaler, "scale_", means)
    target_trans = _scaler_tensor(target_scaler, "trans_", means)
    world_means = (means - input_trans) / input_scale

    converted = {}
    for key, value in gs_params.items():
        if key == "means":
            converted[key] = world_means * target_scale + target_trans
        elif key == "scales":
            converted[key] = value - torch.log(input_scale) + torch.log(target_scale)
        else:
            converted[key] = value.clone()
    return converted


def gaussian_checkpoint_space(
    gs_params: Mapping[str, torch.Tensor],
    scaler: MinMaxScaler,
) -> Dict[str, torch.Tensor]:
    """Undo GS_SR coordinate normalization while retaining its selected rows."""
    means = gs_params["means"]
    scale = _scaler_tensor(scaler, "scale_", means)
    trans = _scaler_tensor(scaler, "trans_", means)
    converted = {}
    for key, value in gs_params.items():
        if key == "means":
            converted[key] = (value - trans) / scale
        elif key == "scales":
            converted[key] = value - torch.log(scale)
        else:
            converted[key] = value.clone()
    return converted


def activate_gaussian_params(
    gs_params: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    activated = {}
    for key in STATISTICS_PARAMETERS:
        value = gs_params[key]
        if key == "scales":
            value = torch.exp(value)
        elif key == "opacities":
            value = torch.sigmoid(value)
        elif key == "quats":
            value = torch.nn.functional.normalize(value, dim=-1)
        activated[key] = value
    return activated


def validate_statistics_gaussians(
    gs_params: Mapping[str, torch.Tensor],
) -> int:
    missing = [key for key in STATISTICS_PARAMETERS if key not in gs_params]
    if missing:
        raise KeyError("Gaussian parameters are missing %s" % missing)

    gaussian_count = None
    for key in STATISTICS_PARAMETERS:
        value = gs_params[key]
        if not torch.is_tensor(value) or value.ndim < 1:
            raise ValueError("Gaussian parameter %s is not a row tensor" % key)
        if not torch.is_floating_point(value):
            raise ValueError("Gaussian parameter %s is not floating point" % key)
        if gaussian_count is None:
            gaussian_count = int(value.shape[0])
        elif value.shape[0] != gaussian_count:
            raise ValueError("Gaussian parameter row counts do not match")
        if not torch.isfinite(value).all().item():
            raise ValueError("Gaussian parameter %s contains non-finite values" % key)
    if not gaussian_count:
        raise ValueError("Gaussian parameters contain zero rows")
    return gaussian_count


def scene_parameter_summary(
    gs_params: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    gaussian_count = validate_statistics_gaussians(gs_params)
    summary = {}
    for key in STATISTICS_PARAMETERS:
        value = gs_params[key].detach().to(device="cpu", dtype=torch.float64)
        mean = value.mean(dim=0)
        std = value.std(dim=0, correction=0)
        summary[key] = {
            "gaussian_count": gaussian_count,
            "channel_shape": list(value.shape[1:]),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return summary


def _statistics_group(
    source_kind: str,
    coordinate_space: str,
    representation: str,
    gs_params: Mapping[str, torch.Tensor],
    **metadata,
) -> Dict[str, Any]:
    return {
        "source_kind": source_kind,
        "coordinate_space": coordinate_space,
        "representation": representation,
        **metadata,
        "parameters": scene_parameter_summary(gs_params),
    }


def add_distribution_groups(
    groups: List[Dict[str, Any]],
    source_kind: str,
    spaces: Mapping[str, Mapping[str, torch.Tensor]],
    **metadata,
) -> None:
    for coordinate_space in STATISTICS_SPACES:
        raw = spaces[coordinate_space]
        groups.append(
            _statistics_group(
                source_kind,
                coordinate_space,
                "raw",
                raw,
                **metadata,
            )
        )
        groups.append(
            _statistics_group(
                source_kind,
                coordinate_space,
                "post_activation",
                activate_gaussian_params(raw),
                **metadata,
            )
        )


def add_difference_groups(
    groups: List[Dict[str, Any]],
    fitted_spaces: Mapping[str, Mapping[str, torch.Tensor]],
    baseline_spaces: Mapping[str, Mapping[str, torch.Tensor]],
    **metadata,
) -> None:
    for coordinate_space in STATISTICS_SPACES:
        fitted_raw = fitted_spaces[coordinate_space]
        baseline_raw = baseline_spaces[coordinate_space]
        validate_statistics_gaussians(fitted_raw)
        validate_statistics_gaussians(baseline_raw)
        raw_difference = {
            key: fitted_raw[key] - baseline_raw[key]
            for key in STATISTICS_PARAMETERS
        }
        fitted_activated = activate_gaussian_params(fitted_raw)
        baseline_activated = activate_gaussian_params(baseline_raw)
        activated_difference = {
            key: fitted_activated[key] - baseline_activated[key]
            for key in STATISTICS_PARAMETERS
        }
        groups.append(
            _statistics_group(
                "paired_differences",
                coordinate_space,
                "raw",
                raw_difference,
                **metadata,
            )
        )
        groups.append(
            _statistics_group(
                "paired_differences",
                coordinate_space,
                "post_activation",
                activated_difference,
                **metadata,
            )
        )


def load_statistics_resolution(
    dataset_root: Path,
    scene: str,
    resolution: int,
) -> Dict[str, Any]:
    paths = scene_resolution_paths(dataset_root, scene, resolution)
    checkpoint_path = latest_checkpoint(paths["nerfstudio_dir"])
    processed, scaler, gaussian_count = load_gaussian_params(
        checkpoint_path,
        torch.device("cpu"),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "scaler": scaler,
        "gaussian_count": gaussian_count,
        "processed": processed,
        "checkpoint": gaussian_checkpoint_space(processed, scaler),
    }


def statistics_cache_path(
    cache_root: Path,
    scene: str,
    source_resolution: int,
    target_resolution: int,
) -> Path:
    return (
        cache_root
        / scene
        / ("ir%d_tr%d" % (source_resolution, target_resolution))
        / "matching_target.pt"
    )


def _source_attributes(
    gs_params: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(gs_params.items())
    }


def load_compatible_statistics_cache(
    path: Path,
    scene: str,
    source_resolution: int,
    target_resolution: int,
    expected_source: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Matching cache payload is not a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Matching cache is missing metadata")

    expected_metadata = {
        "version": 1,
        "scene_name": scene,
        "input_resolution": int(source_resolution),
        "target_resolution": int(target_resolution),
        "source_attributes": _source_attributes(expected_source),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError("Matching cache metadata mismatch for %s" % key)

    fitted = payload.get("target_gs")
    if not isinstance(fitted, dict) or set(fitted) != set(expected_source):
        raise ValueError("Matching cache attributes do not match the source")
    result = {}
    for key, source_value in expected_source.items():
        value = fitted[key]
        if not torch.is_tensor(value):
            raise ValueError("Matching cache parameter %s is not a tensor" % key)
        if value.shape != source_value.shape or value.dtype != source_value.dtype:
            raise ValueError("Matching cache parameter %s shape or dtype mismatch" % key)
        result[key] = value.detach().cpu()
    validate_statistics_gaussians(result)
    return result


def _artifact_fingerprint(path: Path) -> Dict[str, Any]:
    fingerprint = {"path": str(path)}
    try:
        stat = path.stat()
    except FileNotFoundError:
        fingerprint["exists"] = False
    else:
        fingerprint.update(
            {
                "exists": True,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint


def statistics_input_fingerprint(args, scene: str) -> Dict[str, Any]:
    checkpoints = {}
    for resolution in args.resolutions:
        paths = scene_resolution_paths(args.dataset_root, scene, resolution)
        try:
            path = latest_checkpoint(paths["nerfstudio_dir"])
        except Exception as exc:
            checkpoints[str(resolution)] = {
                "error": "%s:%s" % (type(exc).__name__, exc)
            }
        else:
            checkpoints[str(resolution)] = _artifact_fingerprint(path)

    caches = {}
    if len(set(args.resolutions)) >= 2:
        for direction in ("lr_to_hr", "hr_to_lr"):
            source_resolution, target_resolution = _direction_resolutions(
                direction,
                args.resolutions,
            )
            caches[direction] = _artifact_fingerprint(
                statistics_cache_path(
                    cache_root_path(args),
                    scene,
                    source_resolution,
                    target_resolution,
                )
            )
    return {
        "record_version": STATISTICS_RECORD_VERSION,
        "resolutions": list(args.resolutions),
        "checkpoints": checkpoints,
        "caches": caches,
    }


def _read_latest_statistics_records(path: Path) -> Dict[str, Dict[str, Any]]:
    latest = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in %s line %d: %s" % (path, line_number, exc)
                ) from exc
            scene = record.get("scene")
            if scene:
                latest[str(scene)] = record
    return latest


def process_statistics_scene(
    args,
    scene: str,
    fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    missing_caches: List[Dict[str, Any]] = []
    resolutions = {}

    for resolution in args.resolutions:
        try:
            bundle = load_statistics_resolution(
                args.dataset_root,
                scene,
                resolution,
            )
            resolutions[resolution] = bundle
            add_distribution_groups(
                groups,
                "nerfstudio",
                {
                    "checkpoint": bundle["checkpoint"],
                    "processed": bundle["processed"],
                },
                resolution=resolution,
            )
        except Exception as exc:
            errors.append(
                {
                    "source_kind": "nerfstudio",
                    "resolution": str(resolution),
                    "reason": "%s:%s" % (type(exc).__name__, exc),
                }
            )

    if len(set(args.resolutions)) >= 2:
        for direction in ("lr_to_hr", "hr_to_lr"):
            source_resolution, target_resolution = _direction_resolutions(
                direction,
                args.resolutions,
            )
            cache_path = statistics_cache_path(
                cache_root_path(args),
                scene,
                source_resolution,
                target_resolution,
            )
            if not cache_path.is_file():
                missing_caches.append(
                    {
                        "direction": direction,
                        "source_resolution": source_resolution,
                        "target_resolution": target_resolution,
                        "cache_path": str(cache_path),
                    }
                )
                continue
            if (
                source_resolution not in resolutions
                or target_resolution not in resolutions
            ):
                errors.append(
                    {
                        "source_kind": "pretrain",
                        "direction": direction,
                        "reason": "required_resolution_checkpoint_failed",
                    }
                )
                continue

            try:
                source_bundle = resolutions[source_resolution]
                target_bundle = resolutions[target_resolution]
                expected_source = convert_gaussian_frame(
                    source_bundle["processed"],
                    source_bundle["scaler"],
                    target_bundle["scaler"],
                )
                fitted_target = load_compatible_statistics_cache(
                    cache_path,
                    scene,
                    source_resolution,
                    target_resolution,
                    expected_source,
                )
                fitted_target_spaces = {
                    "processed": fitted_target,
                    "checkpoint": gaussian_checkpoint_space(
                        fitted_target,
                        target_bundle["scaler"],
                    ),
                }
                add_distribution_groups(
                    groups,
                    "pretrain",
                    fitted_target_spaces,
                    direction=direction,
                    source_resolution=source_resolution,
                    target_resolution=target_resolution,
                    resolution=target_resolution,
                )

                fitted_source = convert_gaussian_frame(
                    fitted_target,
                    target_bundle["scaler"],
                    source_bundle["scaler"],
                )
                fitted_source_spaces = {
                    "processed": fitted_source,
                    "checkpoint": gaussian_checkpoint_space(
                        fitted_source,
                        source_bundle["scaler"],
                    ),
                }
                baseline_spaces = {
                    "processed": source_bundle["processed"],
                    "checkpoint": source_bundle["checkpoint"],
                }
                add_difference_groups(
                    groups,
                    fitted_source_spaces,
                    baseline_spaces,
                    direction=direction,
                    source_resolution=source_resolution,
                    target_resolution=target_resolution,
                    resolution=source_resolution,
                )
            except Exception as exc:
                errors.append(
                    {
                        "source_kind": "pretrain",
                        "direction": direction,
                        "reason": "%s:%s" % (type(exc).__name__, exc),
                    }
                )

    return {
        "record_version": STATISTICS_RECORD_VERSION,
        "scene": scene,
        "status": "error" if errors else "ok",
        "input_fingerprint": fingerprint,
        "groups": groups,
        "missing_caches": missing_caches,
        "errors": errors,
    }


def _group_output_key(group: Mapping[str, Any]) -> str:
    if group["source_kind"] == "nerfstudio":
        return str(group["resolution"])
    return str(group["direction"])


def aggregate_statistics_records(
    records: Sequence[Mapping[str, Any]],
    scene_count: int,
) -> Dict[str, Any]:
    accumulators = {}
    for record in records:
        for group in record.get("groups", []):
            identity = (
                group["coordinate_space"],
                group["representation"],
                group["source_kind"],
                _group_output_key(group),
            )
            accumulator = accumulators.setdefault(
                identity,
                {
                    "metadata": {
                        key: group[key]
                        for key in (
                            "resolution",
                            "direction",
                            "source_resolution",
                            "target_resolution",
                        )
                        if key in group
                    },
                    "scenes": [],
                    "parameters": defaultdict(
                        lambda: {
                            "channel_shape": None,
                            "means": [],
                            "stds": [],
                            "gaussian_count": 0,
                        }
                    ),
                },
            )
            accumulator["scenes"].append(record["scene"])
            for key, parameter in group["parameters"].items():
                parameter_accumulator = accumulator["parameters"][key]
                channel_shape = parameter["channel_shape"]
                if parameter_accumulator["channel_shape"] is None:
                    parameter_accumulator["channel_shape"] = channel_shape
                elif parameter_accumulator["channel_shape"] != channel_shape:
                    raise ValueError(
                        "Statistics channel shape mismatch for %s" % (identity,)
                    )
                parameter_accumulator["means"].append(
                    torch.as_tensor(parameter["mean"], dtype=torch.float64)
                )
                parameter_accumulator["stds"].append(
                    torch.as_tensor(parameter["std"], dtype=torch.float64)
                )
                parameter_accumulator["gaussian_count"] += int(
                    parameter["gaussian_count"]
                )

    spaces = {
        space: {
            representation: {
                "nerfstudio": {},
                "pretrain": {},
                "paired_differences": {},
            }
            for representation in STATISTICS_REPRESENTATIONS
        }
        for space in STATISTICS_SPACES
    }
    for identity, accumulator in accumulators.items():
        coordinate_space, representation, source_kind, output_key = identity
        parameters = {}
        for key, parameter_accumulator in accumulator["parameters"].items():
            parameters[key] = {
                "channel_shape": parameter_accumulator["channel_shape"],
                "scene_count": len(parameter_accumulator["means"]),
                "gaussian_count": parameter_accumulator["gaussian_count"],
                "mean": torch.stack(
                    parameter_accumulator["means"],
                    dim=0,
                ).mean(dim=0).tolist(),
                "std": torch.stack(
                    parameter_accumulator["stds"],
                    dim=0,
                ).mean(dim=0).tolist(),
            }
        spaces[coordinate_space][representation][source_kind][output_key] = {
            **accumulator["metadata"],
            "scene_count": len(accumulator["scenes"]),
            "scenes": accumulator["scenes"],
            "parameters": parameters,
        }

    return {
        "version": STATISTICS_RECORD_VERSION,
        "aggregation": {
            "scene_weighting": "equal",
            "per_scene_mean": "mean over Gaussian rows",
            "per_scene_std": "population standard deviation (ddof=0)",
            "cross_scene_mean": "arithmetic mean of per-scene means",
            "cross_scene_std": "arithmetic mean of per-scene standard deviations",
            "paired_difference": "fitted minus baseline",
        },
        "activation": {
            "means": "identity",
            "scales": "exp",
            "opacities": "sigmoid",
            "quats": "l2_normalize",
            "features_dc": "identity",
            "features_rest": "identity",
        },
        "candidate_scene_count": scene_count,
        "recorded_scene_count": len(records),
        "successful_scene_count": sum(
            record.get("status") == "ok" for record in records
        ),
        "errors": [
            {
                "scene": record["scene"],
                "errors": record.get("errors", []),
            }
            for record in records
            if record.get("errors")
        ],
        "missing_caches": [
            {
                "scene": record["scene"],
                "caches": record.get("missing_caches", []),
            }
            for record in records
            if record.get("missing_caches")
        ],
        "spaces": spaces,
    }


def run_statistics(args) -> Dict[str, Any]:
    scene_list = (
        getattr(args, "scene_list", None)
        if getattr(args, "scene_list", None) is not None
        else valid_scene_list_path(args)
    )
    scenes = read_scene_list(scene_list)
    records_path = statistics_records_path(args)
    latest_records = _read_latest_statistics_records(records_path)
    fingerprints = {
        scene: statistics_input_fingerprint(args, scene)
        for scene in scenes
    }
    attempted = 0

    for scene in tqdm(scenes, desc="statistics"):
        fingerprint = fingerprints[scene]
        existing = latest_records.get(scene)
        same_input = (
            existing is not None
            and existing.get("input_fingerprint") == fingerprint
        )
        force = getattr(args, "force_statistics", False)
        retry_errors = getattr(args, "retry_errors", False)
        if (
            same_input
            and not force
            and (
                existing.get("status") == "ok"
                or not retry_errors
            )
        ):
            continue
        if args.max_scenes is not None and attempted >= args.max_scenes:
            break
        attempted += 1

        record = process_statistics_scene(args, scene, fingerprint)
        append_jsonl(records_path, record)
        latest_records[scene] = record

    effective_records = [
        latest_records[scene]
        for scene in scenes
        if scene in latest_records
        and latest_records[scene].get("input_fingerprint") == fingerprints[scene]
    ]
    report = aggregate_statistics_records(effective_records, len(scenes))
    report["scene_list"] = str(scene_list)
    report["records_path"] = str(records_path)
    output_path = statistics_json_path(args)
    atomic_write_json(output_path, report)
    print("Gaussian statistics records: %s" % records_path)
    print("Gaussian statistics report: %s" % output_path)
    return report


def run_all(args) -> None:
    run_validate(args)
    run_evaluate(args)
    run_select(args)
    run_pretrain(args)
    run_statistics(args)


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        metavar="RESOLUTION",
    )
    parser.add_argument(
        "--expected_images", type=int, default=DEFAULT_EXPECTED_IMAGES
    )
    parser.add_argument("--max_scenes", type=int, default=None)


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--structure_report", type=Path, default=None)
    parser.add_argument("--structural_scene_list", type=Path, default=None)
    parser.add_argument("--metrics_csv", type=Path, default=None)
    parser.add_argument("--selection_report", type=Path, default=None)
    parser.add_argument("--valid_scene_list", type=Path, default=None)
    parser.add_argument("--pretrain_status", type=Path, default=None)
    parser.add_argument("--cache_root", type=Path, default=None)
    parser.add_argument("--statistics_json", type=Path, default=None)
    parser.add_argument("--statistics_records", type=Path, default=None)


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_chunk_size", type=int, default=16)
    parser.add_argument("--retry_errors", action="store_true")


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min_psnr", type=float, required=True)
    parser.add_argument("--min_gs_count", type=int, required=True)


def add_pretraining_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--direction",
        choices=("none", "lr_to_hr", "hr_to_lr", "both"),
        default="none",
    )
    parser.add_argument("--matching_steps", type=int, default=2000)
    parser.add_argument("--matching_images_per_step", type=int, default=32)
    parser.add_argument("--matching_log_interval", type=int, default=20)
    parser.add_argument("--matching_preview_interval", type=int, default=200)
    parser.add_argument("--matching_l1_weight", type=float, default=1.0)
    parser.add_argument("--matching_lpips_weight", type=float, default=1.0)
    parser.add_argument("--force_refit", action="store_true")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)


def validate_arguments(args, parser: argparse.ArgumentParser) -> None:
    if any(resolution <= 0 for resolution in args.resolutions):
        parser.error("--resolutions must be positive")
    if len(args.resolutions) != len(set(args.resolutions)):
        parser.error("--resolutions must not contain duplicates")
    if args.expected_images <= 0:
        parser.error("--expected_images must be positive")
    if args.max_scenes is not None and args.max_scenes <= 0:
        parser.error("--max_scenes must be positive")
    if hasattr(args, "render_chunk_size") and args.render_chunk_size <= 0:
        parser.error("--render_chunk_size must be positive")
    if hasattr(args, "min_psnr") and not math.isfinite(args.min_psnr):
        parser.error("--min_psnr must be finite")
    if hasattr(args, "min_gs_count") and args.min_gs_count < 0:
        parser.error("--min_gs_count must be non-negative")
    if hasattr(args, "matching_steps") and args.matching_steps <= 0:
        parser.error("--matching_steps must be positive")
    if (
        hasattr(args, "matching_images_per_step")
        and args.matching_images_per_step < 0
    ):
        parser.error("--matching_images_per_step must be non-negative")
    if hasattr(args, "matching_l1_weight"):
        if args.matching_l1_weight < 0 or args.matching_lpips_weight < 0:
            parser.error("matching loss weights must be non-negative")
        if args.matching_l1_weight == 0 and args.matching_lpips_weight == 0:
            parser.error("at least one matching loss weight must be non-zero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess native-resolution Gaussian SR datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    add_layout_arguments(validate_parser)
    add_output_arguments(validate_parser)
    validate_parser.add_argument("--candidate_scene_list", type=Path, default=None)
    validate_parser.set_defaults(func=run_validate)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_layout_arguments(evaluate_parser)
    add_output_arguments(evaluate_parser)
    add_evaluation_arguments(evaluate_parser)
    evaluate_parser.add_argument("--scene_list", type=Path, default=None)
    evaluate_parser.set_defaults(func=run_evaluate)

    select_parser = subparsers.add_parser("select")
    add_layout_arguments(select_parser)
    add_output_arguments(select_parser)
    add_selection_arguments(select_parser)
    select_parser.add_argument("--scene_list", type=Path, default=None)
    select_parser.set_defaults(func=run_select)

    pretrain_parser = subparsers.add_parser("pretrain")
    add_layout_arguments(pretrain_parser)
    add_output_arguments(pretrain_parser)
    add_evaluation_arguments(pretrain_parser)
    add_pretraining_arguments(pretrain_parser)
    pretrain_parser.add_argument("--scene_list", type=Path, default=None)
    pretrain_parser.set_defaults(func=run_pretrain)

    statistics_parser = subparsers.add_parser("statistics")
    add_layout_arguments(statistics_parser)
    add_output_arguments(statistics_parser)
    statistics_parser.add_argument("--retry_errors", action="store_true")
    statistics_parser.add_argument("--force_statistics", action="store_true")
    statistics_parser.add_argument("--scene_list", type=Path, default=None)
    statistics_parser.set_defaults(func=run_statistics)

    all_parser = subparsers.add_parser("all")
    add_layout_arguments(all_parser)
    add_output_arguments(all_parser)
    add_evaluation_arguments(all_parser)
    add_selection_arguments(all_parser)
    all_parser.add_argument("--force_statistics", action="store_true")
    add_pretraining_arguments(all_parser)
    all_parser.add_argument("--candidate_scene_list", type=Path, default=None)
    all_parser.set_defaults(func=run_all)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    args.func(args);


if __name__ == "__main__":
    main()

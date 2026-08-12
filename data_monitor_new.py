#!/usr/bin/env python3

import argparse
import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import gs_utils
from utils.metrics import psnr, ssim
from utils.transform_utils import MinMaxScaler


DEFAULT_RESOLUTIONS = (512, 256, 128)
EXPECTED_IMAGE_COUNT = 128
CKPT_RELATIVE_PATH = Path("splatfacto/nerfstudio_models/step-000015001.ckpt")
CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"
OUTPUT_COLUMNS = [
    "scene",
    "status",
    "skip_reason",
    "resolution",
    "image_dir",
    "ckpt_path",
    "num_images",
    "splat_count",
    "psnr",
    "ssim",
    "lpips",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze native-resolution nerfstudio dataset statistics."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default="/project/ricky/splatformer-sr-data/train-set/objaverse/",
        help=(
            "Root containing <resolution>/colmap and <resolution>/nerfstudio "
            "directories."
        ),
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        metavar="RESOLUTION",
        help="Native resolutions to analyze (default: 512 256 128).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default="objaverse_stats_new.csv",
        help="Path to write the scene-level CSV report."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for rendering and metric computation. Default: cuda",
    )
    parser.add_argument(
        "--scene_list",
        type=Path,
        default=None,
        help="Optional text file of scene ids to analyze, one per line.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional cap on the number of newly processed valid scenes.",
    )
    parser.add_argument(
        "--disable_metrics",
        action="store_true",
        help="Skip PSNR, SSIM, and LPIPS calculation and only report splat statistics.",
    )
    parser.add_argument("--disable_psnr", action="store_true", help="Skip PSNR calculation.")
    parser.add_argument("--disable_ssim", action="store_true", help="Skip SSIM calculation.")
    parser.add_argument("--disable_lpips", action="store_true", help="Skip LPIPS calculation.")
    args = parser.parse_args()
    if any(resolution <= 0 for resolution in args.resolutions):
        parser.error("--resolutions values must all be positive integers.")
    if len(args.resolutions) != len(set(args.resolutions)):
        parser.error("--resolutions must not contain duplicate values.")
    return args


def require_cuda_device(device: str) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(
            "--device must be a CUDA device because gs_utils rasterization is CUDA-only in this repo."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but the analysis pipeline requires it.")
    return torch_device


def resolve_enabled_metrics(args: argparse.Namespace) -> List[str]:
    if args.disable_metrics:
        return []

    enabled = []
    if not args.disable_psnr:
        enabled.append("psnr")
    if not args.disable_ssim:
        enabled.append("ssim")
    if not args.disable_lpips:
        enabled.append("lpips")
    return enabled


def load_scene_filter(scene_list_path: Optional[Path]) -> Optional[Set[str]]:
    if scene_list_path is None:
        return None
    with scene_list_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def list_pngs(image_dir: Path) -> List[Path]:
    return sorted([path for path in image_dir.iterdir() if path.suffix.lower() == ".png"])


def read_image(path: Path) -> torch.Tensor:
    image = np.array(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    if image.ndim != 3:
        raise ValueError("Expected HWC image at %s, found shape %s" % (path, tuple(image.shape)))
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha
    elif image.shape[2] != 3:
        raise ValueError("Expected RGB/RGBA image at %s, found channel count %s" % (path, image.shape[2]))
    return image


def load_camera_metadata(nerfstudio_scene_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    camera_path = nerfstudio_scene_dir / CAMERA_METADATA_NAME
    if not camera_path.is_file():
        raise FileNotFoundError("Missing required camera metadata: %s" % camera_path)

    with camera_path.open("rb") as f:
        meta = pickle.load(f)

    required_keys = ("train_camera_to_worlds", "fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise KeyError("Camera metadata %s is missing keys: %s" % (camera_path, missing))

    cameras = {
        "camera_to_worlds": meta["train_camera_to_worlds"].to(device),
        "fx": torch.as_tensor(meta["fx"], device=device),
        "fy": torch.as_tensor(meta["fy"], device=device),
        "cx": torch.as_tensor(meta["cx"], device=device),
        "cy": torch.as_tensor(meta["cy"], device=device),
        "width": torch.as_tensor(meta["width"], device=device),
        "height": torch.as_tensor(meta["height"], device=device),
        "background_color": torch.zeros(3, device=device),
    }
    return cameras


def tensor_any_over_dims(tensor: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    result = tensor
    for dim in sorted(dims, reverse=True):
        result = result.any(dim=dim)
    return result


def load_gaussian_params(ckpt_path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    gs_params = {k.replace("_model.gauss_params.", ""): v for k, v in ckpt.items() if "gauss_params" in k}

    required_keys = {"means", "scales", "quats", "features_dc", "opacities"}
    missing = sorted(required_keys - set(gs_params.keys()))
    if missing:
        raise KeyError("Checkpoint %s is missing required Gaussian keys: %s" % (ckpt_path, missing))

    select = torch.ones(gs_params["means"].shape[0], dtype=torch.bool)
    for key, value in gs_params.items():
        if not torch.is_tensor(value) or value.shape[0] != select.shape[0]:
            continue
        if key == "features_rest":
            has_nan = torch.isnan(value.sum(dim=1)).any(dim=1)
        else:
            has_nan = torch.isnan(value)
            reduce_dims = tuple(range(1, value.ndim))
            if reduce_dims:
                has_nan = tensor_any_over_dims(has_nan, reduce_dims)
        select &= ~has_nan

    filtered = {}
    for key, value in gs_params.items():
        if torch.is_tensor(value) and value.shape[0] == select.shape[0]:
            filtered[key] = value[select]
        else:
            filtered[key] = value

    scaler = MinMaxScaler()
    filtered["means"] = scaler.fit_transform(filtered["means"])
    filtered["scales"] = filtered["scales"] + torch.log(scaler.scale_)

    inf_mask = torch.isinf(filtered["scales"]).any(dim=1)
    inrange_mask = torch.all((filtered["means"] >= 0) & (filtered["means"] <= 1), dim=1)
    valid_mask = (~inf_mask) & inrange_mask
    for key, value in list(filtered.items()):
        if torch.is_tensor(value) and value.shape[0] == valid_mask.shape[0]:
            filtered[key] = value[valid_mask].to(device)

    return filtered, int(filtered["means"].shape[0])


def validate_scene(
    scene: str,
    dataset_root: Path,
    resolutions: Iterable[int],
    needs_metrics: bool,
) -> Tuple[bool, str]:
    for resolution in resolutions:
        resolution_root = dataset_root / str(resolution)
        colmap_scene_dir = resolution_root / "colmap" / scene
        if not colmap_scene_dir.is_dir():
            return False, "missing_colmap_scene_%d" % resolution

        nerfstudio_scene_dir = resolution_root / "nerfstudio" / scene
        if not nerfstudio_scene_dir.is_dir():
            return False, "missing_nerfstudio_scene_%d" % resolution

        image_dir = colmap_scene_dir / "images"
        if not image_dir.is_dir():
            return False, "missing_images_%d" % resolution
        if len(list_pngs(image_dir)) != EXPECTED_IMAGE_COUNT:
            return False, "bad_image_count_%d" % resolution

        ckpt_path = nerfstudio_scene_dir / CKPT_RELATIVE_PATH
        if not ckpt_path.is_file():
            return False, "missing_ckpt_%d" % resolution

        if needs_metrics:
            camera_path = nerfstudio_scene_dir / "splatfacto" / CAMERA_METADATA_NAME
            if not camera_path.is_file():
                return False, "missing_camera_metadata_%d" % resolution

    return True, "valid"


def compute_metrics(
    pred_images: torch.Tensor,
    gt_images: torch.Tensor,
    enabled_metrics: List[str],
    lpips_fn,
) -> Dict[str, Optional[float]]:
    results = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
    }
    if not enabled_metrics:
        return results

    pred_float = torch.clamp(pred_images, 0.0, 1.0)
    gt_float = torch.clamp(gt_images, 0.0, 1.0)

    if "psnr" in enabled_metrics:
        results["psnr"] = float(psnr(pred_float, gt_float).mean().item())
    if "ssim" in enabled_metrics:
        results["ssim"] = float(
            ssim(pred_float.permute(0, 3, 1, 2), gt_float.permute(0, 3, 1, 2), window_size=11, size_average=False)
            .mean()
            .item()
        )
    if "lpips" in enabled_metrics:
        if lpips_fn is None:
            raise RuntimeError("LPIPS was requested but the LPIPS model was not initialized.")
        results["lpips"] = float(
            lpips_fn(pred_float.permute(0, 3, 1, 2), gt_float.permute(0, 3, 1, 2), normalize=True)
            .mean()
            .item()
        )
    return results


def make_skip_row(scene: str, reason: str) -> Dict[str, object]:
    return {
        "scene": scene,
        "status": "skipped",
        "skip_reason": reason,
        "resolution": "",
        "image_dir": "",
        "ckpt_path": "",
        "num_images": "",
        "splat_count": "",
        "psnr": "",
        "ssim": "",
        "lpips": "",
    }


def analyze_resolution(
    scene: str,
    resolution: int,
    dataset_root: Path,
    device: torch.device,
    enabled_metrics: List[str],
    lpips_fn,
) -> Dict[str, object]:
    resolution_root = dataset_root / str(resolution)
    image_dir = resolution_root / "colmap" / scene / "images"
    image_paths = list_pngs(image_dir)
    nerfstudio_scene_dir = resolution_root / "nerfstudio" / scene
    ckpt_path = nerfstudio_scene_dir / CKPT_RELATIVE_PATH

    gs_params, splat_count = load_gaussian_params(ckpt_path, device)
    row = {
        "scene": scene,
        "status": "processed",
        "skip_reason": "",
        "resolution": resolution,
        "image_dir": str(image_dir),
        "ckpt_path": str(ckpt_path),
        "num_images": len(image_paths),
        "splat_count": splat_count,
        "psnr": "",
        "ssim": "",
        "lpips": "",
    }

    if not enabled_metrics:
        return row

    cameras = load_camera_metadata(nerfstudio_scene_dir / "splatfacto", device)
    if len(image_paths) != len(cameras["camera_to_worlds"]):
        raise ValueError(
            "Scene %s resolution %d: image count %d does not match camera count %d"
            % (scene, resolution, len(image_paths), len(cameras["camera_to_worlds"]))
        )

    gt_images = torch.stack([read_image(path) for path in image_paths], dim=0).to(device)
    with torch.no_grad():
        pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_params, cameras)
    pred_images = torch.stack(pred_images, dim=0)

    metrics = compute_metrics(pred_images, gt_images, enabled_metrics, lpips_fn)
    row.update(metrics)
    return row


def iter_candidate_scenes(
    dataset_root: Path,
    resolutions: Iterable[int],
    requested_scenes: Optional[Set[str]],
) -> Iterable[str]:
    scenes = set()
    for resolution in resolutions:
        resolution_root = dataset_root / str(resolution)
        for output_name in ("colmap", "nerfstudio"):
            output_root = resolution_root / output_name
            if output_root.is_dir():
                scenes.update(path.name for path in output_root.iterdir() if path.is_dir())

    sorted_scenes = sorted(scenes)
    if requested_scenes is not None:
        sorted_scenes = [scene for scene in sorted_scenes if scene in requested_scenes]
    return sorted_scenes


def normalize_existing_row(row: Dict[str, str]) -> Dict[str, object]:
    normalized = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
    if normalized["status"] == "":
        normalized["status"] = "processed"
    return normalized


def load_existing_rows(output_csv: Path) -> List[Dict[str, object]]:
    if not output_csv.is_file():
        return []
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [normalize_existing_row(row) for row in reader]


def get_completed_or_skipped_scenes(
    rows: List[Dict[str, object]], resolutions: Iterable[int]
) -> Set[str]:
    required_resolutions = set(resolutions)
    processed_resolutions = defaultdict(set)
    skipped_scenes = set()

    for row in rows:
        scene = str(row.get("scene", "")).strip()
        if not scene:
            continue
        status = str(row.get("status", "processed")).strip() or "processed"
        if status == "skipped":
            skipped_scenes.add(scene)
        elif status == "processed":
            try:
                resolution = int(row.get("resolution", ""))
            except (TypeError, ValueError):
                continue
            processed_resolutions[scene].add(resolution)

    completed_scenes = {
        scene
        for scene, completed_resolutions in processed_resolutions.items()
        if required_resolutions.issubset(completed_resolutions)
    }
    return skipped_scenes | completed_scenes


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def summarize_metric(values: List[object]) -> Optional[float]:
    filtered = [float(value) for value in values if value not in (None, "")]
    if not filtered:
        return None
    return float(np.mean(np.array(filtered, dtype=np.float64)))


def save_histograms(rows: List[Dict[str, object]], output_csv: Path) -> None:
    processed_rows = [row for row in rows if row.get("status") == "processed" and row.get("splat_count") not in ("", None)]
    if not processed_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("Skipping histogram export: matplotlib is not installed (%s)" % exc)
        return

    def choose_bins(values: List[float]) -> int:
        return min(80, max(20, int(np.sqrt(len(values)) * 4)))

    def save_distribution_plots(values: List[float], stem: str, title_prefix: str, count_label: str) -> None:
        if not values:
            return
        bins = choose_bins(values)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins, color="#4C72B0", edgecolor="black")
        ax.set_title("%s Histogram" % title_prefix)
        ax.set_xlabel("Splat count")
        ax.set_ylabel(count_label)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / (stem + "_hist.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins, cumulative=True, color="#C44E52", edgecolor="black")
        ax.set_title("%s Accumulated Counts" % title_prefix)
        ax.set_xlabel("Splat count")
        ax.set_ylabel("Accumulated %s" % count_label.lower())
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / (stem + "_accumulated.png"), dpi=200)
        plt.close(fig)

    figure_dir = output_csv.parent / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in processed_rows:
        grouped[int(row["resolution"])].append(float(row["splat_count"]))

    for resolution in sorted(grouped, reverse=True):
        values = grouped.get(resolution, [])
        save_distribution_plots(
            values,
            "splat_count_res-%d" % resolution,
            "Splat Count %dpx" % resolution,
            "Scenes",
        )

    all_values = [float(row["splat_count"]) for row in processed_rows]
    save_distribution_plots(all_values, "splat_count_all", "Splat Count All Resolutions", "Rows")


def print_summary(
    rows: List[Dict[str, object]],
    skip_reasons: Counter,
    enabled_metrics: List[str],
    reused_scene_count: int,
) -> None:
    processed_rows = [row for row in rows if row.get("status") == "processed"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]

    print("Processed %d resolution rows across %d valid scenes." % (len(processed_rows), len({row["scene"] for row in processed_rows})))
    print("Skipped scenes recorded: %d" % len({row["scene"] for row in skipped_rows}))
    print("Scenes reused from existing CSV: %d" % reused_scene_count)
    print("New skip reasons this run: %d" % sum(skip_reasons.values()))
    for reason, count in sorted(skip_reasons.items()):
        print("  %s: %d" % (reason, count))

    grouped = defaultdict(list)
    for row in processed_rows:
        grouped[int(row["resolution"])].append(row)

    for resolution in sorted(grouped, reverse=True):
        resolution_rows = grouped[resolution]
        print("\nResolution %dpx" % resolution)
        if not resolution_rows:
            print("  No valid rows.")
            continue

        splat_counts = np.array(
            [float(row["splat_count"]) for row in resolution_rows],
            dtype=np.float64,
        )
        print("  scene_count: %d" % len(resolution_rows))
        print(
            "  splat_count mean/min/max: %.2f / %.0f / %.0f"
            % (splat_counts.mean(), splat_counts.min(), splat_counts.max())
        )

        for metric_name in ("psnr", "ssim", "lpips"):
            if metric_name not in enabled_metrics:
                print("  %s: disabled" % metric_name)
                continue
            metric_mean = summarize_metric([row[metric_name] for row in resolution_rows])
            if metric_mean is None:
                print("  %s: unavailable" % metric_name)
            elif metric_name == "psnr":
                print("  psnr mean: %.4f" % metric_mean)
            else:
                print("  %s mean: %.6f" % (metric_name, metric_mean))



def main() -> None:
    args = parse_args()
    device = require_cuda_device(args.device)
    enabled_metrics = resolve_enabled_metrics(args)
    needs_metrics = bool(enabled_metrics)

    lpips_fn = None
    if "lpips" in enabled_metrics:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS is enabled but the lpips package is not installed. "
                "Use --disable_lpips or --disable_metrics to run without it."
            ) from exc
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)

    requested_scenes = load_scene_filter(args.scene_list)
    existing_rows = load_existing_rows(args.output_csv)
    completed_or_skipped_scenes = get_completed_or_skipped_scenes(
        existing_rows, args.resolutions
    )

    candidate_scenes = list(
        iter_candidate_scenes(args.dataset_root, args.resolutions, requested_scenes)
    )
    pending_scenes = [scene for scene in candidate_scenes if scene not in completed_or_skipped_scenes]
    reused_scene_count = len(candidate_scenes) - len(pending_scenes)

    new_rows = []
    skip_reasons = Counter()

    valid_scene_count = 0
    for scene in tqdm(pending_scenes, desc="Scenes"):
        is_valid, reason = validate_scene(
            scene, args.dataset_root, args.resolutions, needs_metrics
        )
        if not is_valid:
            skip_reasons[reason] += 1
            new_rows.append(make_skip_row(scene, reason))
            continue

        scene_rows = []
        try:
            for resolution in args.resolutions:
                row = analyze_resolution(
                    scene,
                    resolution,
                    args.dataset_root,
                    device,
                    enabled_metrics,
                    lpips_fn,
                )
                scene_rows.append(row)
        except Exception as exc:
            reason = "analysis_error:%s" % type(exc).__name__
            skip_reasons[reason] += 1
            print("Skipping scene %s: %s" % (scene, exc))
            new_rows.append(make_skip_row(scene, reason))
            continue

        new_rows.extend(scene_rows)
        valid_scene_count += 1
        if args.max_scenes is not None and valid_scene_count >= args.max_scenes:
            break

    all_rows = existing_rows + new_rows
    write_csv(all_rows, args.output_csv)
    save_histograms(all_rows, args.output_csv)
    print_summary(all_rows, skip_reasons, enabled_metrics, reused_scene_count)
    print("\nWrote CSV: %s" % args.output_csv)
    print("Histogram directory: %s" % (args.output_csv.parent / "figure"))


if __name__ == "__main__":
    main()

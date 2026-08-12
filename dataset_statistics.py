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


EXPECTED_FACTORS = (1, 2, 4)
EXPECTED_IMAGE_COUNT = 128
FACTOR_TO_IMAGE_DIR = {
    1: "images",
    2: "images_2",
    4: "images_4",
}
CKPT_RELATIVE_PATH = Path("splatfacto/nerfstudio_models/step-000015001.ckpt")
CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"
OUTPUT_COLUMNS = [
    "scene",
    "status",
    "skip_reason",
    "factor",
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
        description="Analyze generated dataset statistics for multi-factor nerfstudio scenes."
    )
    parser.add_argument(
        "--colmap_root", 
        type=Path, 
        default="/project2/ricky/splatformer-data/train-set-512/objaverse/colmap/",
        help="Root directory of COLMAP scenes."
    )
    parser.add_argument(
        "--nerfstudio_root",
        type=Path,
        default="/project2/ricky/splatformer-data/train-set-512/objaverse/nerfstudio/",
        help="Root directory of nerfstudio scene outputs."
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default="objaverse_stats.csv",
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
    return parser.parse_args()


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


def validate_scene(scene: str, colmap_root: Path, nerfstudio_root: Path, needs_metrics: bool) -> Tuple[bool, str]:
    colmap_scene_dir = colmap_root / scene
    if not colmap_scene_dir.is_dir():
        return False, "missing_colmap_scene"

    nerfstudio_scene_root = nerfstudio_root / scene
    if not nerfstudio_scene_root.is_dir():
        return False, "missing_nerfstudio_scene"

    for factor in EXPECTED_FACTORS:
        image_dir = colmap_scene_dir / FACTOR_TO_IMAGE_DIR[factor]
        if not image_dir.is_dir():
            return False, "missing_%s" % FACTOR_TO_IMAGE_DIR[factor]
        if len(list_pngs(image_dir)) != EXPECTED_IMAGE_COUNT:
            return False, "bad_image_count_%s" % FACTOR_TO_IMAGE_DIR[factor]

        ckpt_path = nerfstudio_scene_root / ("df-%d" % factor) / CKPT_RELATIVE_PATH
        if not ckpt_path.is_file():
            return False, "missing_ckpt_df_%d" % factor

        if needs_metrics:
            camera_path = nerfstudio_scene_root / ("df-%d" % factor) / "splatfacto" / CAMERA_METADATA_NAME
            if not camera_path.is_file():
                return False, "missing_camera_metadata_df_%d" % factor

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
        "factor": "",
        "image_dir": "",
        "ckpt_path": "",
        "num_images": "",
        "splat_count": "",
        "psnr": "",
        "ssim": "",
        "lpips": "",
    }


def analyze_factor(
    scene: str,
    factor: int,
    colmap_root: Path,
    nerfstudio_root: Path,
    device: torch.device,
    enabled_metrics: List[str],
    lpips_fn,
) -> Dict[str, object]:
    image_dir = colmap_root / scene / FACTOR_TO_IMAGE_DIR[factor]
    image_paths = list_pngs(image_dir)
    ckpt_path = nerfstudio_root / scene / ("df-%d" % factor) / CKPT_RELATIVE_PATH

    gs_params, splat_count = load_gaussian_params(ckpt_path, device)
    row = {
        "scene": scene,
        "status": "processed",
        "skip_reason": "",
        "factor": factor,
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

    nerfstudio_scene_dir = nerfstudio_root / scene / ("df-%d" % factor) / "splatfacto"
    cameras = load_camera_metadata(nerfstudio_scene_dir, device)
    if len(image_paths) != len(cameras["camera_to_worlds"]):
        raise ValueError(
            "Scene %s df-%d: image count %d does not match camera count %d"
            % (scene, factor, len(image_paths), len(cameras["camera_to_worlds"]))
        )

    gt_images = torch.stack([read_image(path) for path in image_paths], dim=0).to(device)
    with torch.no_grad():
        pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_params, cameras)
    pred_images = torch.stack(pred_images, dim=0)

    metrics = compute_metrics(pred_images, gt_images, enabled_metrics, lpips_fn)
    row.update(metrics)
    return row


def iter_candidate_scenes(
    colmap_root: Path, nerfstudio_root: Path, requested_scenes: Optional[Set[str]]
) -> Iterable[str]:
    colmap_scenes = {path.name for path in colmap_root.iterdir() if path.is_dir()}
    nerfstudio_scenes = {path.name for path in nerfstudio_root.iterdir() if path.is_dir()}
    scenes = sorted(colmap_scenes | nerfstudio_scenes)
    if requested_scenes is not None:
        scenes = [scene for scene in scenes if scene in requested_scenes]
    return scenes


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


def get_completed_or_skipped_scenes(rows: List[Dict[str, object]]) -> Set[str]:
    scenes = set()
    for row in rows:
        scene = str(row.get("scene", "")).strip()
        if not scene:
            continue
        status = str(row.get("status", "processed")).strip() or "processed"
        if status in ("processed", "skipped"):
            scenes.add(scene)
    return scenes


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
        grouped[int(row["factor"])].append(float(row["splat_count"]))

    for factor in EXPECTED_FACTORS:
        values = grouped.get(factor, [])
        save_distribution_plots(values, "splat_count_df-%d" % factor, "Splat Count df-%d" % factor, "Scenes")

    all_values = [float(row["splat_count"]) for row in processed_rows]
    save_distribution_plots(all_values, "splat_count_all", "Splat Count All Factors", "Rows")


def print_summary(
    rows: List[Dict[str, object]],
    skip_reasons: Counter,
    enabled_metrics: List[str],
    reused_scene_count: int,
) -> None:
    processed_rows = [row for row in rows if row.get("status") == "processed"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]

    print("Processed %d factor rows across %d valid scenes." % (len(processed_rows), len({row["scene"] for row in processed_rows})))
    print("Skipped scenes recorded: %d" % len({row["scene"] for row in skipped_rows}))
    print("Scenes reused from existing CSV: %d" % reused_scene_count)
    print("New skip reasons this run: %d" % sum(skip_reasons.values()))
    for reason, count in sorted(skip_reasons.items()):
        print("  %s: %d" % (reason, count))

    grouped = defaultdict(list)
    for row in processed_rows:
        grouped[int(row["factor"])].append(row)

    for factor in EXPECTED_FACTORS:
        factor_rows = grouped[factor]
        print("\nFactor df-%d" % factor)
        if not factor_rows:
            print("  No valid rows.")
            continue

        splat_counts = np.array([float(row["splat_count"]) for row in factor_rows], dtype=np.float64)
        print("  scene_count: %d" % len(factor_rows))
        print(
            "  splat_count mean/min/max: %.2f / %.0f / %.0f"
            % (splat_counts.mean(), splat_counts.min(), splat_counts.max())
        )

        for metric_name in ("psnr", "ssim", "lpips"):
            if metric_name not in enabled_metrics:
                print("  %s: disabled" % metric_name)
                continue
            metric_mean = summarize_metric([row[metric_name] for row in factor_rows])
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
    completed_or_skipped_scenes = get_completed_or_skipped_scenes(existing_rows)

    candidate_scenes = list(iter_candidate_scenes(args.colmap_root, args.nerfstudio_root, requested_scenes))
    pending_scenes = [scene for scene in candidate_scenes if scene not in completed_or_skipped_scenes]
    reused_scene_count = len(candidate_scenes) - len(pending_scenes)

    new_rows = []
    skip_reasons = Counter()

    valid_scene_count = 0
    for scene in tqdm(pending_scenes, desc="Scenes"):
        is_valid, reason = validate_scene(scene, args.colmap_root, args.nerfstudio_root, needs_metrics)
        if not is_valid:
            skip_reasons[reason] += 1
            new_rows.append(make_skip_row(scene, reason))
            continue

        scene_rows = []
        try:
            for factor in EXPECTED_FACTORS:
                row = analyze_factor(
                    scene,
                    factor,
                    args.colmap_root,
                    args.nerfstudio_root,
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

"""Measure the rendering impact of Gaussian parameter augmentations."""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch

from utils.data_augmentation import (
    GAUSSIAN_PARAMETERS,
    jitter_gaussian_parameter,
    jitter_gaussian_parameters,
    rotate_camera_to_worlds,
    sample_uniform_z_rotation_quaternion,
    quaternion_inverse,
    rotate_gaussians,
    sample_uniform_rotation_quaternion,
)

from utils.gs_normalization import normalize_gaussian_quaternions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gin_file", action="append", required=True, help="Gin file; repeat to compose configs.")
    parser.add_argument("--gin_param", action="append", default=[], help="Gin override; repeat for multiple bindings.")
    parser.add_argument("--dataset_scope", choices=("train_dataset", "test_dataset"), default="test_dataset")
    parser.add_argument("--scene_name", default="", help="Scene name; an empty value selects the first scene.")
    parser.add_argument("--gs_resolution", choices=("source", "target"), default="source")
    parser.add_argument("--jitter_levels", type=float, nargs="+", default=(0.01, 0.05, 0.1))
    parser.add_argument("--trials", type=int, default=3, help="Trials per jitter level and rotation round trip.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pivot", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--chunk_size", type=int, default=0, help="Views per rendering/metric chunk; 0 uses all views.")
    parser.add_argument("--preview_views", type=int, default=4, help="Number of views in each preview; 0 disables previews.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=Path, default=Path("output_data_augmentation"))
    parser.add_argument("--mode", choices=("sweep", "configured"), default="sweep")
    parser.add_argument("--random_jitter", choices=("True", "False"), default="False")
    parser.add_argument("--random_rotate", choices=("True", "False"), default="False")
    parser.add_argument("--rotation_mode", choices=("full", "gravity_consistent"), default="full")
    parser.add_argument("--rotation_pivot", type=float, nargs=3, default=None)
    parser.add_argument("--jitter_max_levels", default=None, help="Python dictionary of per-attribute maximum jitter levels.")
    parser.add_argument("--mae_threshold", type=float, default=1e-5)
    parser.add_argument("--max_error_threshold", type=float, default=1e-2)
    args = parser.parse_args(argv)
    args.random_jitter = args.random_jitter == "True"
    args.random_rotate = args.random_rotate == "True"
    args.rotation_pivot = args.pivot if args.rotation_pivot is None else args.rotation_pivot
    try:
        levels = ast.literal_eval(args.jitter_max_levels) if args.jitter_max_levels is not None else dict.fromkeys(GAUSSIAN_PARAMETERS, .01)
        args.jitter_max_levels = {str(key): float(value) for key, value in levels.items()}
    except (ValueError, SyntaxError, TypeError, AttributeError):
        parser.error("--jitter_max_levels must be a dictionary of finite non-negative numbers")
    if set(args.jitter_max_levels) - set(GAUSSIAN_PARAMETERS) or any(not math.isfinite(v) or v < 0 for v in args.jitter_max_levels.values()):
        parser.error("Invalid jitter parameters or levels")
    if any(not math.isfinite(v) for v in args.rotation_pivot):
        parser.error("--rotation_pivot must contain finite values")
    if any(not math.isfinite(v) or v < 0 for v in (args.mae_threshold, args.max_error_threshold)):
        parser.error("Invariance thresholds must be finite and non-negative")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if any(not math.isfinite(level) or level < 0 for level in args.jitter_levels):
        parser.error("--jitter_levels must be finite and non-negative")
    if args.chunk_size < 0 or args.preview_views < 0:
        parser.error("--chunk_size and --preview_views must be non-negative")
    return args


def _render_views(gs_params, cameras, chunk_size, device):
    from utils import gpu_utils, gs_utils

    view_count = len(cameras["camera_to_worlds"])
    chunk_size = view_count if chunk_size <= 0 else min(chunk_size, view_count)
    rendered = []
    with torch.no_grad():
        for start in range(0, view_count, chunk_size):
            end = min(start + chunk_size, view_count)
            chunk_cameras = {
                key: value[start:end] if key == "camera_to_worlds" else value
                for key, value in cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)
            images, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_params, chunk_cameras, batched=True)
            rendered.extend(image.detach().cpu() for image in images)
    return rendered


class _ImageMetricEvaluator:
    def __init__(self, device):
        import lpips

        self.device = torch.device(device)
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device).eval()

    @staticmethod
    def _prepare(predictions, references):
        predictions = torch.stack(predictions).float()
        references = torch.stack(references).float()
        if references.shape[-1] == 4:
            predictions = predictions * references[..., 3:4]
            references = references[..., :3]
        return predictions.clamp(0, 1), references.clamp(0, 1)

    def evaluate(self, predictions, references, chunk_size):
        from utils.metrics import psnr, ssim

        count = len(predictions)
        if count != len(references) or count == 0:
            raise ValueError("Metric inputs must contain the same non-zero number of images")
        chunk_size = count if chunk_size <= 0 else min(chunk_size, count)
        values = {"psnr": [], "ssim": [], "lpips": []}
        with torch.no_grad():
            for start in range(0, count, chunk_size):
                pred, ref = self._prepare(predictions[start:start + chunk_size], references[start:start + chunk_size])
                pred, ref = pred.to(self.device), ref.to(self.device)
                values["psnr"].append(psnr(pred, ref).reshape(-1))
                values["ssim"].append(ssim(pred.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2),
                                             window_size=11, size_average=False).reshape(-1))
                values["lpips"].append(self.lpips(pred.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2),
                                                   normalize=True).reshape(-1))
        return {key: torch.cat(parts).mean().item() for key, parts in values.items()}


def _summarize_metric_trials(trials):
    summary = {}
    for comparison in ("vs_ground_truth", "vs_original_render"):
        summary[comparison] = {}
        for metric in ("psnr", "ssim", "lpips"):
            values = np.asarray([trial[comparison][metric] for trial in trials], dtype=np.float64)
            summary[comparison][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def _summarize_rotation_trials(trials):
    summary = _summarize_metric_trials(trials)
    summary["roundtrip_errors"] = {}
    for parameter in trials[0]["roundtrip_errors"]:
        summary["roundtrip_errors"][parameter] = {}
        for statistic in ("max_abs", "mean_abs"):
            values = np.asarray([trial["roundtrip_errors"][parameter][statistic] for trial in trials])
            summary["roundtrip_errors"][parameter][statistic] = {
                "mean": float(values.mean()), "std": float(values.std()), "max": float(values.max())
            }
    return summary


def _roundtrip_errors(reference, restored):
    errors = {}
    for key, expected in reference.items():
        if not torch.is_tensor(expected) or key not in restored:
            continue
        actual = restored[key].to(device=expected.device, dtype=expected.dtype)
        if key == "quats":
            expected = torch.nn.functional.normalize(expected, dim=-1)
            actual = torch.nn.functional.normalize(actual, dim=-1)
            actual = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0, -actual, actual)
        difference = (actual - expected).abs()
        errors[key] = {"max_abs": difference.max().item(), "mean_abs": difference.mean().item()}
    return errors


def _uint8_rgb(image):
    image = image[..., :3].detach().cpu().float().clamp(0, 1)
    return (image.numpy() * 255).round().astype(np.uint8)


def _save_preview(path, columns, view_count):
    import cv2

    rows = []
    for view_index in range(min(view_count, *(len(images) for images in columns.values()))):
        tiles = []
        for label, images in columns.items():
            image = _uint8_rgb(images[view_index])
            label_bar = np.zeros((28, image.shape[1], 3), dtype=np.uint8)
            cv2.putText(label_bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate((label_bar, image), axis=0))
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.concatenate(rows, axis=0)[..., ::-1])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def _load_experiment(args):
    import gin

    from dataset.GS_SR import SplatFactoSRDataset

    gin.parse_config_files_and_bindings(args.gin_file, args.gin_param, skip_unknown=True)
    with gin.unlock_config():
        gin.bind_parameter(f"{args.dataset_scope}/SplatFactoSRDataset.load_src_gs", True)
        gin.bind_parameter(f"{args.dataset_scope}/SplatFactoSRDataset.load_tgt_gs", True)
        gin.bind_parameter(f"{args.dataset_scope}/SplatFactoSRDataset.load_tgt_images", True)
        gin.bind_parameter(f"{args.dataset_scope}/SplatFactoSRDataset.split_across_gpus", False)
    dataset = SplatFactoSRDataset.from_gin_scope(args.dataset_scope)
    scene_index = dataset.scene_index(args.scene_name) if args.scene_name else 0
    scene = dataset.load_scene(scene_index, sample_views=False) if args.mode == "configured" else dataset.load_scene(scene_index)
    gs_resolution = dataset.src_resolution if args.gs_resolution == "source" else dataset.tgt_resolution
    target_entry = scene["data"][dataset.tgt_resolution]
    return dataset, scene, scene["data"][gs_resolution]["gs_params"], target_entry["images"], target_entry["cameras"]


def run_sweep(args):
    from utils import gpu_utils

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset, scene, source_gs, target_images, target_cameras = _load_experiment(args)
    if not target_images:
        raise ValueError("The selected scene has no target images")
    source_gs = gpu_utils.move_to_device(source_gs, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pivot = torch.tensor(args.pivot, dtype=source_gs["means"].dtype, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = _ImageMetricEvaluator(device)
    original_render_images = _render_views(source_gs, target_cameras, args.chunk_size, device)
    original_vs_ground_truth = evaluator.evaluate(original_render_images, target_images, args.chunk_size)
    report = {
        "metadata": {
            "scene_name": scene["scene_name"], "scene_idx": scene["scene_idx"],
            "dataset_scope": args.dataset_scope, "gs_resolution": args.gs_resolution,
            "source_resolution": dataset.src_resolution, "target_resolution": dataset.tgt_resolution,
            "jitter_levels": list(args.jitter_levels), "trials": args.trials, "seed": args.seed,
            "pivot": list(args.pivot), "gin_files": list(args.gin_file), "gin_params": list(args.gin_param),
            "metric_nonfinite_encoding": "string",
        },
        "original_unaugmented_render": {"vs_ground_truth": original_vs_ground_truth},
        "jitter": {},
        "rotation_roundtrip": {},
    }
    if args.preview_views > 0:
        _save_preview(args.output_dir / "previews" / "original_unaugmented_render.png",
                      {"original render": original_render_images, "ground truth": target_images}, args.preview_views)

    for parameter in GAUSSIAN_PARAMETERS:
        if parameter not in source_gs:
            continue
        report["jitter"][parameter] = {}
        for level in args.jitter_levels:
            trials = []
            for trial_index in range(args.trials):
                augmented = jitter_gaussian_parameter(source_gs, parameter, level, generator=generator)
                augmented_images = _render_views(augmented, target_cameras, args.chunk_size, device)
                trial = {
                    "trial": trial_index,
                    "vs_ground_truth": evaluator.evaluate(augmented_images, target_images, args.chunk_size),
                    "vs_original_render": evaluator.evaluate(augmented_images, original_render_images, args.chunk_size),
                }
                trials.append(trial)
                if trial_index == 0 and args.preview_views > 0:
                    level_name = str(level).replace(".", "p")
                    _save_preview(args.output_dir / "previews" / f"jitter_{parameter}_{level_name}.png",
                                  {"original render": original_render_images, "augmented render": augmented_images,
                                   "ground truth": target_images}, args.preview_views)
            report["jitter"][parameter][str(level)] = {
                "trials": trials, "aggregate": _summarize_metric_trials(trials)
            }
            print(f"jitter parameter={parameter} level={level}: {report['jitter'][parameter][str(level)]['aggregate']}")

    rotation_trials = []
    for trial_index in range(args.trials):
        rotation = sample_uniform_rotation_quaternion(source_gs["means"].dtype, device, generator)
        rotated = rotate_gaussians(source_gs, rotation, pivot, rotate_sh=True)
        restored = rotate_gaussians(rotated, quaternion_inverse(rotation), pivot, rotate_sh=True)
        restored_images = _render_views(restored, target_cameras, args.chunk_size, device)
        trial = {
            "trial": trial_index, "rotation_wxyz": rotation.detach().cpu().tolist(),
            "roundtrip_errors": _roundtrip_errors(source_gs, restored),
            "vs_ground_truth": evaluator.evaluate(restored_images, target_images, args.chunk_size),
            "vs_original_render": evaluator.evaluate(restored_images, original_render_images, args.chunk_size),
        }
        rotation_trials.append(trial)
        if trial_index == 0 and args.preview_views > 0:
            _save_preview(args.output_dir / "previews" / "rotation_roundtrip.png",
                          {"original render": original_render_images, "restored render": restored_images,
                           "ground truth": target_images}, args.preview_views)
    report["rotation_roundtrip"] = {
        "trials": rotation_trials, "aggregate": _summarize_rotation_trials(rotation_trials)
    }

    output_path = args.output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)
        handle.write("\n")
    print(f"Wrote augmentation report to {output_path}")
    return report


def augment_trial(prepared, cameras, args, generator):
    """Start one independent trial, rotating before jitter as in training."""
    rotation = prepared["means"].new_tensor([1., 0., 0., 0.])
    rotated, rotated_cameras = prepared, dict(cameras)
    if args.random_rotate:
        sampler = sample_uniform_z_rotation_quaternion if args.rotation_mode == "gravity_consistent" else sample_uniform_rotation_quaternion
        rotation = sampler(prepared["means"].dtype, prepared["means"].device, generator)
        rotated = rotate_gaussians(prepared, rotation, args.rotation_pivot)
        rotated_cameras["camera_to_worlds"] = rotate_camera_to_worlds(cameras["camera_to_worlds"], rotation, args.rotation_pivot)
    augmented, levels = rotated, dict.fromkeys(GAUSSIAN_PARAMETERS, 0.)
    if args.random_jitter:
        augmented, levels = jitter_gaussian_parameters(rotated, args.jitter_max_levels, generator)
    return augmented, rotated, rotated_cameras, rotation, levels


def render_difference(predictions, references, args):
    """Measure unmasked floating-point RGB errors, with per-view thresholds."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("Render comparisons require matching non-empty view lists")
    views = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        error = prediction[..., :3].float() - reference[..., :3].float()
        mse = error.square().mean().item()
        mae, maximum = error.abs().mean().item(), error.abs().max().item()
        views.append({"view": index, "mae": mae, "max_abs": maximum,
                      "psnr": -10 * math.log10(mse) if mse > 0 else (float("inf") if mse == 0 else float("nan")),
                      "passed": math.isfinite(mae) and math.isfinite(maximum) and mae <= args.mae_threshold and maximum <= args.max_error_threshold})
    return {"mae": sum(v["mae"] for v in views) / len(views),
            "max_abs": max(v["max_abs"] for v in views),
            "psnr": sum(v["psnr"] for v in views) / len(views),
            "passed": all(v["passed"] for v in views), "views": views}


def quaternion_norm_summary(gs):
    norms = gs["quats"].float().norm(dim=-1)
    return {"min": norms.min().item(), "mean": norms.mean().item(), "max": norms.max().item()}


def save_gaussian_artifact(path, gs, cameras, metadata):
    """Save scene-frame Gaussian tensors and their matching cameras, not network weights."""
    from utils.gs_utils import export_ply_forviewer

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"artifact_type": "gaussian_scene", "gs_params": {k: v.detach().cpu() for k, v in gs.items()},
               "cameras": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in cameras.items()},
               "metadata": metadata}
    torch.save(payload, path)
    export_ply_forviewer(payload["gs_params"], path.with_suffix(".ply"))


def run_configured(args):
    from utils import gpu_utils

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dataset, scene, original, target_images, cameras = _load_experiment(args)
    original = gpu_utils.move_to_device(original, device)
    cameras = gpu_utils.move_to_device(cameras, device)
    prepared = normalize_gaussian_quaternions(original)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = _ImageMetricEvaluator(device)
    original_images = _render_views(original, cameras, args.chunk_size, device)
    baseline = _render_views(prepared, cameras, args.chunk_size, device)
    settings = {key: getattr(args, key) for key in ("random_jitter", "random_rotate", "rotation_mode", "rotation_pivot",
                "jitter_max_levels", "seed", "trials", "mae_threshold", "max_error_threshold", "gs_resolution", "dataset_scope")}
    metadata = {"scene_name": scene["scene_name"], "scene_idx": scene["scene_idx"], **settings,
                "source_resolution": dataset.src_resolution, "target_resolution": dataset.tgt_resolution,
                "coordinate_frame": scene.get("coordinate_frame"), "coordinate_frame_version": scene.get("coordinate_frame_version"),
                "coordinate_resolution": scene.get("coordinate_resolution"),
                "gin_files": list(args.gin_file), "gin_params": list(args.gin_param),
                "image_names": scene["data"][dataset.tgt_resolution].get("images_name", []),
                "quaternion_representation": "unit_unstandardized", "error_preview_gain": 100}
    normalization = render_difference(baseline, original_images, args)
    report = {"metadata": metadata, "normalization": normalization, "passed": normalization["passed"],
              "original_norms": quaternion_norm_summary(original), "normalized_norms": quaternion_norm_summary(prepared),
              "original_vs_ground_truth": evaluator.evaluate(original_images, target_images, args.chunk_size),
              "normalized_vs_ground_truth": evaluator.evaluate(baseline, target_images, args.chunk_size), "trials": []}
    save_gaussian_artifact(args.output_dir / "normalized.pt", prepared, cameras, metadata)
    if args.preview_views:
        _save_preview(args.output_dir / "previews/normalization.png",
                      {"original": original_images, "normalized": baseline,
                       "error x100": [(a - b).abs() * 100 for a, b in zip(baseline, original_images)]}, args.preview_views)
    for index in range(args.trials):
        augmented, rotated, trial_cameras, rotation, levels = augment_trial(prepared, cameras, args, generator)
        rotated_images = _render_views(rotated, trial_cameras, args.chunk_size, device) if args.random_rotate else baseline
        trial = {"trial": index, "rotation_wxyz": rotation.cpu().tolist(), "jitter_levels": levels,
                 "quaternion_norms": quaternion_norm_summary(augmented)}
        if args.random_rotate:
            inverse = quaternion_inverse(rotation)
            restored = rotate_gaussians(rotated, inverse, args.rotation_pivot)
            restored_cameras = {**trial_cameras, "camera_to_worlds": rotate_camera_to_worlds(trial_cameras["camera_to_worlds"], inverse, args.rotation_pivot)}
            restored_images = _render_views(restored, restored_cameras, args.chunk_size, device)
            trial["rotation"] = render_difference(rotated_images, baseline, args)
            trial["restoration"] = render_difference(restored_images, baseline, args)
            trial["roundtrip_errors"] = _roundtrip_errors(prepared, restored)
            report["passed"] &= trial["rotation"]["passed"] and trial["restoration"]["passed"]
        images = _render_views(augmented, trial_cameras, args.chunk_size, device) if args.random_jitter else rotated_images
        # Jitter changes images intentionally; only zero jitter is an invariance check.
        jitter_difference = render_difference(images, rotated_images, args)
        trial["jitter_difference"] = {k: v for k, v in jitter_difference.items() if k != "passed"}
        trial["jitter_difference"]["views"] = [{k: v for k, v in view.items() if k != "passed"} for view in jitter_difference["views"]]
        if not any(levels.values()):
            trial["zero_jitter_passed"] = jitter_difference["passed"]
            report["passed"] &= jitter_difference["passed"]
        trial["vs_ground_truth"] = evaluator.evaluate(images, target_images, args.chunk_size)
        artifact = f"trial_{index:03d}.pt"
        trial["checkpoint"] = artifact
        save_gaussian_artifact(args.output_dir / artifact, augmented, trial_cameras, {**metadata, **trial})
        if args.preview_views:
            _save_preview(args.output_dir / f"previews/trial_{index:03d}.png",
                          {"normalized": baseline, "rotated": rotated_images, "augmented": images,
                           "rotation error x100": [(a - b).abs() * 100 for a, b in zip(rotated_images, baseline)],
                           "jitter error x100": [(a - b).abs() * 100 for a, b in zip(images, rotated_images)]}, args.preview_views)
        report["trials"].append(trial)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(_json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output_dir / 'metrics.json'}; invariance passed={report['passed']}")
    return report


def run(args):
    return run_configured(args) if args.mode == "configured" else run_sweep(args)


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    if report.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

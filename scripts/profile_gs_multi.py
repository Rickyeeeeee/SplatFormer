#!/usr/bin/env python3
"""Profile CPU time and RAM used by ``SplatFactoMultiLevelDataset``.

Example:
    python scripts/profile_gs_multi.py \
        --gin-file configs/model/ptv3.gin \
        --gin-file configs/train/sr_2stage.gin \
        --scope train_dataset \
        --num-scenes 3

The script deliberately profiles only the CPU-side dataset path: checkpoint
loading/filtering, camera metadata loading, and image decoding.  It does not
move tensors to CUDA or run the model/rasterizer.
"""

import argparse
import cProfile
import gc
import json
import os
import pstats
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import gin
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.GS_multi import SplatFactoMultiLevelDataset  # noqa: E402
from models.feature_predictor import FeaturePredictor  # noqa: E402,F401


T = TypeVar("T")


# ``configs/train/sr_2stage.gin`` is shared with the training entry point.
# Register the two training bindings that GS_multi reads, while allowing the
# profiler to ignore optimizer/logging settings that it never uses.
@gin.configurable
def set_seed(seed: int) -> int:
    return seed


@gin.configurable
def training(pretrain_steps: int = 0, **unused: Any) -> int:
    del unused
    return pretrain_steps


def _parse_scene_indices(value: str) -> List[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scene indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("scene indices cannot be empty")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("scene indices must not contain duplicates")
    return indices


def _read_proc_status() -> Dict[str, Optional[int]]:
    """Return Linux process RSS/HWM/thread count without an extra dependency."""
    result: Dict[str, Optional[int]] = {"rss_kib": None, "hwm_kib": None, "threads": None}
    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                key, _, raw_value = line.partition(":")
                value = raw_value.strip().split()[0] if raw_value.strip() else ""
                if key == "VmRSS":
                    result["rss_kib"] = int(value)
                elif key == "VmHWM":
                    result["hwm_kib"] = int(value)
                elif key == "Threads":
                    result["threads"] = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return result


def _process_snapshot() -> Dict[str, Optional[int]]:
    snapshot = _read_proc_status()
    # Linux reports ru_maxrss in KiB. It is a process-wide high-water mark,
    # not an isolated peak for the current stage.
    snapshot["ru_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return snapshot


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _tensor_shape(value: Any) -> Optional[List[int]]:
    return list(value.shape) if torch.is_tensor(value) else None


def _factor_payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    gs_params = payload["gs_params"]
    images = payload["images"]
    cameras = payload["cameras"]
    return {
        "num_gaussians": int(gs_params["means"].shape[0]),
        "num_views": len(images),
        "image_shapes": sorted({tuple(image.shape) for image in images}),
        # These are tensor bytes referenced by the returned payload, rather
        # than a claim about newly allocated bytes (some tensors are shared).
        "referenced_gs_tensor_bytes": _tensor_bytes(gs_params),
        "referenced_image_tensor_bytes": _tensor_bytes(images),
        "referenced_camera_tensor_bytes": _tensor_bytes(cameras),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _profile_stage(name: str, operation: Callable[[], T]) -> Tuple[T, Dict[str, Any]]:
    before = _process_snapshot()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    value = operation()
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    after = _process_snapshot()
    return value, {
        "stage": name,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "average_cpu_cores": cpu_seconds / wall_seconds if wall_seconds > 0 else None,
        "memory_before": before,
        "memory_after": after,
        "rss_delta_kib": (
            after["rss_kib"] - before["rss_kib"]
            if after["rss_kib"] is not None and before["rss_kib"] is not None
            else None
        ),
    }


def _select_scene_indices(
    dataset: SplatFactoMultiLevelDataset,
    requested_indices: Optional[Sequence[int]],
    num_scenes: int,
) -> List[int]:
    if requested_indices is not None:
        indices = list(requested_indices)
    else:
        indices = list(range(min(num_scenes, len(dataset.folders))))
    invalid = [index for index in indices if index < 0 or index >= len(dataset.folders)]
    if invalid:
        raise ValueError(f"Scene indices are outside [0, {len(dataset.folders) - 1}]: {invalid}")
    if not indices:
        raise ValueError("No scenes selected for profiling")
    return indices


def _select_camera_ids(
    dataset: SplatFactoMultiLevelDataset,
    scene: Dict[str, Any],
    seed: int,
    max_views: Optional[int],
) -> np.ndarray:
    primary_entry = scene["factor_data"][dataset.primary_factor]
    total_views = len(primary_entry["meta"]["camera_to_worlds"])
    if dataset.train_or_test == "train":
        sample_count = total_views if dataset.image_per_scene is None else min(dataset.image_per_scene, total_views)
        camera_ids = np.random.default_rng(seed + scene["idx"]).permutation(total_views)[:sample_count]
    else:
        camera_ids = np.arange(total_views)
    if max_views is not None:
        camera_ids = camera_ids[:max_views]
    if len(camera_ids) == 0:
        raise ValueError("Selected zero views; choose --max-views greater than zero")
    return camera_ids


def _profile_scene(
    dataset: SplatFactoMultiLevelDataset,
    scene_idx: int,
    seed: int,
    max_views: Optional[int],
) -> Dict[str, Any]:
    scene, scene_stage = _profile_stage("load_scene", lambda: dataset.load_scene(scene_idx))
    scene_stage.update({"scene_idx": scene_idx, "scene_name": scene["scene_name"]})

    camera_ids = _select_camera_ids(dataset, scene, seed, max_views)
    background, background_stage = _profile_stage("build_background", dataset.build_background)
    background_stage.update({"scene_idx": scene_idx, "scene_name": scene["scene_name"]})

    factor_stages = []
    payloads = {}
    for factor in dataset.factors:
        payload, stage = _profile_stage(
            f"prepare_factor_payload[{factor}]",
            lambda factor=factor: dataset._prepare_factor_payload(
                scene["factor_data"][factor], background=background, cam_ids=camera_ids
            ),
        )
        stage.update({"scene_idx": scene_idx, "scene_name": scene["scene_name"], "factor": factor})
        stage["payload"] = _factor_payload_summary(payload)
        payloads[factor] = payload
        factor_stages.append(stage)

    # Match the normal iterator's lifetime: all factor payloads coexist until
    # it yields the batch, then become reclaimable before the next scene.
    result = {
        "scene_idx": scene_idx,
        "scene_name": scene["scene_name"],
        "camera_ids": [int(camera_id) for camera_id in camera_ids],
        "stages": [scene_stage, background_stage, *factor_stages],
    }
    del payloads, scene
    gc.collect()
    return result


def _write_cprofile_report(profiler: cProfile.Profile, output_dir: Path, profile_lines: int) -> None:
    profiler.dump_stats(str(output_dir / "profile.prof"))
    with open(output_dir / "profile_cumulative.txt", "w") as report_file:
        pstats.Stats(profiler, stream=report_file).strip_dirs().sort_stats("cumulative").print_stats(profile_lines)


def _print_summary(scene_results: Iterable[Dict[str, Any]], output_dir: Path) -> None:
    stages = [stage for scene in scene_results for stage in scene["stages"]]
    print("\nGS_multi profile summary (sorted by wall time):")
    for stage in sorted(stages, key=lambda item: item["wall_seconds"], reverse=True):
        print(
            f"  scene={stage['scene_idx']:>4} {stage['stage']:<30} "
            f"wall={stage['wall_seconds']:.3f}s cpu={stage['process_cpu_seconds']:.3f}s "
            f"cores={stage['average_cpu_cores']:.2f} rss_delta={stage['rss_delta_kib']} KiB"
        )
    print(f"\nWrote JSON metrics and cProfile reports to {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gin-file", "--gin_file", action="append", dest="gin_files", required=True)
    parser.add_argument("--gin-param", "--gin_param", action="append", dest="gin_params", default=[])
    parser.add_argument("--scope", choices=("train_dataset", "test_dataset"), default="train_dataset")
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--scene-indices", type=_parse_scene_indices)
    parser.add_argument("--max-views", type=int, help="Cap decoded views per scene for a quick profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-lines", type=int, default=80)
    parser.add_argument("--no-cprofile", action="store_true", help="Skip cProfile and only collect stage timings")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.num_scenes <= 0:
        raise ValueError("--num-scenes must be positive")
    if args.max_views is not None and args.max_views <= 0:
        raise ValueError("--max-views must be positive")
    if args.profile_lines <= 0:
        raise ValueError("--profile-lines must be positive")

    output_dir = args.output_dir or REPO_ROOT / "profiles" / (
        "gs_multi_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gin.clear_config()
    gin.parse_config_files_and_bindings(args.gin_files, args.gin_params, skip_unknown=True)
    with gin.config_scope(args.scope):
        dataset, dataset_stage = _profile_stage("build_dataset", SplatFactoMultiLevelDataset)

    scene_indices = _select_scene_indices(dataset, args.scene_indices, args.num_scenes)
    profiler = cProfile.Profile()
    if not args.no_cprofile:
        profiler.enable()
    try:
        scene_results = [
            _profile_scene(dataset, scene_idx, seed=args.seed, max_views=args.max_views)
            for scene_idx in scene_indices
        ]
    finally:
        if not args.no_cprofile:
            profiler.disable()

    dataset_stage["scene_count"] = len(dataset.folders)
    dataset_stage["scope"] = args.scope
    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "gin_files": args.gin_files,
            "gin_params": args.gin_params,
            "scope": args.scope,
            "selected_scene_indices": scene_indices,
            "max_views": args.max_views,
            "seed": args.seed,
            "dataset_factors": list(dataset.factors),
            "dataset_train_or_test": dataset.train_or_test,
        },
        "dataset_stage": dataset_stage,
        "scenes": scene_results,
    }
    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(_json_ready(result), metrics_file, indent=2)
    with open(output_dir / "config.gin", "w") as config_file:
        config_file.write(gin.operative_config_str())
    if not args.no_cprofile:
        _write_cprofile_report(profiler, output_dir, args.profile_lines)
    _print_summary(scene_results, output_dir)


if __name__ == "__main__":
    main()

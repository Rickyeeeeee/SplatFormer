#!/usr/bin/env python3
"""Compute processed Gaussian-splat statistics separately for each downsample factor.

The dataset loader is intentionally reused here. Consequently, the values in
the report are the same values seen by the SR training code: invalid splats are
filtered, configured outlier/truncation rules are applied, means are normalized
per scene, and scales are adjusted for that normalization.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gin
import torch
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_predictor import FeaturePredictor  # Registers Gin bindings used below.


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_GIN_FILES = (
    str(SCRIPT_ROOT / "configs/model/ptv3.gin"),
    str(SCRIPT_ROOT / "configs/train/sr_2stage.gin"),
)

ALL_GS_PARAMETERS = (
    "means",
    "scales",
    "opacities",
    "quats",
    "features_dc",
    "features_rest",
)


@gin.configurable
def training(pretrain_steps: int = 0, **unused_kwargs: Any) -> None:
    """Register the binding queried by ``SplatFactoMultiLevelDataset``."""

    del pretrain_steps, unused_kwargs


def _json_value(tensor: torch.Tensor, shape: Tuple[int, ...]) -> Any:
    return tensor.detach().cpu().reshape(shape if shape else ()).tolist()


class ChannelwiseStreamingStats:
    """Population statistics for tensors shaped ``[N, *channel_shape]``."""

    def __init__(self) -> None:
        self.channel_shape: Optional[Tuple[int, ...]] = None
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.minimum: Optional[torch.Tensor] = None
        self.maximum: Optional[torch.Tensor] = None

    def can_accept(self, value: torch.Tensor) -> Optional[str]:
        if not torch.is_tensor(value):
            return "value is not a tensor"
        if value.ndim < 1:
            return "tensor must have a leading Gaussian dimension"
        if value.shape[0] == 0:
            return "tensor has zero Gaussian rows"
        if not torch.is_floating_point(value):
            return "tensor is not floating point"
        if not torch.isfinite(value).all().item():
            return "tensor contains non-finite values"

        channel_shape = tuple(value.shape[1:])
        if self.channel_shape is not None and channel_shape != self.channel_shape:
            return f"channel shape {channel_shape} does not match {self.channel_shape}"
        return None

    def update(self, value: torch.Tensor) -> None:
        problem = self.can_accept(value)
        if problem is not None:
            raise ValueError(problem)

        channel_shape = tuple(value.shape[1:])
        flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(value.shape[0], -1)
        batch_count = int(flat.shape[0])
        batch_mean = flat.mean(dim=0)
        batch_m2 = ((flat - batch_mean) ** 2).sum(dim=0)
        batch_min = flat.min(dim=0).values
        batch_max = flat.max(dim=0).values

        if self.count == 0:
            self.channel_shape = channel_shape
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
            return

        assert self.mean is not None and self.m2 is not None
        assert self.minimum is not None and self.maximum is not None
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total_count)
        self.m2 = self.m2 + batch_m2 + delta.square() * (self.count * batch_count / total_count)
        self.minimum = torch.minimum(self.minimum, batch_min)
        self.maximum = torch.maximum(self.maximum, batch_max)
        self.count = total_count

    def as_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "gaussian_count": 0,
                "channel_shape": None,
                "mean": None,
                "min": None,
                "max": None,
                "variance": None,
                "std": None,
            }

        assert self.channel_shape is not None
        assert self.mean is not None and self.m2 is not None
        assert self.minimum is not None and self.maximum is not None
        variance = self.m2 / self.count
        return {
            "gaussian_count": self.count,
            "channel_shape": list(self.channel_shape),
            "mean": _json_value(self.mean, self.channel_shape),
            "min": _json_value(self.minimum, self.channel_shape),
            "max": _json_value(self.maximum, self.channel_shape),
            "variance": _json_value(variance, self.channel_shape),
            "std": _json_value(torch.sqrt(variance), self.channel_shape),
        }


def _scene_skip_record(
    scene_idx: Optional[int], scene_name: Optional[str], exception: BaseException
) -> Dict[str, Any]:
    return {
        "scene_idx": scene_idx,
        "scene_name": scene_name,
        "skip_reason": "scene_load_or_statistics_error",
        "exception_type": type(exception).__name__,
        "exception_reason": str(exception),
    }


def _dataset_skip_records(dataset: SplatFactoMultiLevelDataset) -> List[Dict[str, Any]]:
    records = []
    for skipped in getattr(dataset, "skipped_scenes", []):
        records.append(
            {
                "scene_idx": skipped.get("scene_idx"),
                "scene_name": skipped.get("scene_name"),
                "skip_reason": skipped.get("skip_reason", skipped.get("reason")),
                "exception_type": skipped.get("exception_type"),
                "exception_reason": skipped.get("exception_reason"),
            }
        )
    return records


def _validate_loaded_scene(
    scene: Mapping[str, Any],
    factors: Sequence[int],
    accumulators: Mapping[int, Mapping[str, ChannelwiseStreamingStats]],
) -> None:
    factor_data = scene.get("factor_data")
    if not isinstance(factor_data, Mapping):
        raise ValueError("load_scene returned no factor_data mapping")

    for factor in factors:
        if factor not in factor_data:
            raise ValueError(f"load_scene did not return df-{factor}")
        gs_params = factor_data[factor].get("gs_params")
        if not isinstance(gs_params, Mapping):
            raise ValueError(f"df-{factor} has no gs_params mapping")

        gaussian_count: Optional[int] = None
        for parameter in ALL_GS_PARAMETERS:
            if parameter not in gs_params:
                raise KeyError(f"df-{factor} is missing Gaussian parameter '{parameter}'")
            value = gs_params[parameter]
            if not torch.is_tensor(value) or value.ndim < 1:
                raise ValueError(f"df-{factor} parameter '{parameter}' is not a Gaussian tensor")
            if gaussian_count is None:
                gaussian_count = int(value.shape[0])
            elif value.shape[0] != gaussian_count:
                raise ValueError(f"df-{factor} Gaussian dimensions disagree across parameters")
            problem = accumulators[factor][parameter].can_accept(value)
            if problem is not None:
                raise ValueError(f"df-{factor} parameter '{parameter}': {problem}")


def analyze_dataset(dataset: SplatFactoMultiLevelDataset, show_progress: bool = True) -> Dict[str, Any]:
    """Load each complete scene once and aggregate its processed GS tensors."""

    factors = list(dataset.factors)
    accumulators = {
        factor: {parameter: ChannelwiseStreamingStats() for parameter in ALL_GS_PARAMETERS}
        for factor in factors
    }
    factor_scene_counts = {factor: 0 for factor in factors}
    skipped_scenes = _dataset_skip_records(dataset)
    successful_scene_count = 0

    iterator: Iterable[int] = range(len(dataset.folders))
    if show_progress:
        iterator = tqdm(iterator, desc="Loading scenes")

    for scene_idx in iterator:
        scene_info = dataset.folders[scene_idx]
        scene_name = scene_info.get("scene_name")
        try:
            scene = dataset.load_scene(scene_idx)
            _validate_loaded_scene(scene, factors, accumulators)
        except Exception as exc:
            skipped_scenes.append(_scene_skip_record(scene_idx, scene_name, exc))
            print(f"Skipping scene_idx={scene_idx} scene_name={scene_name}: {type(exc).__name__}: {exc}")
            continue

        for factor in factors:
            gs_params = scene["factor_data"][factor]["gs_params"]
            for parameter in ALL_GS_PARAMETERS:
                accumulators[factor][parameter].update(gs_params[parameter])
            factor_scene_counts[factor] += 1
        successful_scene_count += 1

    factor_report = {}
    for factor in factors:
        parameter_report = {
            parameter: accumulators[factor][parameter].as_dict()
            for parameter in ALL_GS_PARAMETERS
        }
        factor_report[f"df-{factor}"] = {
            "scene_count": factor_scene_counts[factor],
            "gaussian_count": parameter_report["means"]["gaussian_count"],
            "parameters": parameter_report,
        }

    return {
        "candidate_scene_count": len(dataset.folders),
        "successful_scene_count": successful_scene_count,
        "skipped_scene_count": len(skipped_scenes),
        "skipped_scenes": skipped_scenes,
        "factors": factor_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute processed Gaussian parameter statistics for each dataset df factor."
    )
    parser.add_argument(
        "--gin_file",
        action="append",
        default=None,
        help="Gin configuration file. Repeat to override the default two-stage Gin stack.",
    )
    parser.add_argument(
        "--gin_param",
        action="append",
        default=[],
        help="Gin parameter binding. Repeat for multiple overrides.",
    )
    parser.add_argument(
        "--dataset_scope",
        choices=("train_dataset", "test_dataset"),
        default="train_dataset",
        help="Gin scope defining the SplatFactoMultiLevelDataset to analyze.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("gs_statistics.json"),
        help="Destination JSON report (default: gs_statistics.json).",
    )
    args = parser.parse_args()
    if args.gin_file is None:
        args.gin_file = list(DEFAULT_GIN_FILES)
    return args


def build_dataset(
    gin_files: Sequence[str], gin_params: Sequence[str], dataset_scope: str
) -> SplatFactoMultiLevelDataset:
    # Training configs commonly contain optimizer/model bindings irrelevant to
    # dataset construction. Keep recognized dataset/model/training bindings and
    # ignore the rest, because this tool never instantiates a predictor.
    gin.parse_config_files_and_bindings(gin_files, gin_params, skip_unknown=True)
    with gin.unlock_config():
        gin.bind_parameter("FeaturePredictor.input_features", list(ALL_GS_PARAMETERS))

    with gin.config_scope(dataset_scope):
        return SplatFactoMultiLevelDataset()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.gin_file, args.gin_param, args.dataset_scope)
    report = analyze_dataset(dataset)
    report.update(
        {
            "dataset_scope": args.dataset_scope,
            "gin_files": list(args.gin_file),
            "gin_params": list(args.gin_param),
            "statistics_definition": {
                "values": "processed tensors returned by SplatFactoMultiLevelDataset.load_scene",
                "variance": "population variance (ddof=0)",
                "aggregation": "Gaussian-weighted across all successfully loaded scenes",
            },
        }
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print(
        f"Processed {report['successful_scene_count']}/{report['candidate_scene_count']} candidate scenes; "
        f"recorded {report['skipped_scene_count']} skipped scenes."
    )
    for factor, summary in report["factors"].items():
        print(f"{factor}: scenes={summary['scene_count']} gaussians={summary['gaussian_count']}")
    print(f"Wrote JSON report: {args.output_json}")


if __name__ == "__main__":
    main()

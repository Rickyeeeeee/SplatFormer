"""Filesystem index and metric normalization for the evaluation viewer."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2


METRICS: Tuple[str, ...] = ("psnr", "ssim", "lpips")
SORT_FIELDS = {
    "scene_idx",
    "scene_name",
    "image_id",
    "image_name",
    "psnr",
    "ssim",
    "lpips",
    "gain_psnr",
    "gain_ssim",
    "gain_lpips",
}


class EvaluationDataError(RuntimeError):
    """Raised when an evaluation artifact cannot be interpreted safely."""


@dataclass(frozen=True)
class IterationRecord:
    name: str
    path: Path
    scenes: Mapping[str, Mapping[str, Any]]
    metric_sets: Mapping[str, Optional[Mapping[str, float]]]
    modified_ns: int


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"Could not read {path}: {exc}") from exc


def _finite_number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metrics(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    result: Dict[str, float] = {}
    for metric in METRICS:
        number = _finite_number(value.get(metric))
        if number is None:
            return None
        result[metric] = number
    return result


def metric_gain(prediction: Optional[Mapping[str, float]], baseline: Optional[Mapping[str, float]]) -> Optional[Dict[str, float]]:
    """Return gains with a consistent positive-is-better convention."""

    if prediction is None or baseline is None:
        return None
    return {
        "psnr": float(prediction["psnr"]) - float(baseline["psnr"]),
        "ssim": float(prediction["ssim"]) - float(baseline["ssim"]),
        "lpips": float(baseline["lpips"]) - float(prediction["lpips"]),
    }


def _mean_metrics(rows: Iterable[Optional[Mapping[str, float]]]) -> Optional[Dict[str, float]]:
    valid = [row for row in rows if row is not None]
    if not valid:
        return None
    return {metric: sum(float(row[metric]) for row in valid) / len(valid) for metric in METRICS}


def _sort_value(row: Mapping[str, Any], field: str) -> Tuple[int, Any]:
    if field.startswith("gain_"):
        value = (row.get("gain") or {}).get(field[5:])
    elif field in METRICS:
        value = (row.get("metrics") or {}).get(field)
    else:
        value = row.get(field)
    return (1 if value is None else 0, value if value is not None else 0)


def _sorted_page(
    rows: Sequence[Dict[str, Any]],
    sort: str,
    order: str,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    if sort not in SORT_FIELDS:
        raise ValueError(f"Unsupported sort field: {sort}")
    if order not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")
    page = max(1, page)
    page_size = max(1, min(500, page_size))
    present = [row for row in rows if _sort_value(row, sort)[0] == 0]
    missing = [row for row in rows if _sort_value(row, sort)[0] == 1]
    present.sort(key=lambda row: _sort_value(row, sort)[1], reverse=order == "desc")
    ordered = present + missing
    start = (page - 1) * page_size
    return {
        "items": ordered[start : start + page_size],
        "total": len(ordered),
        "page": page,
        "page_size": page_size,
        "pages": max(1, math.ceil(len(ordered) / page_size)),
        "sort": sort,
        "order": order,
    }


class EvaluationIndex:
    """Thread-safe, read-only view of an evaluation output directory."""

    def __init__(self, eval_dir: Path, reference_iteration: str = "00000000") -> None:
        self.eval_dir = Path(eval_dir).expanduser().resolve()
        self.reference_iteration = reference_iteration
        self._iterations: Dict[str, IterationRecord] = {}
        self._warnings: List[str] = []
        self._lock = threading.RLock()
        self.refresh()

    def refresh(self) -> Dict[str, Any]:
        if not self.eval_dir.is_dir():
            raise EvaluationDataError(f"Evaluation directory does not exist: {self.eval_dir}")

        iterations: Dict[str, IterationRecord] = {}
        warnings: List[str] = []
        for path in sorted(self.eval_dir.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.isdigit():
                continue
            summary_path = path / "scene_average_metrics.json"
            if not summary_path.is_file():
                warnings.append(f"{path.name}: waiting for scene_average_metrics.json")
                continue
            try:
                payload = _read_json(summary_path)
                if not isinstance(payload, list):
                    raise EvaluationDataError(f"{summary_path} must contain a list")
                scenes: Dict[str, Mapping[str, Any]] = {}
                for row in payload:
                    if not isinstance(row, Mapping) or not isinstance(row.get("scene_name"), str):
                        continue
                    scenes[row["scene_name"]] = row
                metric_sets = {
                    "prediction": self._aggregate_metric_file(path / "metrics.json"),
                    "input": self._aggregate_metric_file(path / "metrics_input.json"),
                    "gt_low_res": self._aggregate_metric_file(path / "metrics_gt_low_res.json"),
                    "gt_high_res": self._aggregate_metric_file(path / "metrics_gt_high_res.json"),
                }
                iterations[path.name] = IterationRecord(
                    name=path.name,
                    path=path.resolve(),
                    scenes=scenes,
                    metric_sets=metric_sets,
                    modified_ns=summary_path.stat().st_mtime_ns,
                )
            except EvaluationDataError as exc:
                warnings.append(f"{path.name}: {exc}")

        with self._lock:
            self._iterations = iterations
            self._warnings = warnings
        return self.status()

    @staticmethod
    def _aggregate_metric_file(path: Path) -> Optional[Dict[str, float]]:
        if not path.is_file():
            return None
        return _metrics(_read_json(path))

    @property
    def default_iteration(self) -> Optional[str]:
        with self._lock:
            if not self._iterations:
                return None
            return max(self._iterations, key=lambda value: int(value))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "eval_dir": str(self.eval_dir),
                "reference_iteration": self.reference_iteration,
                "reference_available": self.reference_iteration in self._iterations,
                "default_iteration": self.default_iteration,
                "iteration_count": len(self._iterations),
                "warnings": list(self._warnings),
            }

    def _iteration(self, name: str) -> IterationRecord:
        with self._lock:
            record = self._iterations.get(name)
        if record is None:
            raise KeyError(f"Unknown or incomplete iteration: {name}")
        return record

    def iterations(self) -> Dict[str, Any]:
        default = self.default_iteration
        reference = self._iterations.get(self.reference_iteration)
        items = []
        with self._lock:
            records = sorted(self._iterations.values(), key=lambda row: int(row.name), reverse=True)
        for record in records:
            values = [_metrics(row.get("output_gs")) for row in record.scenes.values()]
            prediction = record.metric_sets.get("prediction") or _mean_metrics(values)
            metric_sets = dict(record.metric_sets)
            metric_sets["prediction"] = prediction
            if reference:
                for name in ("input", "gt_low_res", "gt_high_res"):
                    metric_sets[name] = metric_sets.get(name) or reference.metric_sets.get(name)
            items.append(
                {
                    "name": record.name,
                    "step": int(record.name),
                    "scene_count": len(record.scenes),
                    "metrics": prediction,
                    "metric_sets": metric_sets,
                    "is_default": record.name == default,
                    "is_reference": record.name == self.reference_iteration,
                    "modified_ns": record.modified_ns,
                }
            )
        return {
            **self.status(),
            "reference_scene_count": len(reference.scenes) if reference else 0,
            "items": items,
        }

    def scenes(
        self,
        iteration: str,
        sort: str = "gain_psnr",
        order: str = "desc",
        search: str = "",
        page: int = 1,
        page_size: int = 200,
    ) -> Dict[str, Any]:
        current = self._iteration(iteration)
        reference = self._iterations.get(self.reference_iteration)
        needle = search.strip().lower()
        rows: List[Dict[str, Any]] = []
        for scene_name, scene in current.scenes.items():
            if needle and needle not in scene_name.lower():
                continue
            prediction = _metrics(scene.get("output_gs"))
            reference_scene = reference.scenes.get(scene_name) if reference else None
            baseline = _metrics(reference_scene.get("input_gs")) if reference_scene else None
            gt_low_res = _metrics(reference_scene.get("gt_low_res_gs")) if reference_scene else None
            gt_high_res = _metrics(reference_scene.get("gt_high_res_gs")) if reference_scene else None
            rows.append(
                {
                    "scene_idx": scene.get("scene_idx"),
                    "scene_name": scene_name,
                    "metrics": prediction,
                    "input_metrics": baseline,
                    "gt_low_res_metrics": gt_low_res,
                    "gt_high_res_metrics": gt_high_res,
                    "gain": metric_gain(prediction, baseline),
                    "reference_available": bool(reference_scene),
                    "view_metrics_available": (current.path / scene_name / "metrics.json").is_file(),
                }
            )
        result = _sorted_page(rows, sort, order, page, page_size)
        result.update({"iteration": iteration, "search": search})
        return result

    @staticmethod
    def _view_records(path: Path) -> List[Mapping[str, Any]]:
        if not path.is_file():
            return []
        payload = _read_json(path)
        return [row for row in payload if isinstance(row, Mapping)] if isinstance(payload, list) else []

    def views(
        self,
        iteration: str,
        scene_name: str,
        sort: str = "gain_psnr",
        order: str = "desc",
        search: str = "",
        page: int = 1,
        page_size: int = 500,
    ) -> Dict[str, Any]:
        current = self._iteration(iteration)
        if scene_name not in current.scenes:
            raise KeyError(f"Unknown scene {scene_name!r} in iteration {iteration}")
        current_rows = self._view_records(current.path / scene_name / "metrics.json")
        reference = self._iterations.get(self.reference_iteration)
        baseline_rows = (
            self._view_records(reference.path / scene_name / "metrics_input.json")
            if reference and scene_name in reference.scenes
            else []
        )
        gt_low_res_rows = (
            self._view_records(reference.path / scene_name / "metrics_gt_low_res.json")
            if reference and scene_name in reference.scenes
            else []
        )
        gt_high_res_rows = (
            self._view_records(reference.path / scene_name / "metrics_gt_high_res.json")
            if reference and scene_name in reference.scenes
            else []
        )
        baseline_by_name = {str(row.get("image_name")): row for row in baseline_rows}
        gt_low_res_by_name = {str(row.get("image_name")): row for row in gt_low_res_rows}
        gt_high_res_by_name = {str(row.get("image_name")): row for row in gt_high_res_rows}
        needle = search.strip().lower()
        rows: List[Dict[str, Any]] = []
        for row in current_rows:
            image_name = str(row.get("image_name", ""))
            if not image_name or (needle and needle not in image_name.lower()):
                continue
            prediction = _metrics(row)
            baseline = _metrics(baseline_by_name.get(image_name))
            pred_path = current.path / scene_name / "pred" / image_name
            compare_path = (
                reference.path / scene_name / "compare" / image_name if reference else None
            )
            rows.append(
                {
                    "image_id": row.get("image_id"),
                    "image_name": image_name,
                    "metrics": prediction,
                    "input_metrics": baseline,
                    "gt_low_res_metrics": _metrics(gt_low_res_by_name.get(image_name)),
                    "gt_high_res_metrics": _metrics(gt_high_res_by_name.get(image_name)),
                    "gain": metric_gain(prediction, baseline),
                    "images": {
                        "prediction": pred_path.is_file(),
                        "input": bool(compare_path and compare_path.is_file()),
                        "gt": bool(compare_path and compare_path.is_file()),
                    },
                }
            )
        result = _sorted_page(rows, sort, order, page, page_size)
        result.update({"iteration": iteration, "scene_name": scene_name, "search": search})
        return result

    def prediction_path(self, iteration: str, scene_name: str, image_name: str) -> Path:
        record = self._iteration(iteration)
        if scene_name not in record.scenes:
            raise KeyError(f"Unknown scene: {scene_name}")
        known = {str(row.get("image_name")) for row in self._view_records(record.path / scene_name / "metrics.json")}
        if image_name not in known or Path(image_name).name != image_name:
            raise KeyError(f"Unknown view: {image_name}")
        path = (record.path / scene_name / "pred" / image_name).resolve()
        if self.eval_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def reference_image(self, scene_name: str, image_name: str, kind: str) -> bytes:
        if kind not in {"gt", "input"}:
            raise ValueError("Reference image kind must be 'gt' or 'input'")
        reference = self._iteration(self.reference_iteration)
        if scene_name not in reference.scenes or Path(image_name).name != image_name:
            raise KeyError(f"Unknown reference scene or view: {scene_name}/{image_name}")
        compare_path = (reference.path / scene_name / "compare" / image_name).resolve()
        if self.eval_dir not in compare_path.parents or not compare_path.is_file():
            raise FileNotFoundError(compare_path)
        image = cv2.imread(str(compare_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim not in {2, 3}:
            raise EvaluationDataError(f"Could not decode comparison strip: {compare_path}")
        height, width = image.shape[:2]
        if width % 3 != 0 or height <= 0:
            raise EvaluationDataError(
                f"Comparison strip must contain three equal-width panels: {compare_path} has {width}x{height}"
            )
        panel_width = width // 3
        panel_index = 0 if kind == "gt" else 1
        panel = image[:, panel_index * panel_width : (panel_index + 1) * panel_width]
        encoded, buffer = cv2.imencode(".png", panel)
        if not encoded:
            raise EvaluationDataError(f"Could not encode {kind} panel from {compare_path}")
        return buffer.tobytes()

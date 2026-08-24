#!/usr/bin/env python3
"""Summarize SR isolation runs and apply the experiment decision tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any


RUN_ORDER = ("E0", "E1", "E2", "E3", "E4")
ATTRIBUTES = (
    "means",
    "scales",
    "opacities",
    "quats",
    "features_dc",
    "features_rest",
)
METRICS = ("psnr", "ssim", "lpips")
DEFAULT_CONTROL_PSNR = 27.0791


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--runs-subdir", default="runs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--control-reference-psnr", type=float, default=DEFAULT_CONTROL_PSNR)
    parser.add_argument("--control-tolerance-db", type=float, default=0.25)
    parser.add_argument("--target-gap-db", type=float, default=1.25)
    parser.add_argument("--minimum-gain-recovery", type=float, default=0.85)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mean_metric(value: Any) -> float | None:
    values: list[float] = []

    def collect(item: Any) -> None:
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            values.append(float(item))
        elif isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)

    collect(value)
    return fmean(values) if values else None


def _final_metrics(path: Path) -> dict[str, float] | None:
    report = _safe_json(path)
    if not isinstance(report, dict):
        return None
    result = {name: _mean_metric(_extract_metric(report, name)) for name in METRICS}
    result = {key: value for key, value in result.items() if value is not None}
    return result or None


def _extract_metric(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        if name in value:
            return value[name]
        return {key: _extract_metric(child, name) for key, child in value.items()}
    return None


def _matching_metrics(path: Path) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    report = _safe_json(path)
    if not isinstance(report, dict):
        return None, None
    entries = list(report.items())
    input_entry = next(
        (value for key, value in entries if "input_low_res" in key),
        entries[0][1] if entries else None,
    )
    fitted_entry = next(
        (value for key, value in entries if "fitted_target" in key),
        entries[1][1] if len(entries) > 1 else None,
    )

    def normalize(entry: Any) -> dict[str, float] | None:
        if not isinstance(entry, dict):
            return None
        output = {
            key: float(entry[key])
            for key in METRICS
            if key in entry and isinstance(entry[key], (int, float))
        }
        return output or None

    return normalize(input_entry), normalize(fitted_entry)


def _metrics_from_log(text: str, label: str) -> dict[str, float] | None:
    if label == "final":
        matches = re.findall(r"Final eval:\s+([^\n]+)", text)
    elif label == "input":
        matches = re.findall(r"Fitted alignment .*input_low_res.*?:\s+([^\n]+)", text)
    else:
        matches = re.findall(r"Fitted alignment .*fitted_target.*?:\s+([^\n]+)", text)
    if not matches:
        return None
    values = {
        key: float(value)
        for key, value in re.findall(
            r"(psnr|ssim|lpips):\s*([-+0-9.eE]+)", matches[-1]
        )
    }
    return values or None


def _last_losses(text: str) -> tuple[int | None, dict[str, float]]:
    lines = re.findall(r"(?:^|\n).*?step=(\d+)\s+total=([^\n]+)", text)
    if not lines:
        return None, {}
    step, tail = lines[-1]
    values = {
        key: float(value)
        for key, value in re.findall(r"([a-z_]+)=([-+0-9.eE]+)", "total=" + tail)
    }
    losses = {"total": values["total"]} if "total" in values else {}
    for attribute in ATTRIBUTES:
        key = f"{attribute}_loss"
        weighted_key = f"{attribute}_weighted"
        if key in values:
            losses[key] = values[key]
        if weighted_key in values:
            losses[weighted_key] = values[weighted_key]
    return int(step), losses


def _ply_vertices(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            for _ in range(200):
                line = stream.readline()
                if not line:
                    break
                decoded = line.decode("ascii", errors="replace").strip()
                match = re.fullmatch(r"element vertex (\d+)", decoded)
                if match:
                    return int(match.group(1))
                if decoded == "end_header":
                    break
    except OSError:
        return None
    return None


def _parse_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    wanted = {
        "dataset_root",
        "matching_total_steps",
        "test_scene_list",
        "total_steps",
        "matching_fit.image_per_step",
        "matching_fit.total_steps",
        "set_seed.seed",
        "training.total_steps",
        "training.eval_interval",
        "training.save_interval",
        "FeaturePredictor.output_features",
        "FeaturePredictor.output_features_type",
        "FeaturePredictor.max_scale_normalized",
        "feature_mse_loss.loss_weights",
    }
    resolved: dict[str, Any] = {}
    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if stripped.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    for line in logical_lines:
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in wanted:
            resolved[key] = value
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "selected_bindings": resolved,
    }


def _hashes(run_dir: Path) -> dict[str, str]:
    candidates = (
        run_dir / "config.gin",
        run_dir / "checkpoints" / "model_last.pth",
        run_dir / "matching_init" / "00_input_low_res_gs.ply",
        run_dir / "matching_init" / "01_fitted_target_gs.ply",
        run_dir / "ablation_backend.json",
        run_dir / "run_manifest.json",
    )
    return {
        str(path.relative_to(run_dir)): _sha256(path)
        for path in candidates
        if path.is_file()
    }


def summarize_run(
    name: str,
    run_dir: Path,
    *,
    target_gap_db: float,
    minimum_gain_recovery: float,
) -> dict[str, Any]:
    log_path = run_dir / "overfit.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    input_metrics, fitted_metrics = _matching_metrics(
        run_dir / "matching_init" / "render_metrics.json"
    )
    input_metrics = input_metrics or _metrics_from_log(text, "input")
    fitted_metrics = fitted_metrics or _metrics_from_log(text, "fitted")
    final_metrics = _final_metrics(run_dir / "eval_final" / "metrics.json")
    final_metrics = final_metrics or _metrics_from_log(text, "final")
    last_step, losses = _last_losses(text)

    input_psnr = input_metrics.get("psnr") if input_metrics else None
    fitted_psnr = fitted_metrics.get("psnr") if fitted_metrics else None
    final_psnr = final_metrics.get("psnr") if final_metrics else None
    available_gain = (
        fitted_psnr - input_psnr
        if fitted_psnr is not None and input_psnr is not None
        else None
    )
    recovered_gain = (
        final_psnr - input_psnr
        if final_psnr is not None and input_psnr is not None
        else None
    )
    gain_recovery = (
        recovered_gain / available_gain
        if available_gain is not None
        and recovered_gain is not None
        and available_gain > 0
        else None
    )
    target_gap = (
        fitted_psnr - final_psnr
        if fitted_psnr is not None and final_psnr is not None
        else None
    )
    complete_metrics = all(
        value is not None for value in (input_psnr, fitted_psnr, final_psnr)
    )
    converged = bool(
        complete_metrics
        and target_gap is not None
        and target_gap <= target_gap_db
        and gain_recovery is not None
        and gain_recovery >= minimum_gain_recovery
    )
    return {
        "name": name,
        "path": str(run_dir.resolve()),
        "exists": run_dir.is_dir(),
        "complete": final_metrics is not None,
        "last_training_step": last_step,
        "metrics": {
            "input": input_metrics,
            "fitted_target": fitted_metrics,
            "final": final_metrics,
        },
        "available_psnr_gain_db": available_gain,
        "recovered_psnr_gain_db": recovered_gain,
        "gain_recovery_fraction": gain_recovery,
        "gap_to_fitted_target_db": target_gap,
        "converged": converged,
        "thresholds": {
            "maximum_target_gap_db": target_gap_db,
            "minimum_gain_recovery_fraction": minimum_gain_recovery,
        },
        "final_attribute_losses": losses,
        "splat_counts": {
            "input": _ply_vertices(
                run_dir / "matching_init" / "00_input_low_res_gs.ply"
            ),
            "fitted_target": _ply_vertices(
                run_dir / "matching_init" / "01_fitted_target_gs.ply"
            ),
        },
        "hashes": _hashes(run_dir),
        "resolved_config": _parse_config(run_dir / "config.gin"),
        "run_manifest": _safe_json(run_dir / "run_manifest.json"),
        "backend": _safe_json(run_dir / "ablation_backend.json"),
    }


def decide(
    runs: dict[str, dict[str, Any]],
    *,
    control_reference_psnr: float,
    control_tolerance_db: float,
) -> dict[str, Any]:
    completed = {name: bool(run["complete"]) for name, run in runs.items()}
    e0_final = ((runs["E0"].get("metrics") or {}).get("final") or {}).get("psnr")
    e0_difference = (
        abs(e0_final - control_reference_psnr) if e0_final is not None else None
    )
    control_reproduced = bool(
        e0_difference is not None and e0_difference <= control_tolerance_db
    )
    decision = "incomplete"
    explanation = "E0-E3 have not all completed."
    next_run: str | None = next(
        (name for name in ("E0", "E1", "E2", "E3") if not completed[name]), None
    )

    if completed["E0"] and not control_reproduced:
        decision = "invalid_control_environment"
        explanation = (
            f"E0 is {e0_difference:.3f} dB from the control reference, exceeding "
            f"the {control_tolerance_db:.3f} dB tolerance. Stop the matrix."
        )
        next_run = None
    elif completed["E0"] and control_reproduced:
        if not completed["E1"]:
            next_run = "E1"
        elif not runs["E1"]["converged"]:
            decision = "native_128_rendering"
            explanation = "E0 reproduces but E1 fails; native 128 rendering caused the regression."
            next_run = None
        elif not completed["E2"]:
            next_run = "E2"
        elif not runs["E2"]["converged"]:
            decision = "custom_gsplat_producer"
            explanation = "E1 passes but E2 fails; custom gsplat Gaussian training caused the regression."
            next_run = None
        elif not completed["E3"]:
            next_run = "E3"
        elif runs["E3"]["converged"]:
            decision = "no_regression_reproduced"
            explanation = "E0-E3 pass; the reported regression was not reproduced by this matrix."
            next_run = None
        elif not completed["E4"]:
            decision = "e4_required"
            explanation = "E2 passes and E3 fails; run E4 to isolate loader versus current training code."
            next_run = "E4"
        elif runs["E4"]["converged"]:
            decision = "new_dataset_loader"
            explanation = "E2 passes, E3 fails, and E4 passes; the new dataset loader caused the regression."
            next_run = None
        else:
            decision = "current_non_loader_training_code"
            explanation = "E2 passes while E3 and E4 fail; current non-loader training code caused it."
            next_run = None

    parity_warnings: list[str] = []
    e2_fitted = ((runs["E2"].get("metrics") or {}).get("fitted_target") or {}).get("psnr")
    e3_fitted = ((runs["E3"].get("metrics") or {}).get("fitted_target") or {}).get("psnr")
    if e2_fitted is not None and e3_fitted is not None:
        difference = abs(e2_fitted - e3_fitted)
        if difference > 0.2:
            parity_warnings.append(
                f"E2/E3 fitted-target PSNR differs by {difference:.3f} dB (>0.2 dB)."
            )
    return {
        "classification": decision,
        "explanation": explanation,
        "next_run": next_run,
        "control_reference_psnr": control_reference_psnr,
        "control_difference_db": e0_difference,
        "control_tolerance_db": control_tolerance_db,
        "control_reproduced": control_reproduced,
        "parity_warnings": parity_warnings,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        f"# SR refactor isolation: `{report['scene_name']}`",
        "",
        f"Decision: **{decision['classification']}** — {decision['explanation']}",
        "",
        "| Run | Input PSNR | Fitted PSNR | Final PSNR | Gap | Gain recovered | Converged | Splats (in/fit) |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for name in RUN_ORDER:
        run = report["runs"][name]
        metrics = run["metrics"]
        input_psnr = (metrics["input"] or {}).get("psnr")
        fitted_psnr = (metrics["fitted_target"] or {}).get("psnr")
        final_psnr = (metrics["final"] or {}).get("psnr")
        fraction = run["gain_recovery_fraction"]
        fraction_text = "—" if fraction is None else f"{100.0 * fraction:.1f}%"
        counts = run["splat_counts"]
        lines.append(
            "| {name} | {input} | {fitted} | {final} | {gap} | {fraction} | {conv} | {inc}/{fitc} |".format(
                name=name,
                input=_fmt(input_psnr),
                fitted=_fmt(fitted_psnr),
                final=_fmt(final_psnr),
                gap=_fmt(run["gap_to_fitted_target_db"]),
                fraction=fraction_text,
                conv=_fmt(run["converged"]),
                inc=_fmt(counts["input"], 0),
                fitc=_fmt(counts["fitted_target"], 0),
            )
        )
    lines.extend(["", "## Final attribute losses", ""])
    lines.append("| Run | " + " | ".join(ATTRIBUTES) + " |")
    lines.append("|---|" + "---:|" * len(ATTRIBUTES))
    for name in RUN_ORDER:
        losses = report["runs"][name]["final_attribute_losses"]
        lines.append(
            f"| {name} | "
            + " | ".join(_fmt(losses.get(f"{attribute}_loss"), 6) for attribute in ATTRIBUTES)
            + " |"
        )
    warnings = decision["parity_warnings"]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"A run converges when the final result is within {report['thresholds']['maximum_target_gap_db']} dB of the fitted target and recovers at least {100 * report['thresholds']['minimum_gain_recovery_fraction']:.0f}% of available PSNR gain.",
            f"E0 must reproduce {decision['control_reference_psnr']:.4f} dB within ±{decision['control_tolerance_db']:.2f} dB.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = args.experiment_root / args.runs_subdir / args.scene_name
    runs = {
        name: summarize_run(
            name,
            runs_root / name,
            target_gap_db=args.target_gap_db,
            minimum_gain_recovery=args.minimum_gain_recovery,
        )
        for name in RUN_ORDER
    }
    return {
        "version": 1,
        "scene_name": args.scene_name,
        "experiment_root": str(args.experiment_root.resolve()),
        "runs_root": str(runs_root.resolve()),
        "thresholds": {
            "maximum_target_gap_db": args.target_gap_db,
            "minimum_gain_recovery_fraction": args.minimum_gain_recovery,
        },
        "runs": runs,
        "decision": decide(
            runs,
            control_reference_psnr=args.control_reference_psnr,
            control_tolerance_db=args.control_tolerance_db,
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.experiment_root / "summaries" / args.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

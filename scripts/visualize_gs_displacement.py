#!/usr/bin/env python3
"""Inspect per-Gaussian prediction residuals in a Viser 3D scene.

The two-stage trainer writes row-aligned PLYs, so Gaussian ``i`` in the input
and prediction files describes the same Gaussian.  This tool keeps that
correspondence: it draws input/predicted positions, displacement vectors, and
a displacement-magnitude heatmap, then writes residual arrays and summaries.

Examples:
  # Use one viewer folder produced by train-sr-2stage.py.
  python scripts/visualize_gs_displacement.py \
      --viewer_dir outputs/.../viewer/<scene>

  # Compare the input with the stage-one (means-only) output instead.
  python scripts/visualize_gs_displacement.py \
      --viewer_dir outputs/.../viewer/<scene> --prediction stage1

  # Compare any two row-aligned Gaussian PLYs.
  python scripts/visualize_gs_displacement.py \
      --input_ply input.ply --predicted_ply prediction.ply
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814
DEFAULT_INPUT_NAME = "00_input_gs.ply"
PREDICTION_NAMES = {
    "stage1": "01_stage1_output_gs.ply",
    "stage2": "02_stage2_output_gs.ply",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--viewer_dir",
        type=Path,
        help="Trainer viewer directory containing point_cloud/00_input_gs.ply.",
    )
    source.add_argument("--input_ply", type=Path, help="Row-aligned input Gaussian PLY.")
    parser.add_argument("--predicted_ply", type=Path, help="Row-aligned predicted Gaussian PLY.")
    parser.add_argument(
        "--prediction",
        choices=tuple(PREDICTION_NAMES),
        default="stage2",
        help="Prediction chosen from --viewer_dir (default: stage2).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Residual output directory (default: <prediction parent>/displacement_residuals).",
    )
    parser.add_argument("--sample_count", type=int, default=25_000, help="Maximum points displayed; 0 displays all (25,000 is the responsive default).")
    parser.add_argument("--arrow_count", type=int, default=500, help="Maximum mean-displacement arrows displayed; 0 displays all sampled points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hist_bins", type=int, default=80)
    parser.add_argument("--write_per_point_csv", action="store_true", help="Also write a potentially large per-point CSV.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--point_size", type=float, default=0.003)
    parser.add_argument("--dry_run", action="store_true", help="Write residual files without starting Viser.")
    return parser.parse_args()


def _sorted_prefixed_fields(names, prefix):
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name[len(prefix) :]))


def _normalize_quaternions(quats):
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    return quats / np.clip(norm, 1e-8, None)


def _read_gs_ply(path):
    path = Path(path)
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    required = {"x", "y", "z"}
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")

    count = len(vertex.data)
    def stacked(fields, default):
        return (
            np.stack([vertex[field] for field in fields], axis=1).astype(np.float32)
            if set(fields).issubset(names)
            else np.broadcast_to(default, (count, len(default))).astype(np.float32).copy()
        )

    features_dc = stacked(["f_dc_0", "f_dc_1", "f_dc_2"], np.zeros(3, dtype=np.float32))
    rest_fields = _sorted_prefixed_fields(names, "f_rest_")
    features_rest = (
        np.stack([vertex[field] for field in rest_fields], axis=1).astype(np.float32)
        if rest_fields
        else np.zeros((count, 0), dtype=np.float32)
    )
    scales = stacked(["scale_0", "scale_1", "scale_2"], np.full(3, -5.0, dtype=np.float32))
    quats = stacked(["rot_0", "rot_1", "rot_2", "rot_3"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    opacities = np.asarray(vertex["opacity"], dtype=np.float32) if "opacity" in names else np.zeros(count, dtype=np.float32)
    return {
        "means": np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32),
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacities": opacities.reshape(-1),
        "scales": scales,
        "quats": _normalize_quaternions(quats),
    }


def _sigmoid(values):
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def _summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _aligned_quaternion_delta(input_quats, predicted_quats):
    # q and -q represent the same rotation.  Put each prediction in the input
    # quaternion's hemisphere before reporting a component residual.
    predicted_quats = predicted_quats.copy()
    dots = np.sum(input_quats * predicted_quats, axis=1)
    predicted_quats[dots < 0.0] *= -1.0
    delta = predicted_quats - input_quats
    angles = np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))
    return delta, angles


def compute_residuals(input_gs, predicted_gs):
    count = input_gs["means"].shape[0]
    if predicted_gs["means"].shape[0] != count:
        raise ValueError(
            "The PLYs have different Gaussian counts; this viewer requires row-aligned files "
            f"({count} input vs {predicted_gs['means'].shape[0]} prediction)."
        )
    if input_gs["features_rest"].shape[1] != predicted_gs["features_rest"].shape[1]:
        raise ValueError("The PLYs have different f_rest channel counts and cannot be compared row-wise.")

    means_delta = predicted_gs["means"] - input_gs["means"]
    dc_delta = predicted_gs["features_dc"] - input_gs["features_dc"]
    rest_delta = predicted_gs["features_rest"] - input_gs["features_rest"]
    opacity_logit_delta = predicted_gs["opacities"] - input_gs["opacities"]
    opacity_probability_delta = _sigmoid(predicted_gs["opacities"]) - _sigmoid(input_gs["opacities"])
    scale_log_delta = predicted_gs["scales"] - input_gs["scales"]
    scale_linear_delta = np.exp(np.clip(predicted_gs["scales"], -30.0, 20.0)) - np.exp(np.clip(input_gs["scales"], -30.0, 20.0))
    quat_delta, quat_angle_deg = _aligned_quaternion_delta(input_gs["quats"], predicted_gs["quats"])
    return {
        "means_delta": means_delta,
        "means_l2": np.linalg.norm(means_delta, axis=1),
        "features_dc_delta": dc_delta,
        "features_dc_l2": np.linalg.norm(dc_delta, axis=1),
        "features_rest_delta": rest_delta,
        "features_rest_l2": np.linalg.norm(rest_delta, axis=1),
        "opacity_logit_delta": opacity_logit_delta,
        "opacity_probability_delta": opacity_probability_delta,
        "scale_log_delta": scale_log_delta,
        "scale_log_l2": np.linalg.norm(scale_log_delta, axis=1),
        "scale_linear_delta": scale_linear_delta,
        "quaternion_delta": quat_delta,
        "quaternion_l2": np.linalg.norm(quat_delta, axis=1),
        "quaternion_geodesic_deg": quat_angle_deg,
    }


def _summary_rows(residuals):
    rows = []
    for name, values in residuals.items():
        values = np.asarray(values)
        if values.ndim == 1:
            rows.append({"residual": name, "component": "", **_summary(values), "mean_abs": float(np.mean(np.abs(values)))})
        else:
            rows.append({"residual": name, "component": "l2", **_summary(np.linalg.norm(values, axis=1)), "mean_abs": float(np.mean(np.abs(values)))})
            for component in range(values.shape[1]):
                component_values = values[:, component]
                rows.append({"residual": name, "component": str(component), **_summary(component_values), "mean_abs": float(np.mean(np.abs(component_values)))})
    return rows


def write_residual_outputs(output_dir, input_path, predicted_path, input_gs, predicted_gs, residuals, bins, write_per_point_csv):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "per_gaussian_residuals.npz", **residuals)
    rows = _summary_rows(residuals)
    fields = ["residual", "component", "count", "min", "mean", "mean_abs", "std", "median", "p95", "p99", "max"]
    with (output_dir / "residual_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "residual_summary.json").open("w") as handle:
        json.dump({
            "input_ply": str(input_path),
            "predicted_ply": str(predicted_path),
            "gaussian_count": int(input_gs["means"].shape[0]),
            "residuals": rows,
        }, handle, indent=2)

    if write_per_point_csv:
        with (output_dir / "per_gaussian_residuals.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "index", "mean_dx", "mean_dy", "mean_dz", "mean_l2", "dc_l2", "rest_l2",
                "opacity_logit_delta", "opacity_probability_delta", "scale_log_l2", "quat_geodesic_deg",
            ])
            for index in range(input_gs["means"].shape[0]):
                writer.writerow([
                    index, *residuals["means_delta"][index], residuals["means_l2"][index],
                    residuals["features_dc_l2"][index], residuals["features_rest_l2"][index],
                    residuals["opacity_logit_delta"][index], residuals["opacity_probability_delta"][index],
                    residuals["scale_log_l2"][index], residuals["quaternion_geodesic_deg"][index],
                ])

    _write_histograms(output_dir, residuals, bins)


def _write_histograms(output_dir, residuals, bins):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping residual histogram image: {exc}", flush=True)
        return

    plotted = [
        ("mean displacement L2", residuals["means_l2"]),
        ("DC feature L2", residuals["features_dc_l2"]),
        ("SH-rest feature L2", residuals["features_rest_l2"]),
        ("opacity probability delta", residuals["opacity_probability_delta"]),
        ("log-scale L2", residuals["scale_log_l2"]),
        ("rotation geodesic (degrees)", residuals["quaternion_geodesic_deg"]),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (name, values) in zip(axes.flat, plotted):
        axis.hist(values, bins=max(1, int(bins)), color="#4C78A8", alpha=0.9)
        axis.axvline(np.mean(values), color="#F58518", linewidth=1.5, label=f"mean={np.mean(values):.4g}")
        axis.set_title(name)
        axis.set_xlabel("signed residual" if "opacity" in name else "magnitude")
        axis.set_ylabel("Gaussians")
        axis.legend(fontsize=8)
    figure.suptitle("Predicted minus input Gaussian residuals")
    figure.tight_layout()
    figure.savefig(output_dir / "residual_histograms.png", dpi=160)
    plt.close(figure)


def _rgb_from_dc(features_dc):
    return np.round(np.clip(features_dc * C0 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)


def _turbo_like_colors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    low, high = np.percentile(values, [1.0, 99.0])
    t = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    stops = np.array([[49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233], [254, 224, 144], [253, 174, 97], [215, 48, 39]], dtype=np.float32)
    position = t * (len(stops) - 1)
    left = np.floor(position).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    mix = (position - left)[:, None]
    return np.round(stops[left] * (1.0 - mix) + stops[right] * mix).astype(np.uint8)


def _resolve_paths(args):
    if args.viewer_dir is not None:
        point_cloud_dir = args.viewer_dir / "point_cloud"
        input_path = point_cloud_dir / DEFAULT_INPUT_NAME
        predicted_path = point_cloud_dir / PREDICTION_NAMES[args.prediction]
        if args.predicted_ply is not None:
            raise ValueError("--predicted_ply can only be used with --input_ply, not --viewer_dir")
    else:
        if args.predicted_ply is None:
            raise ValueError("--predicted_ply is required with --input_ply")
        input_path, predicted_path = args.input_ply, args.predicted_ply
    for path in (input_path, predicted_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing PLY: {path}")
    return input_path, predicted_path


def _server_component(server, modern_name, legacy_name):
    component = getattr(server, modern_name, None)
    return component if component is not None else server


def _server_method(component, server, modern_name, legacy_name):
    method = getattr(component, modern_name, None)
    if method is None:
        method = getattr(server, legacy_name, None)
    if method is None:
        raise AttributeError(f"Installed viser exposes neither {modern_name} nor {legacy_name}")
    return method


def launch_viewer(input_gs, predicted_gs, residuals, display_permutation, display_count, arrow_count, args):
    try:
        import viser
    except ImportError as exc:
        raise ImportError("Viser is required for the interactive viewer. Install requirements.txt, or use --dry_run.") from exc

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = _server_component(server, "scene", "")
    gui = _server_component(server, "gui", "")
    display_indices = display_permutation[:display_count]
    input_points = input_gs["means"][display_indices]
    predicted_points = predicted_gs["means"][display_indices]
    residual_options = {
        "means": residuals["means_l2"],
        "scales (log)": residuals["scale_log_l2"],
        "rotations (degrees)": residuals["quaternion_geodesic_deg"],
        "features_rest": residuals["features_rest_l2"],
    }
    bbox = np.concatenate([input_points, predicted_points], axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0)))
    bbox_diagonal = max(bbox_diagonal, 1e-4)
    point_size_min = max(bbox_diagonal * 1e-5, 1e-6)
    point_size_max = max(bbox_diagonal * 0.05, args.point_size * 10.0)
    point_size = float(np.clip(args.point_size, point_size_min, point_size_max))
    selected_residual = "means"
    interpolation_t = 0.0
    color_by_residual = False
    cloud_handle = None

    def set_display_count(new_count):
        nonlocal display_count, display_indices, input_points, predicted_points
        display_count = min(int(new_count), len(display_permutation))
        display_indices = display_permutation[:display_count]
        input_points = input_gs["means"][display_indices]
        predicted_points = predicted_gs["means"][display_indices]

    def refresh_point_cloud():
        """Recreate one cloud: older Viser handles cannot update points/size."""
        nonlocal cloud_handle
        if cloud_handle is not None:
            cloud_handle.remove()
        points = input_points * (1.0 - interpolation_t) + predicted_points * interpolation_t
        colors = (
            _turbo_like_colors(residual_options[selected_residual][display_indices])
            if color_by_residual
            else _rgb_from_dc(input_gs["features_dc"][display_indices])
        )
        cloud_handle = scene.add_point_cloud(
            "/interpolated_gs", points=points, colors=colors,
            point_size=point_size, point_shape="circle",
        )

    refresh_point_cloud()

    arrow_handles = []
    line_segments = getattr(scene, "add_line_segments", None)
    spline_arrows = line_segments is None
    warned_spline_arrow_cap = False

    def add_arrows(scale):
        nonlocal arrow_handles, warned_spline_arrow_cap
        for handle in arrow_handles:
            handle.remove()
        arrow_handles = []
        arrow_indices = display_indices[:min(arrow_count, len(display_indices))]
        arrow_input_points = input_gs["means"][arrow_indices]
        arrow_delta = residuals["means_delta"][arrow_indices]
        arrow_colors = _turbo_like_colors(residuals["means_l2"][arrow_indices])
        if spline_arrows and len(arrow_input_points) > 500:
            # Older Viser releases have no batched line-segment primitive. One
            # spline per arrow is expensive, so cap that compatibility path.
            arrow_input_points = arrow_input_points[:500]
            arrow_delta = arrow_delta[:500]
            arrow_colors = arrow_colors[:500]
            if not warned_spline_arrow_cap:
                print("Installed Viser has no batched line segments; showing at most 500 displacement arrows.", flush=True)
                warned_spline_arrow_cap = True
        if len(arrow_input_points) == 0:
            return
        # Keep these arrows in the input frame. Rebuilding spline arrows on
        # every interpolation event is the main source of viewer lag on older
        # Viser versions, and the vector is still the exact input-to-prediction
        # mean displacement.
        arrow_start = arrow_input_points
        arrow_end = arrow_start + arrow_delta * float(scale)
        arrow_points = np.stack([arrow_start, arrow_end], axis=1)
        if line_segments is not None:
            arrow_handles.append(line_segments(
                "/displacement_vectors", points=arrow_points, colors=arrow_colors, line_width=1.5,
            ))
            return
        for index, (points, color) in enumerate(zip(arrow_points, arrow_colors)):
            arrow_handles.append(scene.add_spline_catmull_rom(
                f"/displacement_vectors/{index:05d}", positions=points, line_width=1.5,
                color=tuple(int(value) for value in color), segments=1,
            ))

    add_checkbox = _server_method(gui, server, "add_checkbox", "add_gui_checkbox")
    add_slider = _server_method(gui, server, "add_slider", "add_gui_slider")
    add_dropdown = _server_method(gui, server, "add_dropdown", "add_gui_dropdown")
    color_residual = add_checkbox("Color cloud by selected residual", initial_value=False)
    show_arrows = add_checkbox("Show input→predicted mean directions", initial_value=False)
    residual_selector = add_dropdown("Residual heatmap parameter", options=tuple(residual_options), initial_value=selected_residual)

    total_count = len(display_permutation)
    display_values = sorted({
        min(total_count, value)
        for value in (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, display_count, total_count)
    })
    display_options = {
        f"{value:,}" + (" (all)" if value == total_count else ""): value
        for value in display_values
    }
    arrow_values = sorted({0, 50, 100, 250, 500, 1_000, 2_500, 5_000, arrow_count})
    arrow_options = {
        ("0 (hide)" if value == 0 else f"{value:,}"): value
        for value in arrow_values
    }
    display_count_selector = add_dropdown(
        "Displayed Gaussians", options=tuple(display_options),
        initial_value=next(label for label, value in display_options.items() if value == display_count),
    )
    arrow_count_selector = add_dropdown(
        "Mean direction count", options=tuple(arrow_options),
        initial_value=next(label for label, value in arrow_options.items() if value == arrow_count),
    )
    interpolation_slider = add_slider(
        "Position interpolation (input → predicted)", min=0.0, max=1.0, step=0.05, initial_value=interpolation_t,
    )
    point_size_slider = add_slider(
        "Point size", min=point_size_min, max=point_size_max,
        step=(point_size_max - point_size_min) / 100.0, initial_value=point_size,
    )
    arrow_scale = add_slider("Vector scale", min=0.0, max=20.0, step=0.1, initial_value=1.0)

    @color_residual.on_update
    def _(_event):
        nonlocal color_by_residual
        color_by_residual = color_residual.value
        refresh_point_cloud()

    @show_arrows.on_update
    def _(_event):
        if show_arrows.value and selected_residual == "means":
            add_arrows(arrow_scale.value)
        for handle in arrow_handles:
            handle.visible = show_arrows.value and selected_residual == "means"

    @residual_selector.on_update
    def _(_event):
        nonlocal selected_residual
        selected_residual = residual_selector.value
        refresh_point_cloud()
        if show_arrows.value and selected_residual == "means" and not arrow_handles:
            add_arrows(arrow_scale.value)
        for handle in arrow_handles:
            handle.visible = show_arrows.value and selected_residual == "means"

    @display_count_selector.on_update
    def _(_event):
        set_display_count(display_options[display_count_selector.value])
        refresh_point_cloud()
        if show_arrows.value and selected_residual == "means":
            add_arrows(arrow_scale.value)

    @arrow_count_selector.on_update
    def _(_event):
        nonlocal arrow_count
        arrow_count = arrow_options[arrow_count_selector.value]
        if show_arrows.value and selected_residual == "means":
            add_arrows(arrow_scale.value)

    @interpolation_slider.on_update
    def _(_event):
        nonlocal interpolation_t
        interpolation_t = float(interpolation_slider.value)
        refresh_point_cloud()

    @point_size_slider.on_update
    def _(_event):
        nonlocal point_size
        point_size = float(point_size_slider.value)
        refresh_point_cloud()

    @arrow_scale.on_update
    def _(_event):
        add_arrows(arrow_scale.value)
        for handle in arrow_handles:
            handle.visible = show_arrows.value and selected_residual == "means"

    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    print("Blue-to-red colors run from the 1st to 99th percentile of the selected residual.", flush=True)
    print("Position interpolation 0 is input GS; 1 is predicted GS. Only one point cloud is displayed.", flush=True)
    print("For more responsive interpolation, lower --sample_count (for example 10,000).", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping Viser server.", flush=True)


def main():
    args = parse_args()
    if args.sample_count < 0 or args.arrow_count < 0:
        raise ValueError("--sample_count and --arrow_count must be non-negative")
    input_path, predicted_path = _resolve_paths(args)
    input_gs, predicted_gs = _read_gs_ply(input_path), _read_gs_ply(predicted_path)
    residuals = compute_residuals(input_gs, predicted_gs)
    output_dir = args.output_dir or predicted_path.parent / "displacement_residuals"
    write_residual_outputs(output_dir, input_path, predicted_path, input_gs, predicted_gs, residuals, args.hist_bins, args.write_per_point_csv)

    count = input_gs["means"].shape[0]
    print(f"Compared {count:,} row-aligned Gaussians", flush=True)
    print(f"Mean position displacement: {residuals['means_l2'].mean():.6g}", flush=True)
    print(f"P95 position displacement: {np.percentile(residuals['means_l2'], 95):.6g}", flush=True)
    print(f"Residuals written to: {output_dir}", flush=True)
    if args.dry_run:
        return

    rng = np.random.default_rng(args.seed)
    display_count = count if args.sample_count == 0 else min(count, args.sample_count)
    display_permutation = rng.permutation(count)
    arrow_count = display_count if args.arrow_count == 0 else min(display_count, args.arrow_count)
    launch_viewer(input_gs, predicted_gs, residuals, display_permutation, display_count, arrow_count, args)


if __name__ == "__main__":
    main()

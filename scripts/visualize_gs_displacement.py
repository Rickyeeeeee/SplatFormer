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

  # Use an AnchorSplat inference_multi scene directory.
  python scripts/visualize_gs_displacement.py \
      --inference_multi_scene_dir outputs/.../<scene>

  # Compare any two row-aligned Gaussian PLYs.
  python scripts/visualize_gs_displacement.py \
      --input_ply input.ply --output_ply output.ply
"""

import argparse
import csv
import json
import math
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814
DEFAULT_INPUT_NAME = "00_input_gs.ply"
ANCHORSPLAT_OUTPUT_NAME = "01_output_gs.ply"
PREDICTION_NAMES = {
    "stage1": "01_stage1_output_gs.ply",
    "stage2": "02_stage2_output_gs.ply",
    "output": ANCHORSPLAT_OUTPUT_NAME,
}
AUTO_VIEWER_PREDICTIONS = (
    PREDICTION_NAMES["stage2"],
    PREDICTION_NAMES["output"],
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--viewer_dir",
        type=Path,
        help="Trainer viewer directory containing point_cloud/00_input_gs.ply.",
    )
    source.add_argument(
        "--inference_multi_scene_dir",
        type=Path,
        help="AnchorSplat inference_multi scene directory containing gs/input_gs.ply and gs/output_gs.ply.",
    )
    source.add_argument("--input_ply", type=Path, help="Row-aligned input Gaussian PLY.")
    parser.add_argument("--predicted_ply", type=Path, help="Row-aligned predicted Gaussian PLY.")
    parser.add_argument("--output_ply", type=Path, help="Alias for --predicted_ply, matching AnchorSplat inference_external.py.")
    parser.add_argument(
        "--prediction",
        choices=("auto", *PREDICTION_NAMES),
        default="auto",
        help="Prediction chosen from --viewer_dir; auto detects SplatFormer stage2 or AnchorSplat output.",
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
    parser.add_argument("--initial_view_mode", choices=("pointcloud", "gsplat"), default="pointcloud")
    parser.add_argument("--device", default="cuda", help="Torch device for gsplat background rendering.")
    parser.add_argument("--render_height", type=int, default=720, help="Background render height in pixels.")
    parser.add_argument("--render_interval", type=float, default=0.01, help="Minimum seconds between camera-motion gsplat renders.")
    parser.add_argument("--near_plane", type=float, default=1e-2)
    parser.add_argument("--far_plane", type=float, default=1e2)
    parser.add_argument("--radius_clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--rasterize_mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
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


def _resolve_viewer_prediction(point_cloud_dir, prediction):
    if prediction == "auto":
        for filename in AUTO_VIEWER_PREDICTIONS:
            candidate = point_cloud_dir / filename
            if candidate.is_file():
                return candidate
        expected = ", ".join(AUTO_VIEWER_PREDICTIONS)
        raise FileNotFoundError(
            f"Could not auto-detect a prediction PLY in {point_cloud_dir}; expected one of: {expected}"
        )
    return point_cloud_dir / PREDICTION_NAMES[prediction]


def _resolve_paths(args):
    if args.viewer_dir is not None:
        if args.predicted_ply is not None or args.output_ply is not None:
            raise ValueError("--predicted_ply/--output_ply can only be used with --input_ply, not --viewer_dir")
        point_cloud_dir = args.viewer_dir / "point_cloud"
        input_path = point_cloud_dir / DEFAULT_INPUT_NAME
        predicted_path = _resolve_viewer_prediction(point_cloud_dir, args.prediction)
    elif args.inference_multi_scene_dir is not None:
        if args.predicted_ply is not None or args.output_ply is not None:
            raise ValueError("--predicted_ply/--output_ply cannot be combined with --inference_multi_scene_dir")
        gs_dir = args.inference_multi_scene_dir / "gs"
        input_path = gs_dir / "input_gs.ply"
        predicted_path = gs_dir / "output_gs.ply"
    else:
        if args.predicted_ply is not None and args.output_ply is not None:
            raise ValueError("Pass only one of --predicted_ply and --output_ply")
        predicted_path = args.predicted_ply or args.output_ply
        if predicted_path is None:
            raise ValueError("--predicted_ply or --output_ply is required with --input_ply")
        input_path = args.input_ply
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


def _slerp_quaternions(q0, q1, t):
    """Hemisphere-correct spherical interpolation for wxyz quaternions."""
    q0 = _normalize_quaternions(q0)
    q1 = _normalize_quaternions(q1)
    dots = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    omega = np.arccos(dots)
    sin_omega = np.sin(omega)
    linear = _normalize_quaternions((1.0 - t) * q0 + t * q1)
    s0 = np.sin((1.0 - t) * omega) / np.clip(sin_omega, 1e-8, None)
    s1 = np.sin(t * omega) / np.clip(sin_omega, 1e-8, None)
    spherical = _normalize_quaternions(s0 * q0 + s1 * q1)
    return np.where(sin_omega > 1e-5, spherical, linear)


def _interpolate_gs(input_gs, predicted_gs, t):
    """Interpolate every Gaussian attribute in PLY parameter space."""
    t = float(np.clip(t, 0.0, 1.0))
    return {
        "means": (1.0 - t) * input_gs["means"] + t * predicted_gs["means"],
        "features_dc": (1.0 - t) * input_gs["features_dc"] + t * predicted_gs["features_dc"],
        "features_rest": (1.0 - t) * input_gs["features_rest"] + t * predicted_gs["features_rest"],
        "opacities": (1.0 - t) * input_gs["opacities"] + t * predicted_gs["opacities"],
        "scales": (1.0 - t) * input_gs["scales"] + t * predicted_gs["scales"],
        "quats": _slerp_quaternions(input_gs["quats"], predicted_gs["quats"], t),
    }


def _validate_interpolation_endpoints(input_gs, predicted_gs):
    """Catch regressions in the input/prediction morph before serving a viewer."""
    at_input = _interpolate_gs(input_gs, predicted_gs, 0.0)
    at_predicted = _interpolate_gs(input_gs, predicted_gs, 1.0)
    for key in ("means", "features_dc", "features_rest", "opacities", "scales"):
        if not np.allclose(at_input[key], input_gs[key]) or not np.allclose(at_predicted[key], predicted_gs[key]):
            raise AssertionError(f"Interpolation endpoint check failed for {key}")
    for actual, expected, label in (
        (at_input["quats"], input_gs["quats"], "input quaternions"),
        (at_predicted["quats"], predicted_gs["quats"], "predicted quaternions"),
    ):
        if not np.allclose(np.abs(np.sum(actual * expected, axis=1)), 1.0, atol=1e-5):
            raise AssertionError(f"Interpolation endpoint check failed for {label}")


def _flat_rest_to_sh(features_rest):
    """Undo export_ply_forviewer's (N, 3, K) -> flattened PLY conversion."""
    flat_count = features_rest.shape[1]
    if flat_count == 0:
        return None, None
    if flat_count % 3 != 0:
        raise ValueError(f"f_rest channel count must be divisible by 3, got {flat_count}")
    coefficient_count = flat_count // 3
    sh_degree = int(round(math.sqrt(coefficient_count + 1) - 1))
    if (sh_degree + 1) ** 2 != coefficient_count + 1:
        raise ValueError(
            f"f_rest has {flat_count} channels, which does not correspond to a valid spherical-harmonic degree"
        )
    sh_rest = features_rest.reshape(-1, 3, coefficient_count).transpose(0, 2, 1)
    return sh_rest, sh_degree


def _quat_wxyz_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _camera_to_viewmat(camera):
    rotation_world_camera = _quat_wxyz_to_matrix(np.asarray(camera.wxyz, dtype=np.float32))
    translation_world_camera = np.asarray(camera.position, dtype=np.float32).reshape(3, 1)
    rotation_camera_world = rotation_world_camera.T
    translation_camera_world = -rotation_camera_world @ translation_world_camera
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation_camera_world
    viewmat[:3, 3:4] = translation_camera_world
    return viewmat


def _camera_intrinsics(camera, height):
    height = int(height)
    width = max(1, int(round(height * float(camera.aspect))))
    focal = 0.5 * height / np.tan(0.5 * float(camera.fov))
    return np.array([[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float32), width, height


def _render_gsplat_for_camera(camera, input_gs, predicted_gs, residual_values, state, render_cfg):
    import torch
    from gsplat.rendering import rasterization

    interpolated = _interpolate_gs(input_gs, predicted_gs, state["interpolation_t"])
    color_mode = state["color_mode"]
    if color_mode == "Residual heatmap":
        colors = _turbo_like_colors(residual_values[state["residual_name"]]).astype(np.float32) / 255.0
        sh_degree = None
    else:
        sh_rest, sh_degree = _flat_rest_to_sh(interpolated["features_rest"])
        if sh_rest is None:
            colors = np.clip(interpolated["features_dc"] * C0 + 0.5, 0.0, 1.0)
            sh_degree = None
        else:
            colors = np.concatenate([interpolated["features_dc"][:, None, :], sh_rest], axis=1)

    device = torch.device(render_cfg["device"])
    K_np, width, height = _camera_intrinsics(camera, render_cfg["height"])
    means = torch.from_numpy(interpolated["means"]).to(device=device, dtype=torch.float32).contiguous()
    quats = torch.from_numpy(interpolated["quats"]).to(device=device, dtype=torch.float32).contiguous()
    scales = torch.from_numpy(np.exp(np.clip(interpolated["scales"], -12.0, 4.0))).to(device=device, dtype=torch.float32).contiguous()
    opacities = torch.from_numpy(_sigmoid(interpolated["opacities"])).to(device=device, dtype=torch.float32).contiguous()
    colors = torch.from_numpy(colors).to(device=device, dtype=torch.float32).contiguous()
    viewmats = torch.from_numpy(_camera_to_viewmat(camera)).to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
    Ks = torch.from_numpy(K_np).to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
    with torch.no_grad():
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=float(render_cfg["near_plane"]),
            far_plane=float(render_cfg["far_plane"]),
            radius_clip=float(render_cfg["radius_clip"]),
            eps2d=float(render_cfg["eps2d"]),
            sh_degree=sh_degree,
            packed=True,
            render_mode="RGB",
            rasterize_mode=render_cfg["rasterize_mode"],
            camera_model="pinhole",
        )
    background = torch.as_tensor(render_cfg["background"], device=device, dtype=render_colors.dtype).view(1, 1, 1, 3)
    render_colors = render_colors + (1.0 - render_alphas) * background
    return render_colors[0].detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()


class GsplatBackgroundRenderer:
    def __init__(self, server, input_gs, predicted_gs, residual_values, state, render_cfg):
        self.server = server
        self.input_gs = input_gs
        self.predicted_gs = predicted_gs
        self.residual_values = residual_values
        self.state = state
        self.render_cfg = render_cfg
        self.lock = threading.Lock()
        self.last_render_time = {}

    def render_client(self, client, force=False):
        if self.state["view_mode"] != "gsplat":
            return
        if getattr(client.camera._state, "update_timestamp", 0.0) == 0.0:
            return
        client_id = id(client)
        now = time.monotonic()
        if not force and now - self.last_render_time.get(client_id, 0.0) < self.render_cfg["interval"]:
            return
        self.last_render_time[client_id] = now
        try:
            with self.lock:
                image = _render_gsplat_for_camera(
                    client.camera, self.input_gs, self.predicted_gs, self.residual_values, self.state, self.render_cfg
                )
            client.set_background_image(image, format="jpeg", jpeg_quality=85)
        except Exception:
            print("gsplat background render failed:", flush=True)
            traceback.print_exc()

    def render_all(self):
        for client in self.server.get_clients().values():
            self.render_client(client, force=True)

    def clear_all(self):
        color = np.round(np.clip(self.render_cfg["background"], 0.0, 1.0) * 255.0).astype(np.uint8)
        image = np.broadcast_to(color, (2, 2, 3)).copy()
        for client in self.server.get_clients().values():
            client.set_background_image(image, format="png")


def _validate_gsplat_available(device):
    try:
        import torch
        import gsplat  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("GSplat background mode requires torch and gsplat in the active environment.") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GSplat background mode requested CUDA, but torch.cuda.is_available() is False.")


def launch_viewer(input_gs, predicted_gs, residuals, display_permutation, display_count, arrow_count, args):
    try:
        import viser
    except ImportError as exc:
        raise ImportError("Viser is required for the interactive viewer. Install requirements.txt, or use --dry_run.") from exc

    if args.initial_view_mode == "gsplat":
        _validate_gsplat_available(args.device)

    server = viser.ViserServer(host=args.host, port=args.port)
    scene = _server_component(server, "scene", "")
    gui = _server_component(server, "gui", "")
    residual_options = {
        "means": residuals["means_l2"],
        "scales (log)": residuals["scale_log_l2"],
        "rotations (degrees)": residuals["quaternion_geodesic_deg"],
        "features_rest": residuals["features_rest_l2"],
    }
    state = {
        "view_mode": args.initial_view_mode,
        "color_mode": "Normal render",
        "residual_name": "means",
        "interpolation_t": 0.0,
    }
    display_indices = display_permutation[:display_count]
    input_points = input_gs["means"][display_indices]
    predicted_points = predicted_gs["means"][display_indices]
    bbox = np.concatenate([input_points, predicted_points], axis=0)
    bbox_diagonal = max(float(np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0))), 1e-4)
    point_size_min = max(bbox_diagonal * 1e-5, 1e-6)
    point_size_max = max(bbox_diagonal * 0.05, args.point_size * 10.0)
    point_size = float(np.clip(args.point_size, point_size_min, point_size_max))
    cloud_handle = None
    arrow_handles = []
    line_segments = getattr(scene, "add_line_segments", None)
    spline_arrows = line_segments is None
    warned_spline_arrow_cap = False

    render_cfg = {
        "device": args.device,
        "height": int(args.render_height),
        "interval": float(args.render_interval),
        "near_plane": float(args.near_plane),
        "far_plane": float(args.far_plane),
        "radius_clip": float(args.radius_clip),
        "eps2d": float(args.eps2d),
        "rasterize_mode": args.rasterize_mode,
        "background": np.asarray(args.background, dtype=np.float32),
    }
    background_renderer = GsplatBackgroundRenderer(server, input_gs, predicted_gs, residual_options, state, render_cfg)

    def set_display_count(new_count):
        nonlocal display_count, display_indices, input_points, predicted_points
        display_count = min(int(new_count), len(display_permutation))
        display_indices = display_permutation[:display_count]
        input_points = input_gs["means"][display_indices]
        predicted_points = predicted_gs["means"][display_indices]

    def refresh_point_cloud():
        nonlocal cloud_handle
        if cloud_handle is not None:
            cloud_handle.remove()
        points = input_points * (1.0 - state["interpolation_t"]) + predicted_points * state["interpolation_t"]
        colors = (
            _turbo_like_colors(residual_options[state["residual_name"]][display_indices])
            if state["color_mode"] == "Residual heatmap"
            else _rgb_from_dc(
                (1.0 - state["interpolation_t"]) * input_gs["features_dc"][display_indices]
                + state["interpolation_t"] * predicted_gs["features_dc"][display_indices]
            )
        )
        cloud_handle = scene.add_point_cloud(
            "/interpolated_gs", points=points, colors=colors, point_size=point_size,
            point_shape="circle", visible=state["view_mode"] == "pointcloud",
        )

    def set_arrows_visible():
        visible = (
            state["view_mode"] == "pointcloud"
            and state["residual_name"] == "means"
            and show_arrows.value
        )
        for handle in arrow_handles:
            handle.visible = visible

    def add_arrows(scale):
        nonlocal arrow_handles, warned_spline_arrow_cap
        for handle in arrow_handles:
            handle.remove()
        arrow_handles = []
        arrow_indices = display_indices[:min(arrow_count, len(display_indices))]
        arrow_start = input_gs["means"][arrow_indices]
        arrow_delta = residuals["means_delta"][arrow_indices]
        arrow_colors = _turbo_like_colors(residuals["means_l2"][arrow_indices])
        if spline_arrows and len(arrow_start) > 500:
            arrow_start, arrow_delta, arrow_colors = arrow_start[:500], arrow_delta[:500], arrow_colors[:500]
            if not warned_spline_arrow_cap:
                print("Installed Viser has no batched line segments; showing at most 500 displacement arrows.", flush=True)
                warned_spline_arrow_cap = True
        if len(arrow_start) == 0:
            return
        arrow_points = np.stack([arrow_start, arrow_start + arrow_delta * float(scale)], axis=1)
        if line_segments is not None:
            arrow_handles.append(line_segments("/displacement_vectors", points=arrow_points, colors=arrow_colors, line_width=1.5))
        else:
            for index, (points, color) in enumerate(zip(arrow_points, arrow_colors)):
                arrow_handles.append(scene.add_spline_catmull_rom(
                    f"/displacement_vectors/{index:05d}", positions=points, line_width=1.5,
                    color=tuple(int(value) for value in color), segments=1,
                ))
        set_arrows_visible()

    add_checkbox = _server_method(gui, server, "add_checkbox", "add_gui_checkbox")
    add_slider = _server_method(gui, server, "add_slider", "add_gui_slider")
    add_dropdown = _server_method(gui, server, "add_dropdown", "add_gui_dropdown")
    view_mode_selector = add_dropdown(
        "Viewer mode", options=("Point cloud", "GSplat background"),
        initial_value="GSplat background" if state["view_mode"] == "gsplat" else "Point cloud",
    )
    color_mode_selector = add_dropdown(
        "Color mode", options=("Normal render", "Residual heatmap"), initial_value=state["color_mode"],
    )
    show_arrows = add_checkbox("Show input→predicted mean directions", initial_value=False)
    residual_selector = add_dropdown("Residual parameter", options=tuple(residual_options), initial_value=state["residual_name"])

    total_count = len(display_permutation)
    display_values = sorted({min(total_count, value) for value in (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, display_count, total_count)})
    display_options = {f"{value:,}" + (" (all)" if value == total_count else ""): value for value in display_values}
    arrow_values = sorted({0, 50, 100, 250, 500, 1_000, 2_500, 5_000, arrow_count})
    arrow_options = {("0 (hide)" if value == 0 else f"{value:,}"): value for value in arrow_values}
    display_count_selector = add_dropdown("Displayed Gaussians", options=tuple(display_options), initial_value=next(label for label, value in display_options.items() if value == display_count))
    arrow_count_selector = add_dropdown("Mean direction count", options=tuple(arrow_options), initial_value=next(label for label, value in arrow_options.items() if value == arrow_count))
    interpolation_slider = add_slider("Full GS interpolation (input → predicted)", min=0.0, max=1.0, step=0.05, initial_value=state["interpolation_t"])
    point_size_slider = add_slider("Point size", min=point_size_min, max=point_size_max, step=(point_size_max - point_size_min) / 100.0, initial_value=point_size)
    arrow_scale = add_slider("Vector scale", min=0.0, max=20.0, step=0.1, initial_value=1.0)

    def refresh_active_view(force_render=True):
        refresh_point_cloud()
        set_arrows_visible()
        if state["view_mode"] == "gsplat" and force_render:
            background_renderer.render_all()

    def set_control_visibility():
        point_controls_visible = state["view_mode"] == "pointcloud"
        for control in (show_arrows, display_count_selector, arrow_count_selector, point_size_slider, arrow_scale):
            control.visible = point_controls_visible

    @view_mode_selector.on_update
    def _(_event):
        requested = "gsplat" if view_mode_selector.value == "GSplat background" else "pointcloud"
        if requested == "gsplat":
            try:
                _validate_gsplat_available(args.device)
            except RuntimeError as exc:
                print(f"Cannot enable GSplat background mode: {exc}", flush=True)
                view_mode_selector.value = "Point cloud"
                return
        state["view_mode"] = requested
        set_control_visibility()
        if requested == "gsplat":
            refresh_active_view(force_render=True)
        else:
            background_renderer.clear_all()
            refresh_active_view(force_render=False)

    @color_mode_selector.on_update
    def _(_event):
        state["color_mode"] = color_mode_selector.value
        refresh_active_view()

    @residual_selector.on_update
    def _(_event):
        state["residual_name"] = residual_selector.value
        refresh_active_view()

    @show_arrows.on_update
    def _(_event):
        if show_arrows.value and state["residual_name"] == "means" and state["view_mode"] == "pointcloud":
            add_arrows(arrow_scale.value)
        set_arrows_visible()

    @display_count_selector.on_update
    def _(_event):
        set_display_count(display_options[display_count_selector.value])
        refresh_active_view(force_render=False)
        if show_arrows.value and state["residual_name"] == "means" and state["view_mode"] == "pointcloud":
            add_arrows(arrow_scale.value)

    @arrow_count_selector.on_update
    def _(_event):
        nonlocal arrow_count
        arrow_count = arrow_options[arrow_count_selector.value]
        if show_arrows.value and state["residual_name"] == "means" and state["view_mode"] == "pointcloud":
            add_arrows(arrow_scale.value)

    @interpolation_slider.on_update
    def _(_event):
        state["interpolation_t"] = float(interpolation_slider.value)
        refresh_active_view()

    @point_size_slider.on_update
    def _(_event):
        nonlocal point_size
        point_size = float(point_size_slider.value)
        refresh_active_view(force_render=False)

    @arrow_scale.on_update
    def _(_event):
        if state["view_mode"] == "pointcloud":
            add_arrows(arrow_scale.value)

    @server.on_client_connect
    def _(client):
        @client.camera.on_update
        def _(_camera):
            background_renderer.render_client(client)
        if state["view_mode"] == "gsplat":
            background_renderer.render_client(client, force=True)
        else:
            background_renderer.clear_all()

    set_control_visibility()
    refresh_point_cloud()
    if state["view_mode"] == "gsplat":
        background_renderer.render_all()
    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    print("Point cloud mode uses sampled Gaussians; GSplat background mode renders all Gaussians.", flush=True)
    print("Residual heatmap uses blue-to-red colors from the selected residual's 1st to 99th percentile.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping Viser server.", flush=True)

def main():
    args = parse_args()
    if args.sample_count < 0 or args.arrow_count < 0:
        raise ValueError("--sample_count and --arrow_count must be non-negative")
    if args.render_height <= 0 or args.render_interval < 0.0:
        raise ValueError("--render_height must be positive and --render_interval must be non-negative")
    input_path, predicted_path = _resolve_paths(args)
    input_gs, predicted_gs = _read_gs_ply(input_path), _read_gs_ply(predicted_path)
    _validate_interpolation_endpoints(input_gs, predicted_gs)
    residuals = compute_residuals(input_gs, predicted_gs)
    output_dir = args.output_dir or predicted_path.parent / "displacement_residuals"
    # write_residual_outputs(output_dir, input_path, predicted_path, input_gs, predicted_gs, residuals, args.hist_bins, args.write_per_point_csv)

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

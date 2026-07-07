#!/usr/bin/env python3
"""Visualize interpolation between paired Gaussian PLY folders.

For densify_init folders, row i in 03_input_high_res_gs.ply is paired with
row i in 02_gt_high_res_gs.ply. For point_cloud folders, input.ply is paired
with output.ply.
"""

import argparse
import re
import threading
import time
import csv
import traceback
from pathlib import Path

import numpy as np
from plyfile import PlyData

SOURCE_PLY = "03_input_high_res_gs.ply"
TARGET_PLY = "02_gt_high_res_gs.ply"
POINT_CLOUD_INPUT_PLY = "input.ply"
POINT_CLOUD_OUTPUT_PLY = "output.ply"
C0 = 0.28209479177387814


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize true gsplat renders from paired Gaussian PLY folders."
    )
    parser.add_argument("--densify_init", action="append", default=[])
    parser.add_argument("--point_cloud_folder", action="append", default=[], help="Folder containing input.ply and output.ply to interpolate.")
    parser.add_argument("--mode", choices=("splats",), default="splats")
    parser.add_argument("--sample_count", type=int, default=0, help="Gaussians to render per folder. 0 means all points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stats_dir",
        default=None,
        help="Directory for per-property distance stats/histograms. Defaults beside each input folder.",
    )
    parser.add_argument("--hist_bins", type=int, default=80, help="Number of bins for saved distance histograms.")
    parser.add_argument("--no_stats", action="store_true", help="Disable property distance stats/histogram output.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--near_plane", type=float, default=1e-2)
    parser.add_argument("--far_plane", type=float, default=1e2)
    parser.add_argument("--radius_clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--rasterize_mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_quats(quats):
    norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / np.clip(norm, 1e-8, None)


def _sorted_prefixed_fields(names, prefix):
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name[len(prefix):]))


def _read_ply_gs(ply_path):
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"{ply_path} has no vertex element")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"{ply_path} is missing coordinate fields: {sorted(missing)}")

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    count = points.shape[0]

    color_fields = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if set(color_fields).issubset(names):
        features_dc = np.stack([vertex[name] for name in color_fields], axis=-1).astype(np.float32)
        colors = np.clip(features_dc * C0 + 0.5, 0.0, 1.0)
        colors_u8 = np.round(colors * 255.0).astype(np.uint8)
    else:
        features_dc = np.zeros((count, 3), dtype=np.float32)
        colors_u8 = np.full((count, 3), 220, dtype=np.uint8)

    rest_fields = _sorted_prefixed_fields(names, "f_rest_")
    if rest_fields:
        features_rest = np.stack([vertex[name] for name in rest_fields], axis=-1).astype(np.float32)
    else:
        features_rest = np.zeros((count, 0), dtype=np.float32)

    scale_fields = ["scale_0", "scale_1", "scale_2"]
    if set(scale_fields).issubset(names):
        scales = np.stack([vertex[name] for name in scale_fields], axis=-1).astype(np.float32)
    else:
        scales = np.full((count, 3), -5.0, dtype=np.float32)

    quat_fields = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if set(quat_fields).issubset(names):
        quats = np.stack([vertex[name] for name in quat_fields], axis=-1).astype(np.float32)
        quats = _normalize_quats(quats)
    else:
        quats = np.zeros((count, 4), dtype=np.float32)
        quats[:, 0] = 1.0

    if "opacity" in names:
        opacities = np.asarray(vertex["opacity"], dtype=np.float32).reshape(-1)
    else:
        opacities = np.zeros((count,), dtype=np.float32)

    return {
        "points": points,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "colors_u8": colors_u8,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }


def _label_from_folder(path):
    path = Path(path)
    if path.name in {"densify_init", "point_cloud"} and path.parent.name:
        label = path.parent.name
    else:
        label = path.name
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return label or "pair"


def _unique_labels(labels):
    counts = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1 else f"{label}_{counts[label]}")
    return out


def _load_paired_ply_folder(folder, source_name, target_name, rng, sample_count, kind):
    folder = Path(folder).expanduser()
    source_path = folder / source_name
    target_path = folder / target_name
    if not folder.is_dir():
        raise FileNotFoundError(f"{kind} folder does not exist: {folder}")
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source PLY: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"missing target PLY: {target_path}")

    source = _read_ply_gs(source_path)
    target = _read_ply_gs(target_path)
    if source["points"].shape != target["points"].shape:
        raise ValueError(f"{folder} source/target point counts differ")
    if source["points"].shape[0] == 0:
        raise ValueError(f"{folder} contains zero points")

    count = source["points"].shape[0]
    requested_count = int(sample_count)
    initial_count = count if requested_count == 0 else min(max(requested_count, 0), count)
    permutation = rng.permutation(count)
    idx = permutation[:initial_count]
    distances = np.linalg.norm(source["points"] - target["points"], axis=1)
    return {
        "path": folder,
        "kind": kind,
        "source_name": source_name,
        "target_name": target_name,
        "label": _label_from_folder(folder),
        "source": source,
        "target": target,
        "permutation": permutation,
        "sample_idx": idx,
        "distances": distances,
        "total_count": count,
        "visible_percent": 100.0 * float(initial_count) / float(count),
        "splat_t": 0.0,
        "group_position": np.zeros(3, dtype=np.float32),
        "assignment_handles": [],
    }


def _load_densify_init_folder(densify_init, rng, sample_count):
    return _load_paired_ply_folder(densify_init, SOURCE_PLY, TARGET_PLY, rng, sample_count, "densify_init")


def _load_point_cloud_folder(point_cloud_folder, rng, sample_count):
    return _load_paired_ply_folder(
        point_cloud_folder,
        POINT_CLOUD_INPUT_PLY,
        POINT_CLOUD_OUTPUT_PLY,
        rng,
        sample_count,
        "point_cloud",
    )


def _safe_l2_distance(a, b):
    if a.shape[-1] == 0:
        return None
    return np.linalg.norm(a.astype(np.float32) - b.astype(np.float32), axis=-1)


def _quat_geodesic_distance(q0, q1):
    dots = np.sum(_normalize_quats(q0) * _normalize_quats(q1), axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _property_distances(source, target):
    values = {
        "means_l2": _safe_l2_distance(source["points"], target["points"]),
        "features_dc_l2": _safe_l2_distance(source["features_dc"], target["features_dc"]),
        "features_rest_l2": _safe_l2_distance(source["features_rest"], target["features_rest"]),
        "opacities_abs": np.abs(source["opacities"].astype(np.float32) - target["opacities"].astype(np.float32)),
        "scales_l2": _safe_l2_distance(source["scales"], target["scales"]),
        "quats_geodesic_rad": _quat_geodesic_distance(source["quats"], target["quats"]),
        "quats_l2": _safe_l2_distance(source["quats"], target["quats"]),
    }
    return {key: value for key, value in values.items() if value is not None}


def _distance_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "average": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def _stats_output_dir(base_stats_dir, data):
    if base_stats_dir is None:
        dirname = "assignment_stats" if data.get("kind") == "densify_init" else "interpolation_stats"
        return Path(data["path"]) / dirname
    return Path(base_stats_dir).expanduser() / data["label"]


def _write_property_distance_outputs(data, base_stats_dir, bins):
    out_dir = _stats_output_dir(base_stats_dir, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    distances = _property_distances(data["source"], data["target"])
    stats_path = out_dir / "property_distance_stats.csv"
    hist_path = out_dir / "property_distance_histograms.csv"
    summaries = {key: _distance_summary(value) for key, value in distances.items()}

    with stats_path.open("w", newline="") as f:
        fieldnames = ["property", "count", "min", "max", "mean", "average", "std", "median", "p95", "p99"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, summary in summaries.items():
            writer.writerow({"property": key, **summary})

    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "bin_left", "bin_right", "count"])
        for key, values in distances.items():
            counts, edges = np.histogram(values, bins=max(int(bins), 1))
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                writer.writerow([key, float(left), float(right), int(count)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(distances.keys())
        cols = 2
        rows = int(np.ceil(len(names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.2 * rows, 3.2)))
        axes = np.asarray(axes).reshape(-1)
        for ax, name in zip(axes, names):
            values = distances[name]
            summary = summaries[name]
            ax.hist(values, bins=max(int(bins), 1), color="#4C78A8", alpha=0.85)
            ax.axvline(summary["mean"], color="#F58518", linewidth=1.5, label=f"mean={summary['mean']:.4g}")
            ax.axvline(summary["median"], color="#54A24B", linewidth=1.2, label=f"median={summary['median']:.4g}")
            ax.set_title(name)
            ax.set_xlabel("distance")
            ax.set_ylabel("count")
            ax.legend(fontsize=8)
        for ax in axes[len(names):]:
            ax.axis("off")
        fig.suptitle(f"{data.get('kind', 'pair')} property distances: {data['label']}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "property_distance_histograms.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to write histogram PNG for {data['label']}: {exc}", flush=True)

    return stats_path


def _format_stats(data, mode):
    distances = data["distances"][data["sample_idx"]]
    dist_stats = "empty" if distances.size == 0 else f"{distances.min():.6g}/{distances.mean():.6g}/{distances.max():.6g}"
    return (
        f"{data['label']}: kind={data.get('kind', 'pair')} path={data['path']} "
        f"source={data.get('source_name', 'source')} target={data.get('target_name', 'target')} "
        f"total={data['total_count']} sampled={len(data['sample_idx'])} "
        f"visible_percent={data['visible_percent']:.2f} "
        f"dist[min/mean/max]={dist_stats} global_mode={mode}"
    )


def _bbox_diagonal(point_sets):
    points = np.concatenate(point_sets, axis=0)
    return float(max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)), 1e-3))


def _distance_colors(distances):
    distances = np.asarray(distances, dtype=np.float32)
    if distances.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    min_d = float(distances.min())
    max_d = float(distances.max())
    t = np.zeros_like(distances) if max_d <= min_d else (distances - min_d) / (max_d - min_d)
    stops = np.array([[49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233], [254, 224, 144], [253, 174, 97], [215, 48, 39]], dtype=np.float32)
    scaled = t * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (scaled - lo)[:, None]
    return np.clip(stops[lo] * (1.0 - frac) + stops[hi] * frac, 0, 255).astype(np.uint8)


def _remove_handles(handles):
    for handle in handles:
        handle.remove()
    handles.clear()


def _visible_indices(data):
    count = data["total_count"]
    visible_count = min(max(int(round(count * float(data["visible_percent"]) / 100.0)), 0), count)
    return data["permutation"][:visible_count]


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _lerp_quats(q0, q1, t):
    q1 = q1.copy()
    same_hemi = np.sum(q0 * q1, axis=-1, keepdims=True) >= 0.0
    q1 = np.where(same_hemi, q1, -q1)
    return _normalize_quats(_lerp(q0, q1, t))


def _interpolated_gs(data, idx):
    t = float(data["splat_t"])
    source = data["source"]
    target = data["target"]
    features_dc = _lerp(source["features_dc"][idx], target["features_dc"][idx], t)
    colors = np.clip(features_dc * C0 + 0.5, 0.0, 1.0).astype(np.float32)
    return {
        "points": _lerp(source["points"][idx], target["points"][idx], t) + data["group_position"][None, :],
        "scales": _lerp(source["scales"][idx], target["scales"][idx], t),
        "quats": _lerp_quats(source["quats"][idx], target["quats"][idx], t),
        "opacities": _lerp(source["opacities"][idx], target["opacities"][idx], t),
        "colors": colors,
    }


def _quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _camera_to_viewmat(camera):
    R_wc = _quat_wxyz_to_matrix(np.asarray(camera.wxyz, dtype=np.float32))
    t_wc = np.asarray(camera.position, dtype=np.float32).reshape(3, 1)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_cw
    viewmat[:3, 3:4] = t_cw
    return viewmat


def _camera_intrinsics(camera, height):
    height = int(height)
    width = max(1, int(round(height * float(camera.aspect))))
    fy = 0.5 * height / np.tan(0.5 * float(camera.fov))
    K = np.array([[fy, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, width, height


def _merged_render_gs(loaded, scene_state):
    if scene_state["mode"] != "splats":
        return None
    groups = []
    for data in loaded:
        idx = _visible_indices(data)
        data["sample_idx"] = idx
        if idx.size > 0:
            groups.append(_interpolated_gs(data, idx))
    if not groups:
        return None
    return {key: np.concatenate([group[key] for group in groups], axis=0) for key in groups[0]}


def _render_gsplat_for_camera(camera, loaded, scene_state, render_cfg):
    import torch
    from gsplat.rendering import rasterization

    K_np, width, height = _camera_intrinsics(camera, render_cfg["height"])
    gs = _merged_render_gs(loaded, scene_state)
    if gs is None:
        bg = np.asarray(render_cfg["background"], dtype=np.float32)
        return np.broadcast_to((bg * 255.0).astype(np.uint8), (height, width, 3)).copy()

    device = torch.device(render_cfg["device"])
    means = torch.from_numpy(gs["points"]).to(device=device, dtype=torch.float32).contiguous()
    quats = torch.from_numpy(gs["quats"]).to(device=device, dtype=torch.float32).contiguous()
    scales = torch.from_numpy(np.exp(np.clip(gs["scales"], -12.0, 4.0))).to(device=device, dtype=torch.float32).contiguous()
    opacities = torch.from_numpy(_sigmoid(gs["opacities"])).to(device=device, dtype=torch.float32).contiguous()
    colors = torch.from_numpy(gs["colors"]).to(device=device, dtype=torch.float32).contiguous()
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
            sh_degree=None,
            packed=True,
            render_mode="RGB",
            rasterize_mode=render_cfg["rasterize_mode"],
            camera_model="pinhole",
        )
    background = torch.tensor(render_cfg["background"], device=device, dtype=render_colors.dtype).view(1, 1, 1, 3)
    render_colors = render_colors + (1.0 - render_alphas) * background
    return render_colors[0].detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()


class GsplatBackgroundRenderer:
    def __init__(self, server, loaded, scene_state, render_cfg):
        self.server = server
        self.loaded = loaded
        self.scene_state = scene_state
        self.render_cfg = render_cfg
        self.lock = threading.Lock()
        self.last_render_time = {}

    def render_client(self, client, force=False):
        if getattr(client.camera._state, "update_timestamp", 0.0) == 0.0:
            return
        client_id = id(client)
        now = time.monotonic()
        if not force and now - self.last_render_time.get(client_id, 0.0) < float(self.render_cfg.get("min_render_interval", 0.25)):
            return
        self.last_render_time[client_id] = now
        try:
            with self.lock:
                image = _render_gsplat_for_camera(client.camera, self.loaded, self.scene_state, self.render_cfg)
            client.set_background_image(image, format="jpeg", jpeg_quality=85)
        except Exception:
            print("gsplat background render failed:", flush=True)
            traceback.print_exc()

    def render_all(self):
        for client in self.server.get_clients().values():
            self.render_client(client, force=True)


def _add_group_to_server(server, data, group_index, group_count, layout_diag, bg_renderer):
    label = data["label"]
    root = f"/assignments/{label}"
    group_gap = 1.4 * layout_diag
    group_y = group_index * group_gap
    if group_count > 1:
        group_y -= 0.5 * group_gap * (group_count - 1)
    data["group_position"] = np.array([0.0, float(group_y), 0.0], dtype=np.float32)
    controls = server.add_transform_controls(
        root,
        scale=max(0.15 * layout_diag, 1e-3),
        line_width=2.0,
        disable_rotations=True,
        position=data["group_position"],
    )

    percent_slider = server.add_gui_slider(
        f"{label} visible %",
        min=0.0,
        max=100.0,
        step=1.0,
        initial_value=float(data["visible_percent"]),
        marks=((0.0, "0%"), (50.0, "50%"), (100.0, "100%")),
    )
    interp_slider = server.add_gui_slider(
        f"{label} input-target",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=float(data["splat_t"]),
        marks=((0.0, "input"), (0.5, "mix"), (1.0, "target")),
    )

    def refresh_group():
        data["visible_percent"] = float(percent_slider.value)
        data["splat_t"] = float(interp_slider.value)
        bg_renderer.render_all()

    @percent_slider.on_update
    def _(_event):
        refresh_group()

    @interp_slider.on_update
    def _(_event):
        refresh_group()

    @controls.on_update
    def _(handle):
        data["group_position"] = np.asarray(handle.position, dtype=np.float32)
        bg_renderer.render_all()


def main():
    args = _parse_args()
    if args.sample_count < 0:
        raise ValueError(f"--sample_count must be non-negative, got {args.sample_count}")
    if args.render_height <= 0:
        raise ValueError(f"--render_height must be positive, got {args.render_height}")

    if not args.densify_init and not args.point_cloud_folder:
        raise ValueError("Pass at least one --densify_init or --point_cloud_folder")

    rng = np.random.default_rng(args.seed)
    loaded = []
    loaded.extend(_load_densify_init_folder(path, rng, args.sample_count) for path in args.densify_init)
    loaded.extend(_load_point_cloud_folder(path, rng, args.sample_count) for path in args.point_cloud_folder)
    labels = _unique_labels([data["label"] for data in loaded])
    scene_state = {"mode": args.mode}
    for data, label in zip(loaded, labels):
        data["label"] = label
        print(_format_stats(data, scene_state["mode"]), flush=True)
        if not args.no_stats:
            stats_path = _write_property_distance_outputs(data, args.stats_dir, args.hist_bins)
            print(f"Wrote property distance stats: {stats_path}", flush=True)
    if args.dry_run:
        return

    import torch
    import viser
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.add_frame("/assignments", show_axes=False, visible=True)
    render_cfg = {
        "height": int(args.render_height),
        "near_plane": float(args.near_plane),
        "far_plane": float(args.far_plane),
        "radius_clip": float(args.radius_clip),
        "eps2d": float(args.eps2d),
        "rasterize_mode": args.rasterize_mode,
        "background": tuple(float(v) for v in args.background),
        "device": args.device,
        "min_render_interval": 0.25,
    }
    bg_renderer = GsplatBackgroundRenderer(server, loaded, scene_state, render_cfg)

    @server.on_client_connect
    def _(client):
        @client.camera.on_update
        def _(_camera):
            bg_renderer.render_client(client)

    layout_diag = max(_bbox_diagonal([data["source"]["points"], data["target"]["points"]]) for data in loaded)
    for group_index, data in enumerate(loaded):
        _add_group_to_server(server, data, group_index, len(loaded), layout_diag, bg_renderer)

    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    print("This viewer renders gsplat backgrounds only.", flush=True)
    print("Loaded densify_init and point_cloud folders render together with gsplat into the background.", flush=True)
    print("Use --point_cloud_folder /path/to/point_cloud to interpolate input.ply -> output.ply.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viser server.", flush=True)


if __name__ == "__main__":
    main()

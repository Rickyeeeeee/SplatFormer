#!/usr/bin/env python3
"""Visualize SR densification assignments saved by overfit-sr-mse.py.

The densify_init folder stores target-ordered point clouds. Row i in
03_input_high_res_gs.ply is the assigned input Gaussian for row i in
02_gt_high_res_gs.ply.
"""

import argparse
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from plyfile import PlyData

SOURCE_PLY = "03_input_high_res_gs.ply"
TARGET_PLY = "02_gt_high_res_gs.ply"
C0 = 0.28209479177387814


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize assignment links and true gsplat renders from densify_init folders."
    )
    parser.add_argument("--densify_init", action="append", required=True)
    parser.add_argument("--mode", choices=("assignments", "splats", "both"), default="assignments")
    parser.add_argument("--sample_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.015)
    parser.add_argument("--line_width", type=float, default=1.0)
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
        "colors_u8": colors_u8,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }


def _label_from_densify_init(path):
    path = Path(path)
    label = path.parent.name if path.name == "densify_init" and path.parent.name else path.name
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return label or "assignment"


def _unique_labels(labels):
    counts = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1 else f"{label}_{counts[label]}")
    return out


def _load_assignment_folder(densify_init, rng, sample_count):
    densify_init = Path(densify_init).expanduser()
    source_path = densify_init / SOURCE_PLY
    target_path = densify_init / TARGET_PLY
    if not densify_init.is_dir():
        raise FileNotFoundError(f"densify_init folder does not exist: {densify_init}")
    if not source_path.is_file():
        raise FileNotFoundError(f"missing assigned input PLY: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"missing target PLY: {target_path}")

    source = _read_ply_gs(source_path)
    target = _read_ply_gs(target_path)
    if source["points"].shape != target["points"].shape:
        raise ValueError(f"{densify_init} input/target point counts differ")
    if source["points"].shape[0] == 0:
        raise ValueError(f"{densify_init} contains zero points")

    count = source["points"].shape[0]
    initial_count = min(max(int(sample_count), 0), count)
    permutation = rng.permutation(count)
    idx = permutation[:initial_count]
    distances = np.linalg.norm(source["points"] - target["points"], axis=1)
    return {
        "path": densify_init,
        "label": _label_from_densify_init(densify_init),
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


def _format_stats(data, mode):
    distances = data["distances"][data["sample_idx"]]
    dist_stats = "empty" if distances.size == 0 else f"{distances.min():.6g}/{distances.mean():.6g}/{distances.max():.6g}"
    return (
        f"{data['label']}: path={data['path']} total={data['total_count']} "
        f"sampled={len(data['sample_idx'])} visible_percent={data['visible_percent']:.2f} "
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
    if scene_state["mode"] not in ("splats", "both"):
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


def _add_assignment_nodes(server, data, point_size, line_width):
    label = data["label"]
    root = f"/assignments/{label}/assignment"
    idx = _visible_indices(data)
    data["sample_idx"] = idx
    handles = []
    if idx.size == 0:
        handles.append(server.add_label(f"{root}/label", text=f"{label}: 0 links", position=(0.0, 0.0, 0.0)))
        return handles
    source = data["source"]["points"][idx]
    target = data["target"]["points"][idx]
    distances = data["distances"][idx]
    diag = _bbox_diagonal([source, target])
    cloud_gap = 0.8 * diag
    source_vis = source + np.array([-0.5 * cloud_gap, 0.0, 0.0], dtype=np.float32)
    target_vis = target + np.array([0.5 * cloud_gap, 0.0, 0.0], dtype=np.float32)
    link_colors = _distance_colors(distances)
    handles.append(server.add_point_cloud(f"{root}/assigned_input", points=source_vis, colors=data["source"]["colors_u8"][idx], point_size=point_size, point_shape="circle"))
    handles.append(server.add_point_cloud(f"{root}/target", points=target_vis, colors=data["target"]["colors_u8"][idx], point_size=point_size, point_shape="circle"))
    for i, (start, end, color) in enumerate(zip(source_vis, target_vis, link_colors)):
        delta = end - start
        control_points = np.stack([start + delta / 3.0, start + 2.0 * delta / 3.0], axis=0)
        handles.append(server.add_spline_cubic_bezier(f"{root}/links/{i:06d}", positions=np.stack([start, end], axis=0), control_points=control_points, line_width=line_width, color=tuple(int(c) for c in color), segments=2))
    center = np.concatenate([source_vis, target_vis], axis=0).mean(axis=0).astype(np.float32)
    handles.append(server.add_label(f"{root}/label", text=f"{label}: {len(idx)} links", position=center + np.array([0.0, 0.0, 0.15 * diag], dtype=np.float32)))
    return handles


def _refresh_assignment_group(server, data, scene_state, point_size, line_width):
    _remove_handles(data["assignment_handles"])
    if scene_state["mode"] in ("assignments", "both"):
        data["assignment_handles"] = _add_assignment_nodes(server, data, point_size, line_width)


class GsplatBackgroundRenderer:
    def __init__(self, server, loaded, scene_state, render_cfg):
        self.server = server
        self.loaded = loaded
        self.scene_state = scene_state
        self.render_cfg = render_cfg
        self.lock = threading.Lock()

    def render_client(self, client):
        if getattr(client.camera._state, "update_timestamp", 0.0) == 0.0:
            return
        try:
            with self.lock:
                image = _render_gsplat_for_camera(client.camera, self.loaded, self.scene_state, self.render_cfg)
            client.set_background_image(image, format="jpeg", jpeg_quality=85)
        except Exception:
            print("gsplat background render failed:", flush=True)
            traceback.print_exc()

    def render_all(self):
        for client in self.server.get_clients().values():
            self.render_client(client)


def _refresh_all_assignments(server, loaded, scene_state, point_size, line_width):
    for data in loaded:
        _refresh_assignment_group(server, data, scene_state, point_size, line_width)


def _add_group_to_server(server, data, group_index, group_count, layout_diag, point_size, line_width, scene_state, bg_renderer):
    label = data["label"]
    root = f"/assignments/{label}"
    group_gap = 1.4 * layout_diag
    group_y = group_index * group_gap
    if group_count > 1:
        group_y -= 0.5 * group_gap * (group_count - 1)
    data["group_position"] = np.array([0.0, float(group_y), 0.0], dtype=np.float32)
    controls = server.add_transform_controls(root, scale=max(0.15 * layout_diag, 1e-3), line_width=2.0, disable_rotations=True, position=data["group_position"])
    server.add_frame(f"{root}/assignment", show_axes=False, visible=True)
    server.add_frame(f"{root}/assignment/links", show_axes=False, visible=True)
    _refresh_assignment_group(server, data, scene_state, point_size, line_width)

    percent_slider = server.add_gui_slider(f"{label} visible %", min=0.0, max=100.0, step=1.0, initial_value=float(data["visible_percent"]), marks=((0.0, "0%"), (50.0, "50%"), (100.0, "100%")))
    interp_slider = server.add_gui_slider(f"{label} input-target", min=0.0, max=1.0, step=0.01, initial_value=float(data["splat_t"]), marks=((0.0, "input"), (0.5, "mix"), (1.0, "target")))

    def refresh_group():
        data["visible_percent"] = float(percent_slider.value)
        data["splat_t"] = float(interp_slider.value)
        _refresh_assignment_group(server, data, scene_state, point_size, line_width)
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

    rng = np.random.default_rng(args.seed)
    loaded = [_load_assignment_folder(path, rng, args.sample_count) for path in args.densify_init]
    labels = _unique_labels([data["label"] for data in loaded])
    scene_state = {"mode": args.mode}
    for data, label in zip(loaded, labels):
        data["label"] = label
        print(_format_stats(data, scene_state["mode"]), flush=True)
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
    }
    bg_renderer = GsplatBackgroundRenderer(server, loaded, scene_state, render_cfg)

    mode_dropdown = server.add_gui_dropdown("Display mode", ("assignments", "splats", "both"), initial_value=scene_state["mode"])

    @mode_dropdown.on_update
    def _(_event):
        scene_state["mode"] = mode_dropdown.value
        _refresh_all_assignments(server, loaded, scene_state, args.point_size, args.line_width)
        bg_renderer.render_all()

    @server.on_client_connect
    def _(client):
        @client.camera.on_update
        def _(_camera):
            bg_renderer.render_client(client)

    layout_diag = max(_bbox_diagonal([data["source"]["points"], data["target"]["points"]]) for data in loaded)
    for group_index, data in enumerate(loaded):
        _add_group_to_server(server, data, group_index, len(loaded), layout_diag, args.point_size, args.line_width, scene_state, bg_renderer)

    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    print("Display mode is global. In splats/both, all loaded assignment folders render together with gsplat into the background.", flush=True)
    print("Pass EMD and nearest densify_init folders to see two separate rendered splat clouds.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viser server.", flush=True)


if __name__ == "__main__":
    main()

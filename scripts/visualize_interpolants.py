#!/usr/bin/env python3
"""Interactively visualize analytic Gaussian interpolants for one SR scene.

The viewer loads the identity-paired ``fit_lr_to_hr`` Gaussians configured by a
Gin experiment. It evaluates the paths from :mod:`sr.interpolants` directly;
no trained checkpoint is required.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/visualize_interpolants.py \
    --config configs/dataset/objaverse-sr.gin \
    --config configs/visualize/interpolants.gin \
    --scene_name 3e288ee8aced4a0797e66d53536112b1 \
    --split test --port 8082
"""

import argparse
from pathlib import Path
import sys
import threading
import time
import traceback

import gin
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.GS_SR import SplatFactoSRDataset
from sr import interpolants
from utils.gs_normalization import GaussianStandardizer, QUATERNION_REPRESENTATIONS
from utils.gs_utils import _prepare_render_inputs


C0 = 0.28209479177387814


@gin.configurable
def flow_matching(
    flow_steps=10,
    flow_noise_std=1.0,
    flow_t_eps=1e-4,
    loss_type="velocity",
    interpolant_type="linear",
    loss_rollout_steps=10,
    eval_noise_seed=0,
    fixed_train_noise=False,
    train_noise_seed=0,
    normalization_variance_floor=1e-8,
    quaternion_representation="raw_standardized",
    gs_statistics_path="/project/ricky/splatformer-sr-data-scaled/gs_statistics.json",
):
    """Register and return the subset of training flow settings used here."""
    if quaternion_representation not in QUATERNION_REPRESENTATIONS:
        raise ValueError(f"Unsupported quaternion_representation={quaternion_representation!r}")
    return {
        "quaternion_representation": quaternion_representation,
        "interpolant_type": interpolant_type,
        "flow_noise_std": float(flow_noise_std),
        "eval_noise_seed": int(eval_noise_seed),
        "normalization_variance_floor": float(normalization_variance_floor),
        "gs_statistics_path": gs_statistics_path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", required=True, help="Gin file; repeat to compose configs.")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--gin_param", action="append", default=[])
    parser.add_argument("--noise_seed", type=int, default=None, help="Defaults to flow_matching.eval_noise_seed.")
    parser.add_argument("--noise_scale", type=float, default=None, help="Defaults to flow_matching.flow_noise_std.")
    parser.add_argument("--sample_count", type=int, default=25_000, help="Maximum displayed points; 0 displays all.")
    parser.add_argument("--seed", type=int, default=0, help="Point display sampling seed.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--point_size", type=float, default=0.003)
    parser.add_argument("--initial_view_mode", choices=("pointcloud", "gsplat"), default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_interval", type=float, default=0.01)
    parser.add_argument("--near_plane", type=float, default=1e-2)
    parser.add_argument("--far_plane", type=float, default=1e2)
    parser.add_argument("--radius_clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--rasterize_mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_count < 0 or args.point_size <= 0:
        parser.error("sample_count must be nonnegative and point_size must be positive")
    if args.render_height <= 0 or args.render_interval < 0:
        parser.error("render_height must be positive and render_interval must be nonnegative")
    if args.near_plane <= 0 or args.far_plane <= args.near_plane:
        parser.error("near_plane must be positive and far_plane must be greater than near_plane")
    if args.radius_clip < 0 or args.eps2d <= 0:
        parser.error("radius_clip must be nonnegative and eps2d must be positive")
    if not np.isfinite(args.background).all() or min(args.background) < 0 or max(args.background) > 1:
        parser.error("background components must be finite values in [0, 1]")
    if args.noise_scale is not None and (not np.isfinite(args.noise_scale) or args.noise_scale < 0):
        parser.error("noise_scale must be finite and nonnegative")
    return args


def cpu_snapshot(gaussians):
    return {key: value.detach().cpu().clone() for key, value in gaussians.items()}


def load_scene(config, scene_name, split="test", bindings=(), noise_seed=None, noise_scale=None):
    """Load and standardize one identity-paired LR-to-HR scene."""
    gin.clear_config()
    config_paths = [config] if isinstance(config, (str, Path)) else config
    for config_path in config_paths:
        gin.parse_config_file(str(config_path), skip_unknown=True)
    if bindings:
        gin.parse_config("\n".join(bindings))
    flow_cfg = flow_matching()
    dataset = SplatFactoSRDataset.from_gin_scope(
        f"{split}_dataset", load_src_gs=True, load_tgt_gs=True,
        load_src_images=False, load_tgt_images=False, split_across_gpus=False,
    )
    scene_idx = dataset.scene_index(scene_name)
    scene = dataset.load_scene(scene_idx, sample_views=False, fit_alignment="fit_lr_to_hr")
    source = cpu_snapshot(scene["data"][dataset.src_resolution]["gs_params"])
    target = cpu_snapshot(scene["fit_lr_to_hr"]["tgt_gs"])
    for key in source:
        if key not in target or source[key].shape != target[key].shape:
            raise ValueError(f"fit_lr_to_hr is not identity-paired for {key}")
    standardizer = GaussianStandardizer(flow_cfg["gs_statistics_path"], flow_cfg["normalization_variance_floor"], flow_cfg["quaternion_representation"])
    source, target = standardizer.prepare_endpoints(source, target, align_target_sign=flow_cfg["interpolant_type"] != "one_sided")
    source_flow, target_flow = standardizer.encode(source), standardizer.encode(target)
    resolved_seed = flow_cfg["eval_noise_seed"] if noise_seed is None else int(noise_seed)
    resolved_scale = flow_cfg["flow_noise_std"] if noise_scale is None else float(noise_scale)
    noise = interpolants.seeded_noise_like(target_flow, resolved_seed + scene_idx)
    metadata = {
        "scene_name": scene["scene_name"], "scene_idx": scene_idx, "split": split,
        "source_resolution": dataset.src_resolution, "target_resolution": dataset.tgt_resolution,
        "noise_seed": resolved_seed, "noise_scale": resolved_scale,
        "quaternion_representation": flow_cfg["quaternion_representation"],
    }
    return source, target, source_flow, target_flow, noise, standardizer, metadata


def analytic_state(mode, t, noise_scale, source, target, source_flow, target_flow, noise, standardizer):
    """Return a decoded path state while handling noisy endpoints exactly."""
    if mode not in interpolants.MODES:
        raise ValueError(f"Unknown interpolant mode {mode!r}")
    t = float(t)
    if not 0 <= t <= 1:
        raise ValueError("Interpolant time must be in [0, 1]")
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    if t == 0:
        state = noise if mode == "one_sided" else source_flow
    elif t == 1:
        state = target_flow
    else:
        path_source = None if mode == "one_sided" else source_flow
        source_means = None if mode == "one_sided" else source["means"]
        state = interpolants.construct_path(
            path_source, target_flow, t, mode, source_means, target["means"], noise_scale, noise=noise
        )["query"]
    return standardizer.decode(state)


def _rgb_from_gaussians(gaussians, indices):
    features_dc = gaussians["features_dc"][indices]
    if gaussians["features_rest"].shape[1] == 0:
        rgb = torch.sigmoid(features_dc)
    else:
        rgb = torch.clamp(features_dc * C0 + 0.5, 0, 1)
    return rgb.mul(255).round().byte().numpy()


def _quat_wxyz_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def _camera_to_viewmat(camera):
    rotation_world_camera = _quat_wxyz_to_matrix(np.asarray(camera.wxyz, dtype=np.float32))
    translation_world_camera = np.asarray(camera.position, dtype=np.float32).reshape(3, 1)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation_world_camera.T
    viewmat[:3, 3:4] = -rotation_world_camera.T @ translation_world_camera
    return viewmat


def _camera_intrinsics(camera, height):
    height = int(height)
    width = max(1, int(round(height * float(camera.aspect))))
    focal = 0.5 * height / np.tan(0.5 * float(camera.fov))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32), width


def _render_gsplat_for_camera(camera, gaussians, render_cfg):
    from gsplat.rendering import rasterization

    device = torch.device(render_cfg["device"])
    current = {key: value.to(device) for key, value in gaussians.items()}
    means, quats, scales, opacities, colors, sh_degree = _prepare_render_inputs(current)
    intrinsics, width = _camera_intrinsics(camera, render_cfg["height"])
    viewmats = torch.from_numpy(_camera_to_viewmat(camera)).to(device).unsqueeze(0).contiguous()
    Ks = torch.from_numpy(intrinsics).to(device).unsqueeze(0).contiguous()
    with torch.no_grad():
        rgb, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=render_cfg["height"],
            near_plane=render_cfg["near_plane"], far_plane=render_cfg["far_plane"],
            radius_clip=render_cfg["radius_clip"], eps2d=render_cfg["eps2d"],
            sh_degree=sh_degree, packed=True, render_mode="RGB",
            rasterize_mode=render_cfg["rasterize_mode"], camera_model="pinhole",
        )
    background = torch.as_tensor(render_cfg["background"], device=device, dtype=rgb.dtype).view(1, 1, 1, 3)
    return (rgb + (1 - alpha) * background)[0].clamp(0, 1).mul(255).byte().cpu().numpy()


class GsplatBackgroundRenderer:
    """Render the current state from native Viser cameras, one GPU job at a time."""

    def __init__(self, server, gaussians, state, render_cfg):
        self.server, self.gaussians, self.state, self.render_cfg = server, gaussians, state, render_cfg
        self.lock = threading.Lock()
        self.last_render_time = {}

    def update_gaussians(self, gaussians):
        with self.lock:
            self.gaussians = gaussians

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
                image = _render_gsplat_for_camera(client.camera, self.gaussians, self.render_cfg)
            client.set_background_image(image, format="jpeg", jpeg_quality=85)
        except Exception:
            print("GSplat background render failed:", flush=True)
            traceback.print_exc()

    def render_all(self):
        for client in self.server.get_clients().values():
            self.render_client(client, force=True)

    def clear_all(self):
        color = np.round(np.asarray(self.render_cfg["background"]) * 255).astype(np.uint8)
        image = np.broadcast_to(color, (2, 2, 3)).copy()
        for client in self.server.get_clients().values():
            client.set_background_image(image, format="png")


def _server_component(server, modern_name):
    return getattr(server, modern_name, server)


def _server_method(component, server, modern_name, legacy_name):
    method = getattr(component, modern_name, None) or getattr(server, legacy_name, None)
    if method is None:
        raise AttributeError(f"Installed Viser exposes neither {modern_name} nor {legacy_name}")
    return method


def _validate_gsplat_available(device):
    try:
        import gsplat  # noqa: F401
    except ImportError as error:
        raise RuntimeError("GSplat background mode requires gsplat") from error
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GSplat background mode requested CUDA, but CUDA is unavailable")


def launch_viewer(source, target, source_flow, target_flow, noise, standardizer, metadata, args):
    try:
        import viser
    except ImportError as error:
        raise ImportError("Viser is required for the interactive viewer; use --dry_run for validation") from error
    if args.initial_view_mode == "gsplat":
        _validate_gsplat_available(args.device)

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = _server_component(server, "scene"), _server_component(server, "gui")
    state = {"mode": "linear", "time": 0.0, "noise_scale": metadata["noise_scale"], "view_mode": args.initial_view_mode}
    current = analytic_state(state["mode"], state["time"], state["noise_scale"], source, target, source_flow, target_flow, noise, standardizer)
    render_cfg = {
        "device": args.device, "height": args.render_height, "interval": args.render_interval,
        "near_plane": args.near_plane, "far_plane": args.far_plane,
        "radius_clip": args.radius_clip, "eps2d": args.eps2d,
        "rasterize_mode": args.rasterize_mode, "background": np.asarray(args.background, dtype=np.float32),
    }
    renderer = GsplatBackgroundRenderer(server, current, state, render_cfg)
    permutation = np.random.default_rng(args.seed).permutation(len(source["means"]))
    display_count = len(permutation) if args.sample_count == 0 else min(args.sample_count, len(permutation))
    point_size = args.point_size
    handles = {"current": None, "source": None, "target": None}

    add_checkbox = _server_method(gui, server, "add_checkbox", "add_gui_checkbox")
    add_dropdown = _server_method(gui, server, "add_dropdown", "add_gui_dropdown")
    add_slider = _server_method(gui, server, "add_slider", "add_gui_slider")
    view_mode = add_dropdown("Viewer mode", options=("GSplat background", "Point cloud"), initial_value="GSplat background" if state["view_mode"] == "gsplat" else "Point cloud")
    mode = add_dropdown("Interpolant", options=interpolants.MODES, initial_value=state["mode"])
    time_slider = add_slider("Interpolant time", min=0.0, max=1.0, step=0.01, initial_value=state["time"])
    noise_limit = max(4.0, 2 * state["noise_scale"])
    noise_slider = add_slider("Noise scale", min=0.0, max=noise_limit, step=0.05, initial_value=state["noise_scale"])
    show_source = add_checkbox("Show source points", initial_value=False)
    show_target = add_checkbox("Show fitted-target points", initial_value=False)
    counts = sorted({min(len(permutation), value) for value in (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, display_count, len(permutation))})
    count_options = {f"{value:,}" + (" (all)" if value == len(permutation) else ""): value for value in counts}
    count_selector = add_dropdown("Displayed Gaussians", options=tuple(count_options), initial_value=next(label for label, value in count_options.items() if value == display_count))
    bbox = torch.cat([source["means"], target["means"]]).numpy()
    diagonal = max(float(np.linalg.norm(bbox.max(0) - bbox.min(0))), 1e-4)
    size_min, size_max = max(diagonal * 1e-5, 1e-6), max(diagonal * .05, point_size * 10)
    point_size = float(np.clip(point_size, size_min, size_max))
    point_size_slider = add_slider("Point size", min=size_min, max=size_max, step=(size_max - size_min) / 100, initial_value=point_size)

    def replace_handle(name, points, colors, visible):
        if handles[name] is not None:
            handles[name].remove()
        handles[name] = scene.add_point_cloud(f"/{name}", points=points, colors=colors, point_size=point_size, point_shape="circle", visible=visible)

    def refresh_points():
        indices = permutation[:display_count]
        point_mode = state["view_mode"] == "pointcloud"
        replace_handle("current", current["means"][indices].numpy(), _rgb_from_gaussians(current, indices), point_mode)
        replace_handle("source", source["means"][indices].numpy(), np.tile((40, 210, 230), (len(indices), 1)), point_mode and show_source.value)
        replace_handle("target", target["means"][indices].numpy(), np.tile((60, 230, 90), (len(indices), 1)), point_mode and show_target.value)

    def refresh_state(force_render=True):
        nonlocal current
        current = analytic_state(state["mode"], state["time"], state["noise_scale"], source, target, source_flow, target_flow, noise, standardizer)
        renderer.update_gaussians(current)
        refresh_points()
        if force_render and state["view_mode"] == "gsplat":
            renderer.render_all()

    def set_point_controls():
        visible = state["view_mode"] == "pointcloud"
        for control in (show_source, show_target, count_selector, point_size_slider):
            control.visible = visible

    @view_mode.on_update
    def _(_event):
        requested = "gsplat" if view_mode.value == "GSplat background" else "pointcloud"
        if requested == "gsplat":
            try:
                _validate_gsplat_available(args.device)
            except RuntimeError as error:
                print(f"Cannot enable GSplat mode: {error}", flush=True)
                view_mode.value = "Point cloud"
                return
        state["view_mode"] = requested
        set_point_controls()
        if requested == "gsplat":
            refresh_state(force_render=True)
        else:
            renderer.clear_all()
            refresh_points()

    @mode.on_update
    def _(_event):
        state["mode"] = mode.value
        refresh_state()

    @time_slider.on_update
    def _(_event):
        state["time"] = float(time_slider.value)
        refresh_state()

    @noise_slider.on_update
    def _(_event):
        state["noise_scale"] = float(noise_slider.value)
        refresh_state()

    @show_source.on_update
    def _(_event):
        refresh_points()

    @show_target.on_update
    def _(_event):
        refresh_points()

    @count_selector.on_update
    def _(_event):
        nonlocal display_count
        display_count = count_options[count_selector.value]
        refresh_points()

    @point_size_slider.on_update
    def _(_event):
        nonlocal point_size
        point_size = float(point_size_slider.value)
        refresh_points()

    @server.on_client_connect
    def _(client):
        @client.camera.on_update
        def _(_camera):
            renderer.render_client(client)
        if state["view_mode"] == "gsplat":
            renderer.render_client(client, force=True)
        else:
            renderer.clear_all()

    set_point_controls()
    refresh_points()
    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    print("Native Viser camera controls drive per-client GSplat background renders.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Viser server.", flush=True)
        if hasattr(server, "stop"):
            server.stop()


def validate_paths(source, target, source_flow, target_flow, noise, standardizer, noise_scale):
    """Evaluate representative states and assert exact analytic endpoints."""
    for mode in interpolants.MODES:
        states = [analytic_state(mode, t, noise_scale, source, target, source_flow, target_flow, noise, standardizer) for t in (0, .25, .5, .75, 1)]
        expected_start = standardizer.decode(noise) if mode == "one_sided" else source
        for key in target:
            torch.testing.assert_close(states[0][key], expected_start[key])
            torch.testing.assert_close(states[-1][key], target[key])
            if any(not torch.isfinite(state[key]).all() for state in states):
                raise ValueError(f"Non-finite {key} values in {mode} path")


def main(argv=None):
    args = parse_args(argv)
    loaded = load_scene(args.config, args.scene_name, args.split, args.gin_param, args.noise_seed, args.noise_scale)
    source, target, source_flow, target_flow, noise, standardizer, metadata = loaded
    validate_paths(source, target, source_flow, target_flow, noise, standardizer, metadata["noise_scale"])
    displacement = torch.linalg.vector_norm(target["means"] - source["means"], dim=-1)
    print(
        f"Loaded {metadata['scene_name']} ({metadata['split']}) with {len(displacement):,} identity-paired Gaussians; "
        f"{metadata['source_resolution']}→{metadata['target_resolution']}, mean displacement={displacement.mean():.6g}, "
        f"noise_seed={metadata['noise_seed']}, noise_scale={metadata['noise_scale']}", flush=True,
    )
    if args.dry_run:
        return
    launch_viewer(source, target, source_flow, target_flow, noise, standardizer, metadata, args)


if __name__ == "__main__":
    main()

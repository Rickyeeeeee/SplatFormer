#!/usr/bin/env python3
"""Inspect seeded noise-to-native-HR-Gaussian interpolation in Viser.

Example:
  python scripts/visualize_noise_to_gs.py \
    --config configs/dataset/objaverse-sr.gin \
    --scene_name 3e288ee8aced4a0797e66d53536112b1

The saved JSON preset describes standardized noise controls for a future
training sampler. The current noise-to-Gaussian trainer does not load it.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import gin
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.GS_SR import SplatFactoSRDataset
from sr import noise_to_gs
from utils.gaussian_viewer import GsplatBackgroundRenderer, _rgb_from_gaussians, _validate_gsplat_available
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS
from utils.spatial_coordinates import morton_order


def default_settings():
    return {key: {"strength": 1.0, "clip": None} for key in STATISTIC_KEYS}


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(STATISTIC_KEYS):
        raise ValueError(f"Preset must specify exactly these attributes: {', '.join(STATISTIC_KEYS)}")
    validated = {}
    for key, value in settings.items():
        if not isinstance(value, dict) or set(value) != {"strength", "clip"}:
            raise ValueError(f"Preset attribute {key} needs strength and clip")
        strength, clip = value["strength"], value["clip"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
            raise ValueError(f"Preset {key}.strength must be finite and nonnegative")
        if clip is not None and (isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0):
            raise ValueError(f"Preset {key}.clip must be null or finite and positive")
        validated[key] = {"strength": float(strength), "clip": None if clip is None else float(clip)}
    return validated


def load_preset(path):
    with open(path) as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Noise preset must have version 1")
    return validate_settings(document.get("attributes"))


def save_preset(path, settings):
    settings = validate_settings(settings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"version": 1, "attributes": settings}, handle, indent=2)
        handle.write("\n")


def adjusted_noise(base_noise, settings):
    """Clip each fixed standard-normal draw before applying its strength."""
    settings = validate_settings(settings)
    return {key: (value if settings[key]["clip"] is None else value.clamp(-settings[key]["clip"], settings[key]["clip"]))
            * settings[key]["strength"] for key, value in base_noise.items()}


def analytic_state(target, target_flow, base_noise, standardizer, time_value, settings):
    if not 0 <= time_value <= 1:
        raise ValueError("Interpolation time must be in [0, 1]")
    if time_value == 1:
        return target
    noise = adjusted_noise(base_noise, settings)
    query, _ = noise_to_gs.construct_path(target_flow, time_value, noise)
    return standardizer.decode(query)


def diagnostics(current, target, time_value):
    """Summarize physical positions and renderer-facing scale and opacity."""
    current_pos, target_pos = current["means"], target["means"]
    target_min, target_max = target_pos.amin(0), target_pos.amax(0)
    outside = ((current_pos < target_min) | (current_pos > target_max)).any(-1).sum().item()
    current_scale, target_scale = current["scales"].exp().flatten(), target["scales"].exp().flatten()
    current_opacity, target_opacity = current["opacities"].sigmoid().flatten(), target["opacities"].sigmoid().flatten()
    current_quat = torch.linalg.vector_norm(current["quats"], dim=-1)
    target_quat = torch.linalg.vector_norm(target["quats"], dim=-1)

    def bounds(values):
        return ", ".join(f"{item:.3g}" for item in values.tolist())

    def quantiles(values):
        return ", ".join(f"{item:.3g}" for item in torch.quantile(values, torch.tensor([0.05, 0.5, 0.95, 1.0])).tolist())

    return (f"**t={time_value:.2f} · {len(current_pos):,} Gaussians**  \n"
            f"Positions outside target box: **{outside:,} / {len(current_pos):,}**  \n"
            f"Position min: [{bounds(current_pos.amin(0))}] · target [{bounds(target_min)}]  \n"
            f"Position max: [{bounds(current_pos.amax(0))}] · target [{bounds(target_max)}]  \n"
            f"Rendered scale (p05, median, p95, max): **{quantiles(current_scale)}** · target {quantiles(target_scale)}  \n"
            f"Opacity (p05, median, p95, max): **{quantiles(current_opacity)}** · target {quantiles(target_opacity)}  \n"
            f"Quaternion norm (p05, median, p95, max): **{quantiles(current_quat)}** · target {quantiles(target_quat)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", required=True, help="Dataset Gin file; repeat to compose configs.")
    parser.add_argument("--gin_param", action="append", default=[])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--source_gs_root", type=Path, help="Dataset root containing source-resolution GS checkpoints.")
    parser.add_argument("--target_gs_root", type=Path, help="Dataset root containing target-resolution GS checkpoints.")
    parser.add_argument("--statistics_path", type=Path, default=Path("/project/ricky/splatformer-sr-data-scaled/gs_statistics.json"))
    parser.add_argument("--noise_seed", type=int, default=0, help="Evaluation seed; actual seed adds scene index.")
    parser.add_argument("--coupling", choices=("random", "morton"), default="random")
    parser.add_argument("--preset", type=Path, help="Previously saved noise settings JSON.")
    parser.add_argument("--export_settings", type=Path, default=Path("noise_to_gs_preset.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--point_size", type=float, default=0.003)
    parser.add_argument("--initial_view_mode", choices=("pointcloud", "gsplat"), default="pointcloud")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_interval", type=float, default=0.05)
    parser.add_argument("--near_plane", type=float, default=0.01)
    parser.add_argument("--far_plane", type=float, default=100.0)
    parser.add_argument("--radius_clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--rasterize_mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    numeric = (args.point_size, args.render_interval, args.near_plane, args.far_plane, args.radius_clip, args.eps2d, *args.background)
    if not np.isfinite(numeric).all() or args.point_size <= 0 or args.render_interval < 0 or args.render_height <= 0:
        parser.error("Rendering settings must be finite; point size and height must be positive")
    if args.near_plane <= 0 or args.far_plane <= args.near_plane or args.radius_clip < 0 or args.eps2d <= 0:
        parser.error("Invalid clipping planes, radius clip, or eps2d")
    if min(args.background) < 0 or max(args.background) > 1:
        parser.error("Background components must be in [0, 1]")
    return args


def load_scene(args):
    gin.clear_config()
    for path in args.config:
        gin.parse_config_file(str(path), skip_unknown=True)
    if args.gin_param:
        gin.parse_config("\n".join(args.gin_param))
    dataset = SplatFactoSRDataset.from_gin_scope(
        f"{args.split}_dataset", max_gs_num=1000, load_src_gs=False,
        load_tgt_gs=True, load_src_images=False, load_tgt_images=False,
        split_across_gpus=False,
    )
    scene_idx = dataset.scene_index(args.scene_name)
    # Image manifests and GS checkpoints may live in separate dataset trees.
    for resolution, root in ((dataset.src_resolution, args.source_gs_root), (dataset.tgt_resolution, args.target_gs_root)):
        if root is not None:
            dataset.folders[scene_idx]["resolution_paths"][resolution]["gsplat_dir"] = str(
                root / f"{args.split}-set" / dataset.dataset_name / str(resolution) / "gsplat" / args.scene_name
            )
    scene = dataset.load_scene(scene_idx, sample_views=False)
    target = {key: value.float().cpu() for key, value in scene["data"][dataset.tgt_resolution]["gs_params"].items()}
    count = len(target["means"])
    if not 1 <= count <= 1000:
        raise ValueError(f"Expected 1 to 1000 native HR Gaussians, got {count}")
    standardizer = GaussianStandardizer(args.statistics_path)
    # Degree-zero MCMC checkpoints have no higher-order SH coefficients.
    if target["features_rest"].shape[1:] == (0, 3):
        for statistics in (standardizer.means, standardizer.variances, standardizer.stds):
            statistics["features_rest"] = torch.empty((0, 3), dtype=torch.float32)
    if args.coupling == "morton":
        order = morton_order(target["means"])
        target = {key: value[order] for key, value in target.items()}
    target_flow = standardizer.encode(target)
    # Match evaluation's device-specific seeded draw, then keep viewer state on CPU.
    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed + scene_idx)
    base_noise = {key: value.cpu() for key, value in
                  noise_to_gs.order_noise(noise_to_gs.sample_noise(count, standardizer, args.device, generator),
                                          args.coupling).items()}
    return target, target_flow, base_noise, standardizer, scene_idx


class NoiseViewer:
    def __init__(self, args, target, target_flow, base_noise, standardizer, settings, scene_idx):
        import viser

        self.args, self.target, self.target_flow = args, target, target_flow
        self.base_noise, self.standardizer = base_noise, standardizer
        self.settings, self.scene_idx = settings, scene_idx
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+z")
        self.state = {"view_mode": args.initial_view_mode}
        self.current = analytic_state(target, target_flow, base_noise, standardizer, 0.0, settings)
        self.current_handle = self.target_handle = None
        render_cfg = {"device": args.device, "height": args.render_height, "interval": args.render_interval,
                      "near_plane": args.near_plane, "far_plane": args.far_plane, "radius_clip": args.radius_clip,
                      "eps2d": args.eps2d, "rasterize_mode": args.rasterize_mode,
                      "background": np.asarray(args.background, dtype=np.float32)}
        self.renderer = GsplatBackgroundRenderer(self.server, self.current, self.state, render_cfg)
        gui = self.server.gui
        gui.add_markdown(f"**{args.scene_name}** · {len(target['means']):,} native HR Gaussians · seed {args.noise_seed + scene_idx}")
        self.time = gui.add_slider("Interpolation time", 0.0, 1.0, 0.01, 0.0)
        self.view_mode = gui.add_dropdown("Rendering", ("Point cloud", "GSplat background"),
                                          initial_value="Point cloud" if args.initial_view_mode == "pointcloud" else "GSplat background")
        self.show_target = gui.add_checkbox("Show target points", initial_value=False)
        self.point_size = gui.add_number("Point size", initial_value=args.point_size, min=1e-6, step=0.001)
        gui.add_markdown("Noise controls act on one fixed standardized draw. Clip 0 disables clipping; strength 1 preserves training noise.")
        self.controls = {}
        with gui.add_folder("Noise by attribute"):
            for key in STATISTIC_KEYS:
                with gui.add_folder(key):
                    strength = gui.add_number("Strength", initial_value=settings[key]["strength"], min=0.0, step=0.05)
                    clip = gui.add_number("Clip ±σ (0 = off)", initial_value=settings[key]["clip"] or 0.0, min=0.0, step=0.1)
                    self.controls[key] = (strength, clip)
        self.save_button = gui.add_button("Save preset")
        self.status = gui.add_markdown(f"Preset destination: `{args.export_settings}`")
        self.stats = gui.add_markdown("")
        self.time.on_update(self.refresh)
        self.show_target.on_update(self.refresh_points)
        self.point_size.on_update(self.refresh_points)
        self.view_mode.on_update(self.change_view_mode)
        for strength, clip in self.controls.values():
            strength.on_update(self.refresh)
            clip.on_update(self.refresh)
        self.save_button.on_click(self.save)
        self.server.on_client_connect(self.on_client_connect)
        self.refresh(None)

    def settings_from_controls(self):
        return {key: {"strength": float(strength.value), "clip": float(clip.value) or None}
                for key, (strength, clip) in self.controls.items()}

    def refresh_points(self, _event):
        point_mode = self.state["view_mode"] == "pointcloud"
        if self.current_handle is not None:
            self.current_handle.remove()
        if self.target_handle is not None:
            self.target_handle.remove()
        indices = np.arange(len(self.current["means"]))
        self.current_handle = self.server.scene.add_point_cloud(
            "/current", points=self.current["means"].numpy(), colors=_rgb_from_gaussians(self.current, indices),
            point_size=float(self.point_size.value), point_shape="circle", visible=point_mode,
        )
        self.target_handle = self.server.scene.add_point_cloud(
            "/target", points=self.target["means"].numpy(), colors=np.tile((60, 230, 90), (len(indices), 1)),
            point_size=float(self.point_size.value), point_shape="circle", visible=point_mode and self.show_target.value,
        )

    def refresh(self, _event):
        self.settings = self.settings_from_controls()
        self.current = analytic_state(self.target, self.target_flow, self.base_noise, self.standardizer,
                                      float(self.time.value), self.settings)
        self.stats.content = diagnostics(self.current, self.target, float(self.time.value))
        self.renderer.update_gaussians(self.current)
        self.refresh_points(None)
        if self.state["view_mode"] == "gsplat":
            self.renderer.render_all()

    def change_view_mode(self, _event):
        requested = "gsplat" if self.view_mode.value == "GSplat background" else "pointcloud"
        if requested == "gsplat":
            try:
                _validate_gsplat_available(self.args.device)
            except RuntimeError as error:
                self.status.content = f"GSplat unavailable: {error}"
                self.view_mode.value = "Point cloud"
                return
        self.state["view_mode"] = requested
        if requested == "pointcloud":
            self.renderer.clear_all()
        self.refresh(None)

    def save(self, _event):
        try:
            save_preset(self.args.export_settings, self.settings_from_controls())
        except OSError as error:
            self.status.content = f"Could not save preset: {error}"
            return
        self.status.content = f"Saved `{self.args.export_settings}`. Current trainer does not load this preset."

    def on_client_connect(self, client):
        @client.camera.on_update
        def _(_camera):
            self.renderer.render_client(client)
        if self.state["view_mode"] == "gsplat":
            self.renderer.render_client(client, force=True)


def main(argv=None):
    args = parse_args(argv)
    settings = load_preset(args.preset) if args.preset else default_settings()
    target, target_flow, base_noise, standardizer, scene_idx = load_scene(args)
    start = analytic_state(target, target_flow, base_noise, standardizer, 0.0, settings)
    end = analytic_state(target, target_flow, base_noise, standardizer, 1.0, settings)
    for key in STATISTIC_KEYS:
        torch.testing.assert_close(end[key], target[key])
        if not torch.isfinite(start[key]).all():
            raise ValueError(f"Non-finite decoded noise attribute: {key}")
    print(diagnostics(start, target, 0.0).replace("  \n", "\n"), flush=True)
    if args.dry_run:
        return
    if args.initial_view_mode == "gsplat":
        _validate_gsplat_available(args.device)
    viewer = NoiseViewer(args, target, target_flow, base_noise, standardizer, settings, scene_idx)
    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.server.stop()


if __name__ == "__main__":
    main()

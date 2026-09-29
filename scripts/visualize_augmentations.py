#!/usr/bin/env python3
"""Compare Gaussian augmentations and DiPT/EDiPT serialized attention patches.

Example:
  python scripts/visualize_augmentations.py --backend edipt \
    --config configs/dataset/objaverse-sr.gin \
    --scene_name 3e288ee8aced4a0797e66d53536112b1

The default model Gin file is loaded before supplied configurations. No trained
checkpoint is needed. Point-cloud mode is the default; GSplat uses --device.
"""

import argparse
import importlib
from pathlib import Path
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gin
import numpy as np
import torch

from sr import interpolants
from utils.augmentation_viewer import (AugmentationComparison, compose_comparison, flow_matching,
                                       model_settings, training_augmentation)
from utils.data_augmentation import GAUSSIAN_PARAMETERS
from utils.gaussian_viewer import GsplatBackgroundRenderer, _rgb_from_gaussians, _validate_gsplat_available
from utils.gs_normalization import GaussianStandardizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("dipt", "edipt"), default="dipt")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--gin_param", action="append", default=[])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0, help="Independent augmentation, display, and order RNG streams.")
    parser.add_argument("--noise_seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--sample_count", type=int, default=25_000, help="Displayed points only; 0 displays all.")
    parser.add_argument("--point_size", type=float, default=.003)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--initial_view_mode", choices=("pointcloud", "gsplat"), default="pointcloud")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_interval", type=float, default=.05)
    parser.add_argument("--near_plane", type=float, default=.01)
    parser.add_argument("--far_plane", type=float, default=100.)
    parser.add_argument("--radius_clip", type=float, default=0.)
    parser.add_argument("--eps2d", type=float, default=.3)
    parser.add_argument("--rasterize_mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--background", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    numeric = (args.point_size, args.render_interval, args.near_plane, args.far_plane, args.radius_clip, args.eps2d, *args.background)
    if not np.isfinite(numeric).all():
        parser.error("Rendering settings must be finite")
    if args.sample_count < 0 or args.point_size <= 0 or args.render_height <= 0 or args.render_interval < 0:
        parser.error("Invalid display count, point size, render height, or interval")
    if args.near_plane <= 0 or args.far_plane <= args.near_plane or args.radius_clip < 0 or args.eps2d <= 0:
        parser.error("Invalid clipping planes, radius clip, or eps2d")
    if min(args.background) < 0 or max(args.background) > 1:
        parser.error("Background components must be in [0, 1]")
    if args.noise_scale is not None and (not np.isfinite(args.noise_scale) or args.noise_scale < 0):
        parser.error("Noise scale must be finite and nonnegative")
    return args


def load_scene(args):
    """Load experiment settings and immutable, identity-paired physical endpoints."""
    from dataset.GS_SR import SplatFactoSRDataset

    # Register real constructors for Gin, but never instantiate a model.
    module = "equivariant_gaussian_dipt_predictor" if args.backend == "edipt" else "diffusion_gaussian_predictor"
    importlib.import_module(f"models.{module}")
    gin.external_configurable(torch.nn.Identity)
    gin.clear_config()
    representation = "unit_unstandardized" if args.backend == "edipt" else "raw_standardized"
    gin.parse_config(f"flow_matching.quaternion_representation = '{representation}'")
    gin.parse_config_file(str(REPO_ROOT / f"configs/model/{args.backend}_gaussian.gin"), skip_unknown=True)
    for path in args.config:
        gin.parse_config_file(str(path), skip_unknown=True)
    if args.gin_param:
        gin.parse_config("\n".join(args.gin_param))
    flow_cfg, augmentation_cfg, settings = flow_matching(), training_augmentation(), model_settings(args.backend)
    standardizer = GaussianStandardizer(flow_cfg["gs_statistics_path"], flow_cfg["normalization_variance_floor"], flow_cfg["quaternion_representation"])
    dataset = SplatFactoSRDataset.from_gin_scope(f"{args.split}_dataset", load_src_gs=True, load_tgt_gs=True,
                                               load_src_images=False, load_tgt_images=False, split_across_gpus=False)
    scene_idx = dataset.scene_index(args.scene_name)
    scene = dataset.load_scene(scene_idx, sample_views=False, fit_alignment="fit_lr_to_hr")
    source = scene["data"][dataset.src_resolution]["gs_params"]
    target = scene["fit_lr_to_hr"]["tgt_gs"]
    if any(key not in target or source[key].shape != target[key].shape for key in source):
        raise ValueError("fit_lr_to_hr endpoints must be identity-paired")
    seed = flow_cfg["eval_noise_seed"] if args.noise_seed is None else args.noise_seed
    if args.noise_scale is not None:
        flow_cfg["flow_noise_std"] = args.noise_scale
    comparison = AugmentationComparison(source, target, standardizer, args.backend, settings, seed + scene_idx, args.seed, flow_cfg["interpolant_type"])
    metadata = {"scene_name": scene["scene_name"], "scene_idx": scene_idx, "split": args.split,
                "source_resolution": dataset.src_resolution, "target_resolution": dataset.tgt_resolution,
                "noise_seed": seed + scene_idx}
    return comparison, metadata, flow_cfg, augmentation_cfg


class AugmentationViewer:
    """One synchronized comparison canvas with cached state and patch controls."""

    def __init__(self, comparison, metadata, flow_cfg, augmentation_cfg, args):
        import viser

        self.comparison, self.args = comparison, args
        self.lock = threading.RLock()
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+z")
        self.state = {"view_mode": args.initial_view_mode}
        self.handles = []
        self.defaults = {**augmentation_cfg, "seed": args.seed}
        self.permutation = np.random.default_rng(args.seed).permutation(len(comparison.source["means"]))
        bounds = torch.cat((comparison.source["means"], comparison.target["means"]))
        self.diagonal = max(float((bounds.amax(0) - bounds.amin(0)).norm()), .01)
        gui = self.server.gui
        gui.add_markdown(f"**{metadata['scene_name']}** · {args.backend.upper()} · {len(bounds) // 2:,} Gaussians\n\n"
                         f"{metadata['source_resolution']} → {metadata['target_resolution']} · {metadata['split']} · noise seed {metadata['noise_seed']}")
        with gui.add_folder("State"):
            self.selection = gui.add_dropdown("Gaussian state", ("Source", "Fitted target", "Interpolant"), initial_value="Source")
            self.mode = gui.add_dropdown("Interpolant", interpolants.MODES, initial_value=flow_cfg["interpolant_type"])
            self.time = gui.add_slider("Time", 0., 1., .01, 0.)
            self.noise_scale = gui.add_number("Euclidean noise", initial_value=flow_cfg["flow_noise_std"], min=0., step=.05)
            self.rotation_noise = gui.add_number("Rotation noise", initial_value=flow_cfg["rotation_noise_std"], min=0., step=.05, visible=args.backend == "edipt")
        with gui.add_folder("Augmentation"):
            self.control_mode = gui.add_dropdown("Controls", ("Training samples", "Manual"))
            self.rotate = gui.add_checkbox("Rotation", initial_value=augmentation_cfg["random_rotate"])
            self.jitter = gui.add_checkbox("Source jitter", initial_value=augmentation_cfg["random_jitter"])
            self.seed = gui.add_number("Augmentation seed", initial_value=args.seed, step=1)
            self.resample = gui.add_button("Resample augmentation")
            self.reset = gui.add_button("Reset augmentation")
            self.applied = gui.add_markdown("")
        with gui.add_folder("Rotation"):
            self.rotation_mode = gui.add_dropdown("Rotation mode", ("full", "gravity_consistent"), initial_value=augmentation_cfg["rotation_mode"])
            self.pivot = gui.add_vector3("Pivot", initial_value=augmentation_cfg["rotation_pivot"], step=.01)
            self.bound = gui.add_checkbox("Bound sampled angle", initial_value=augmentation_cfg["rotation_max_degrees"] is not None)
            self.max_angle = gui.add_slider("Maximum degrees", 0., 180., .1, float(augmentation_cfg["rotation_max_degrees"] or 0.))
            self.axis = gui.add_vector3("Manual axis", initial_value=(0., 0., 1.), step=.1)
            self.angle = gui.add_slider("Manual degrees", -180., 180., .1, 0.)
        with gui.add_folder("Jitter"):
            gui.add_markdown("Relative to each channel's population deviation. Training samples use these as maxima; manual controls use exact levels.")
            self.levels = {key: gui.add_number(key, initial_value=augmentation_cfg["jitter_max_levels"].get(key, 0.), min=0., step=.01)
                           for key in GAUSSIAN_PARAMETERS}
        with gui.add_folder("Attention patches"):
            fixed_label = "unaugmented" if args.backend == "edipt" else "unaugmented (viewer comparison)"
            self.reference_labels = {"augmented": "augmented", fixed_label: "unaugmented"}
            self.reference = gui.add_dropdown("Serialization reference", tuple(self.reference_labels),
                                              initial_value=next(key for key, value in self.reference_labels.items() if value == augmentation_cfg["serialization_reference"]))
            self.block = gui.add_dropdown("Block", tuple(str(i + 1) for i in range(comparison.settings["depth"])))
            self.order = gui.add_dropdown("Serialization order", ("Block default", *comparison.settings["order"]))
            self.patch_size = gui.add_slider("Attention patch size", 1, max(4096, max(comparison.settings["patch_sizes"])), 1, comparison.settings["patch_sizes"][0])
            self.patch_info = gui.add_markdown("")
        with gui.add_folder("Display"):
            self.view_mode = gui.add_dropdown("Rendering", ("Point cloud", "GSplat background"),
                                              initial_value="Point cloud" if args.initial_view_mode == "pointcloud" else "GSplat background")
            self.color_mode = gui.add_dropdown("Colors", ("Patches", "Gaussian RGB"))
            counts = sorted({min(len(self.permutation), n) for n in (1000, 5000, 25000, 50000, len(self.permutation), args.sample_count or len(self.permutation))})
            self.count_labels = {f"{n:,}": n for n in counts}
            self.count = gui.add_dropdown("Displayed points per variant", tuple(self.count_labels), initial_value=f"{min(args.sample_count or len(self.permutation), len(self.permutation)):,}")
            self.point_size = gui.add_slider("Point size", 1e-6, max(self.diagonal * .05, args.point_size * 10), 1e-5, args.point_size)
            self.reset_camera = gui.add_button("Frame comparison")
        self.status = gui.add_markdown("")
        render_cfg = {"device": args.device, "height": args.render_height, "interval": args.render_interval,
                      "near_plane": args.near_plane, "far_plane": args.far_plane, "radius_clip": args.radius_clip,
                      "eps2d": args.eps2d, "rasterize_mode": args.rasterize_mode, "background": np.asarray(args.background)}
        self.renderer = GsplatBackgroundRenderer(self.server, comparison.source, self.state, render_cfg)
        for control in (self.control_mode, self.rotate, self.jitter, self.seed, self.rotation_mode, self.pivot,
                        self.bound, self.max_angle, self.axis, self.angle, *self.levels.values()):
            control.on_update(self.on_augmentation)
        for control in (self.selection, self.time, self.noise_scale, self.rotation_noise):
            control.on_update(self.on_state)
        for control in (self.reference, self.order, self.patch_size, self.color_mode, self.count, self.point_size):
            control.on_update(self.on_display)
        self.mode.on_update(self.on_mode)
        self.block.on_update(self.on_block)
        self.view_mode.on_update(self.on_view_mode)
        self.resample.on_click(self.on_resample)
        self.reset.on_click(self.on_reset)
        self.reset_camera.on_click(self.on_reset_camera)
        self.server.on_client_connect(self.on_client_connect)
        self.on_augmentation(None)

    def augmentation_settings(self):
        return {"control_mode": self.control_mode.value, "random_rotate": self.rotate.value,
                "random_jitter": self.jitter.value and self.mode.value != "one_sided",
                "rotation_mode": self.rotation_mode.value, "rotation_pivot": self.pivot.value,
                "rotation_max_degrees": self.max_angle.value if self.bound.value else None,
                "rotation_axis": self.axis.value, "rotation_degrees": self.angle.value,
                "jitter_max_levels": {key: control.value for key, control in self.levels.items()}}

    def update_control_visibility(self):
        flow = self.selection.value == "Interpolant"
        manual = self.control_mode.value == "Manual"
        for control in (self.mode, self.time, self.noise_scale):
            control.visible = flow
        self.rotation_noise.visible = flow and self.args.backend == "edipt"
        self.jitter.disabled = self.mode.value == "one_sided"
        for control in self.levels.values():
            control.disabled = self.jitter.disabled or not self.jitter.value
        self.axis.visible = manual and self.rotation_mode.value == "full"
        self.angle.visible = manual
        self.bound.visible = not manual
        self.max_angle.visible = not manual and self.bound.value
        for control in (self.count, self.point_size):
            control.visible = self.state["view_mode"] == "pointcloud"

    def refresh(self, recompute_state=True):
        self.update_control_visibility()
        if recompute_state:
            self.gaussians = self.comparison.states(self.selection.value, self.mode.value, self.time.value,
                                                     self.noise_scale.value, self.rotation_noise.value)
        layouts = self.comparison.layouts(self.selection.value, self.mode.value, self.time.value,
                                           self.reference_labels[self.reference.value], int(self.block.value) - 1,
                                           self.order.value, int(self.patch_size.value))
        displayed, combined = compose_comparison(*self.gaussians, layouts, self.color_mode.value, self.pivot.value)
        self.renderer.update_gaussians(combined)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        indices = self.permutation[:self.count_labels[self.count.value]]
        for name, gaussians in zip(("Baseline", "Current"), displayed):
            points = gaussians["means"]
            self.handles.append(self.server.scene.add_point_cloud(f"/{name}/points", points=points[indices].numpy(),
                               colors=_rgb_from_gaussians(gaussians, indices), point_size=self.point_size.value,
                               point_shape="circle", visible=self.state["view_mode"] == "pointcloud"))
            label_position = ((points.amin(0) + points.amax(0)) / 2).numpy()
            label_position[2] = float(points[:, 2].max()) + .05 * self.diagonal
            self.handles.append(self.server.scene.add_label(f"/{name}/label", name, position=label_position))
        self.display_bounds = (combined["means"].amin(0), combined["means"].amax(0))
        row = self.comparison.order_rows[(int(self.block.value) - 1) % len(self.comparison.order_rows)]
        actual_order = self.comparison.settings["order"][row] if self.order.value == "Block default" else self.order.value
        layout = layouts[0]
        self.patch_info.content = (f"**{actual_order}** · {layout['patch_count']:,} patches per variant · effective size {layout['effective_size']}\n\n"
                                   f"Borrowed padding: {layout['borrowed']} · masked padding: {layout['masked']}\n\n"
                                   "Colors identify primary query patches. Borrowed padding also participates in the final DiPT patch.")
        self.status.content = ""
        if self.state["view_mode"] == "gsplat":
            self.renderer.render_all()

    def on_augmentation(self, event):
        with self.lock:
            try:
                rotation, levels = self.comparison.augment(self.augmentation_settings(), int(self.seed.value))
                self.applied.content = (f"Rotation wxyz: `{np.round(rotation.numpy(), 5).tolist()}`\n\n"
                                        + " · ".join(f"{key}: {level:.4g}" for key, level in levels.items()))
                self.refresh()
            except ValueError as error:
                self.status.content = f"**Cannot apply settings:** {error}"

    def on_state(self, event):
        with self.lock:
            self.refresh()

    def on_mode(self, event):
        with self.lock:
            self.comparison.set_mode(self.mode.value)
            self.on_augmentation(event)

    def on_display(self, event):
        with self.lock:
            self.refresh(recompute_state=False)

    def on_block(self, event):
        with self.lock:
            self.patch_size.value = self.comparison.settings["patch_sizes"][int(self.block.value) - 1]
            self.refresh(recompute_state=False)

    def on_view_mode(self, event):
        with self.lock:
            requested = "gsplat" if self.view_mode.value == "GSplat background" else "pointcloud"
            if requested == "gsplat":
                try:
                    _validate_gsplat_available(self.args.device)
                except RuntimeError as error:
                    self.view_mode.value = "Point cloud"
                    self.status.content = str(error)
                    return
            self.state["view_mode"] = requested
            if requested == "pointcloud":
                self.renderer.clear_all()
            self.refresh(recompute_state=False)

    def on_resample(self, event):
        self.seed.value = int(self.seed.value) + 1

    def on_reset(self, event):
        with self.lock, self.server.atomic():
            self.control_mode.value = "Training samples"
            self.rotate.value = self.defaults["random_rotate"]
            self.jitter.value = self.defaults["random_jitter"]
            self.seed.value = self.defaults["seed"]
            self.rotation_mode.value = self.defaults["rotation_mode"]
            self.pivot.value = self.defaults["rotation_pivot"]
            self.bound.value = self.defaults["rotation_max_degrees"] is not None
            self.max_angle.value = float(self.defaults["rotation_max_degrees"] or 0.)
            self.axis.value, self.angle.value = (0., 0., 1.), 0.
            for key, control in self.levels.items():
                control.value = self.defaults["jitter_max_levels"].get(key, 0.)
            self.on_augmentation(event)

    def frame_client(self, client):
        low, high = self.display_bounds
        center = ((low + high) / 2).numpy()
        distance = max(float((high - low).norm()), .1)
        # Moving the native Viser camera also translates its look-at point.
        client.camera.position = center + np.array([0., -1.3 * distance, .7 * distance])
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)

    def on_reset_camera(self, event):
        for client in self.server.get_clients().values():
            self.frame_client(client)

    def on_client_connect(self, client):
        self.frame_client(client)
        client.camera.on_update(self.on_camera)
        if self.state["view_mode"] == "pointcloud":
            self.renderer.clear_all()

    def on_camera(self, event):
        self.renderer.render_client(event.client)


def validate_comparison(comparison, flow_cfg, augmentation_cfg, seed):
    """Exercise augmentations, all analytic modes, and both references without a server."""
    for rotate, jitter in ((False, False), (True, False), (False, True), (True, True)):
        settings = {**augmentation_cfg, "random_rotate": rotate, "random_jitter": jitter}
        comparison.augment(settings, seed)
        for mode in interpolants.MODES:
            if mode == "one_sided" and jitter:
                continue
            for t in (0., .25, .5, .75, 1.):
                states = comparison.states("Interpolant", mode, t, flow_cfg["flow_noise_std"], flow_cfg["rotation_noise_std"])
                if any(not torch.isfinite(value).all() for state in states for value in state.values()):
                    raise ValueError(f"Non-finite state for {mode}, t={t}")
                for reference in ("augmented", "unaugmented"):
                    for block, size in enumerate(comparison.settings["patch_sizes"]):
                        layouts = comparison.layouts("Interpolant", mode, t, reference, block, "Block default", size)
                        if any(len(layout["ids"]) != len(comparison.source["means"]) for layout in layouts):
                            raise ValueError("Patch ownership must cover every Gaussian")


def main(argv=None):
    args = parse_args(argv)
    comparison, metadata, flow_cfg, augmentation_cfg = load_scene(args)
    if args.dry_run:
        validate_comparison(comparison, flow_cfg, augmentation_cfg, args.seed)
        print(f"Validated {args.backend.upper()} augmentations, paths, and patches for {metadata['scene_name']} ({len(comparison.source['means']):,} Gaussians).", flush=True)
        return
    if args.initial_view_mode == "gsplat":
        _validate_gsplat_available(args.device)
    viewer = AugmentationViewer(comparison, metadata, flow_cfg, augmentation_cfg, args)
    print(f"Augmentation viewer running at http://{args.host}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.server.stop()


if __name__ == "__main__":
    main()

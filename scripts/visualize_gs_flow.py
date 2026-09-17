#!/usr/bin/env python3
"""Inspect recorded GSFM sampling states, velocities, and voxel crossings.

Example (choose an available GPU before launching):
  CUDA_VISIBLE_DEVICES=0 python scripts/visualize_gs_flow.py \\
    --config /project2/ricky/outputs/0915-gpu7/objaverse_sr_gsfm_32to128_fit_lr_to_hr_fm-only/config.gin \\
    --checkpoint /project2/ricky/outputs/0915-gpu7/objaverse_sr_gsfm_32to128_fit_lr_to_hr_fm-only/checkpoints/model_00019999.pth \\
    --scene_name 3e288ee8aced4a0797e66d53536112b1 --flow_steps 10

In Sampling, adjust Grid resolution R (cell width 1/R) and Attention window,
then click Apply settings / Resample. This recomputes sampling and all layer/patch
structures in memory; the saved config/checkpoint are unchanged.

Use --split train for training scenes, repeated --gin_param for dataset path
changes, and --dry_run to record/validate a trajectory without starting Viser.
The viewer reads existing experiments and never writes to their directories.

Choose Structure > Transformer encoder to inspect the embedding, pooling, and
attention-block point sets. Coordinates are captured during the first Euler
forward pass and remain fixed across sampling times. Pooling averages coordinates
and reduces features; attention blocks preserve the point count. Splat mode
shows Gaussian renders without point-cloud overlays. Both renderers support all
structure views; gsplat maps pooled groups back to original Gaussian colors,
with all non-color Gaussian parameters preserved.

Choose Structure > Attention patches for the actual serialized windows in every
encoder/decoder attention layer, captured on the first sampling forward pass.
Patch -1 colors all points by their retained-output window; selecting a patch
includes points repeated into that window by padding. Window sizes come from the
model (normally 1024 with Flash Attention), including smaller deep-layer windows.
"""

import argparse
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.flow_viewer import (
    COMPARISONS, DERIVED_ATTRIBUTES, GS_KEYS, LatestRequests, TrajectoryInspection,
    attention_patch_colors, add_segments, as_numpy, heatmap_colors, load_experiment, metric_summary,
    metric_values, pick_gaussian, record_trajectory, render_gaussians,
    selected_voxel_path, sampling_geometry, structure_colors, voxel_edges, voxel_indices,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--gin_param", action="append", default=[])
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_interval", type=float, default=.05)
    parser.add_argument("--sample_count", type=int, default=25000, help="Point overlay limit; 0 shows all.")
    parser.add_argument("--arrow_count", type=int, default=500, help="Maximum update arrows; 0 hides them.")
    parser.add_argument("--voxel_count", type=int, default=2000, help="Stable Gaussian sample used for occupied voxel boxes; 0 hides the overview.")
    parser.add_argument("--point_size", type=float, default=.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_view_mode", choices=("gsplat", "pointcloud"), default="gsplat")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.flow_steps < 1 or args.render_height < 1 or args.point_size <= 0:
        parser.error("flow_steps, render_height, and point_size must be positive")
    if min(args.sample_count, args.arrow_count, args.voxel_count, args.render_interval) < 0:
        parser.error("Display limits and render_interval must be nonnegative")
    return args


class FlowViewer:
    def __init__(self, model, source, target, metadata, trajectory, args):
        import viser

        self.model, self.source, self.target = model, source, target
        self.metadata, self.args = metadata, args
        self.inspection = TrajectoryInspection(trajectory, target, args.seed)
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()
        self.requests = LatestRequests()
        self.stop_event = threading.Event()
        self.handles = {}
        self.gpu_cache_key, self.gpu_cache = None, None
        self.active_colors = None
        self.structure_gs = None
        self.frame_revision = 0
        self.sampling = False
        self.component_count = None
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.scene = getattr(self.server, "scene", self.server)
        self.gui = getattr(self.server, "gui", self.server)
        self._build_gui()
        self.server.on_client_connect(self._connect)
        self.pointer_active = False
        self.refresh()
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

    def control(self, kind, *args, **kwargs):
        method = getattr(self.gui, f"add_{kind}", None)
        if method is None:
            method = getattr(self.server, f"add_gui_{kind}")
        return method(*args, **kwargs)

    def _build_gui(self):
        self.status = self.control("markdown", f"{self.inspection.trajectory.count:,} Gaussians · Ready")
        self.sampling_folder = self.control("folder", "Sampling")
        with self.sampling_folder:
            self.step_slider = self.control("slider", "Recorded state", min=0, max=self.inspection.trajectory.steps, step=1, initial_value=0)
            self.step_slider.on_update(self.refresh)
            self.control("button", "Previous").on_click(self._previous)
            self.control("button", "Next").on_click(self._next)
            self.step_count = self.control("number", "Euler steps for resampling", initial_value=self.inspection.trajectory.steps, min=1, step=1)
            self.grid_resolution = self.control("number", "Grid resolution R (cell = 1/R)", initial_value=float(self.inspection.trajectory.grid_resolution), min=1., step=1.)
            windows = [getattr(module, "patch_size_max", module.patch_size) for module in self.model.modules() if hasattr(module, "get_padding_and_inverse")]
            self.attention_window = self.control("number", "Attention window (points)", initial_value=max(windows, default=1024), min=1, step=1)
            self.resample_button = self.control("button", "Apply settings / Resample")
            self.resample_button.on_click(self._resample)
            self.time_info = self.control("markdown", "")
        self.view_mode = self.control("dropdown", "Viewer mode", options=("gsplat", "pointcloud"), initial_value=self.args.initial_view_mode)
        self.control("button", "Reset camera").on_click(self._reset_cameras)
        self.point_folder = self.control("folder", "Structure")
        with self.point_folder:
            self.cloud_content = self.control("dropdown", "Structure", options=("Sampling trajectory", "Transformer encoder", "Attention patches"), initial_value="Sampling trajectory")
            labels = self._layer_labels()
            self.layer_select = self.control("dropdown", "Encoder layer", options=labels, initial_value=labels[0])
            attention_labels = self._attention_labels()
            self.attention_select = self.control("dropdown", "Attention layer", options=attention_labels, initial_value=attention_labels[0])
            self.patch_select = self.control("number", "Patch (-1 = all)", initial_value=-1, min=-1, step=1)
            self.layer_info = self.control("markdown", "")
            self.point_size = self.control("slider", "Point size", min=.0001, max=.02, step=.0001, initial_value=min(max(self.args.point_size, .0001), .02))
            self.source_overlay = self.control("checkbox", "Source points (cyan)", initial_value=False)
            self.target_overlay = self.control("checkbox", "Fitted-target points (green)", initial_value=False)
        self.render_folder = self.control("folder", "Gaussian appearance", expand_by_default=False)
        with self.render_folder:
            self.reference = self.control("dropdown", "Render selection", options=("Current state", "Source", "Fitted target"), initial_value="Current state")
            self.color_mode = self.control("dropdown", "Color mode", options=("Normal render", "Diagnostic heatmap"), initial_value="Normal render")
        self.feature_folder = self.control("folder", "Feature diagnostics", expand_by_default=False)
        with self.feature_folder:
            self.comparison = self.control("dropdown", "Compare sampling state", options=COMPARISONS, initial_value=COMPARISONS[0])
            self.attribute = self.control("dropdown", "Attribute", options=GS_KEYS + DERIVED_ATTRIBUTES, initial_value="means")
            self.component = self.control("dropdown", "Component", options=("Magnitude", "0", "1", "2"), initial_value="Magnitude")
            self.component_count = 3
            self.autoscale = self.control("checkbox", "Autoscale colors per step", initial_value=False)
            self.show_arrows = self.control("checkbox", "Show positional update arrows", initial_value=False)
            self.arrow_mode = self.control("dropdown", "Arrow quantity", options=("Actual step displacement", "Predicted velocity"), initial_value="Actual step displacement")
            self.velocity_scale = self.control("slider", "Velocity arrow display scale (time units)", min=0., max=1., step=.001, initial_value=1. / self.inspection.trajectory.steps)
            self.legend = self.control("markdown", "")
        self.voxel_folder = self.control("folder", "Voxel inspection")
        with self.voxel_folder:
            self.grid_info = self.control("markdown", "")
            self.source_voxels = self.control("checkbox", "Source occupied cells (cyan)", initial_value=True)
            self.current_voxels = self.control("checkbox", "Current occupied cells (orange)", initial_value=False)
            self.local_voxels = self.control("checkbox", "Selected Gaussian crossed cells", initial_value=False)
            self.show_trajectory = self.control("checkbox", "Selected Gaussian full trajectory", initial_value=False)
            self.voxel_width = self.control("slider", "Edge width (cell units)", min=.05, max=1., step=.05, initial_value=.25)
            self.control("button", "Zoom to selected cells").on_click(self._focus_voxels)
            with self.control("folder", "Voxel statistics", expand_by_default=False):
                self.voxel_info = self.control("markdown", "")
        self.selected_folder = self.control("folder", "Selected Gaussian", expand_by_default=False)
        with self.selected_folder:
            self.selected_index = self.control("number", "Gaussian index", initial_value=0, min=0, max=self.inspection.trajectory.count - 1, step=1)
            self.pick_enabled = self.control("checkbox", "Pick displayed points by clicking", initial_value=False)
            self.control("button", "Focus selected Gaussian").on_click(self._focus)
            with self.control("folder", "Raw attributes", expand_by_default=False):
                self.selected_info = self.control("markdown", "")
        for control in (self.view_mode, self.reference, self.color_mode, self.source_overlay, self.target_overlay, self.point_size, self.comparison, self.attribute, self.component, self.autoscale, self.show_arrows, self.arrow_mode, self.velocity_scale, self.source_voxels, self.current_voxels, self.local_voxels, self.show_trajectory, self.selected_index, self.pick_enabled, self.voxel_width, self.cloud_content, self.layer_select, self.attention_select, self.patch_select):
            control.on_update(self.refresh)

    def _layer_labels(self):
        layers = self.inspection.trajectory.encoder_layers
        return tuple(f"{i}: {layer['name']} · {len(layer['coord']):,} points" for i, layer in enumerate(layers)) or ("No encoder structure recorded",)

    def _attention_labels(self):
        layers = self.inspection.trajectory.attention_layers
        return tuple(f"{i}: {layer['name']} · K={layer['patch_size']}" for i, layer in enumerate(layers)) or ("No attention patches recorded",)

    def _show_attention(self):
        layers = self.inspection.trajectory.attention_layers
        if not layers:
            self.layer_info.content = "No attention patches recorded for this model."
            return
        layer = layers[int(self.attention_select.value.split(":", 1)[0])]
        points = as_numpy(layer["coord"])
        boundaries, tokens = layer["boundaries"], layer["token_indices"]
        count = len(boundaries) - 1
        selected = min(max(int(self.patch_select.value), -1), count - 1)
        if selected >= 0:
            indices = np.unique(tokens[boundaries[selected]:boundaries[selected + 1]])
            colors = attention_patch_colors(np.full(len(indices), selected))
            info = f"Patch {selected}: **{len(indices):,} points / {boundaries[selected + 1] - boundaries[selected]} tokens**"
        else:
            indices = np.arange(len(points))
            colors = attention_patch_colors(layer["patch_ids"])
            info = f"**{len(points):,} points · {count} patches · window {layer['patch_size']}**"
        if self.view_mode.value == "gsplat":
            full_colors = attention_patch_colors(layer["patch_ids"])
            if selected >= 0:
                full_colors[indices] = colors
            self.active_colors = structure_colors(layer, full_colors, indices if selected >= 0 else None)
            self.structure_gs = self.inspection.render_state()
        else:
            self._points("/attention", points[indices], colors, True)
        duplicates = len(tokens) - len(points)
        self.layer_info.content = f"{info} · order slot {layer['order_index']}\n\n{duplicates} padding repeats. All: retained-output patch colors. Select a patch to include its padding points."

    def _show_encoder(self):
        layers = self.inspection.trajectory.encoder_layers
        if not layers:
            self.layer_info.content = "No encoder structure recorded for this model."
            return
        layer = layers[int(self.layer_select.value.split(":", 1)[0])]
        points = as_numpy(layer["coord"])
        # Show every pooled point so the displayed count reflects actual model structure.
        colors = attention_patch_colors(np.arange(len(points)))
        if self.view_mode.value == "gsplat":
            self.active_colors = structure_colors(layer, colors)
            self.structure_gs = self.inspection.render_state()
        else:
            self._points("/encoder", points, colors, True)
        self.layer_info.content = f"**{len(points):,} points · {layer['channels']} channels** · fixed source coordinates"

    def _previous(self, _event):
        self.step_slider.value = max(0, int(self.step_slider.value) - 1)

    def _next(self, _event):
        self.step_slider.value = min(self.inspection.trajectory.steps, int(self.step_slider.value) + 1)

    def _remove_handle(self, name):
        old = self.handles.pop(name, None)
        if old is not None:
            old.remove()

    def _points(self, name, points, colors, visible):
        if not visible:
            self._remove_handle(name)
            return
        # Remove before reusing a scene name; late removal would hide the new node.
        self._remove_handle(name)
        self.handles[name] = self.scene.add_point_cloud(name, points=np.asarray(points, dtype=np.float32), colors=np.asarray(colors, dtype=np.uint8), point_size=float(self.point_size.value), point_shape="circle")

    def _lines(self, name, segments, color, visible, width=None):
        self._remove_handle(name)
        if visible and len(segments):
            width = width or float(self.voxel_width.value) / self.inspection.trajectory.grid_resolution
            self.handles[name] = add_segments(self.scene, name, segments, color, width, pixel_width=max(1., width * self.inspection.trajectory.grid_resolution * 8))

    def _sync_components(self):
        attribute = self.attribute.value
        width = int(np.prod(self.source[attribute].shape[1:])) if attribute in GS_KEYS else (3 if attribute == "linear scales" else 1)
        if width != self.component_count:
            old, selected = self.component, self.component.value
            options = ("Magnitude",) + tuple(str(i) for i in range(width))
            with self.feature_folder:
                self.component = self.control("dropdown", "Component", options=options, initial_value=selected if selected in options else "Magnitude", order=old.order)
            old.remove()
            self.component.on_update(self.refresh)
            self.component_count = width

    def refresh(self, _event=None):
        with self.lock:
            resolution = self.inspection.trajectory.grid_resolution
            self.grid_info.content = f"Active R={resolution:g} · cell={1 / resolution:.5g} · cyan: source · orange: current"
            point_mode = self.view_mode.value == "pointcloud"
            hierarchy = self.cloud_content.value != "Sampling trajectory"
            attention = hierarchy and self.cloud_content.value == "Attention patches"
            self.point_folder.visible = True
            self.point_size.visible = point_mode
            self.layer_select.visible = hierarchy and not attention
            self.attention_select.visible = self.patch_select.visible = attention
            self.layer_info.visible = hierarchy
            self.source_overlay.visible = self.target_overlay.visible = point_mode and not hierarchy
            self.render_folder.visible = not hierarchy or not point_mode
            self.color_mode.visible = not hierarchy
            self.feature_folder.visible = not hierarchy
            self.voxel_folder.visible = self.selected_folder.visible = point_mode and not hierarchy
            self.show_arrows.visible = self.arrow_mode.visible = self.velocity_scale.visible = point_mode and not hierarchy
            self.step_slider.disabled = False
            self._sync_picking()
            inspection = self.inspection
            inspection.step = min(max(int(self.step_slider.value), 0), inspection.trajectory.steps)
            inspection.selected = min(max(int(self.selected_index.value), 0), inspection.trajectory.count - 1)
            inspection.reference = self.reference.value
            self.structure_gs = None
            if hierarchy:
                self.active_colors = None
                for name in list(self.handles):
                    self._remove_handle(name)
                if attention:
                    self._show_attention()
                else:
                    self._show_encoder()
                if not point_mode and self.structure_gs is not None:
                    self.layer_info.content += "\n\nColors mapped to original Gaussians; all other attributes unchanged."
                self.frame_revision += 1
                for client in self.server.get_clients().values():
                    self.requests.request(client)
                return
            self._remove_handle("/encoder")
            self._remove_handle("/attention")
            self._sync_components()
            step, trajectory = inspection.step, inspection.trajectory
            comparison, attribute, component = self.comparison.value, self.attribute.value, self.component.value
            if comparison == "Predicted velocity" and attribute in DERIVED_ATTRIBUTES:
                self.attribute.value = "means"
                return
            values = metric_values(trajectory, step, comparison, attribute, component, self.target)
            bounds = inspection.color_range(comparison, attribute, component, self.autoscale.value)
            colors = heatmap_colors(values, bounds, trajectory.count)
            self.active_colors = colors if self.color_mode.value == "Diagnostic heatmap" else None
            self.frame_revision += 1
            displayed = inspection.indices(self.args.sample_count)
            rendered = inspection.render_state()
            rgb = np.clip(as_numpy(rendered["features_dc"]) * .28209479177387814 + .5, 0, 1) * 255
            if rendered["features_rest"].numel() == 0:
                rgb = torch.sigmoid(rendered["features_dc"]).numpy() * 255
            self._points("/state", as_numpy(rendered["means"])[displayed], (colors if self.active_colors is not None else rgb)[displayed], self.view_mode.value == "pointcloud")
            for name, gs, color, visible in (("/source_points", self.source, (40, 210, 230), self.source_overlay.value), ("/target_points", self.target, (60, 230, 90), self.target_overlay.value)):
                self._points(name, as_numpy(gs["means"])[displayed], np.tile(color, (len(displayed), 1)), visible and point_mode)
            self.time_info.content = f"**{step}/{trajectory.steps} · t={trajectory.times[step]:.3f}**"
            scope = "step" if self.autoscale.value else "trajectory"
            self.legend.content = f"Blue {bounds[0]:.4g} → red {bounds[1]:.4g} ({scope})\n\n{metric_summary(values)}"
            if point_mode:
                self._update_arrows()
                self._update_voxels()
                self._update_selected()
            else:
                for name in list(self.handles):
                    self._remove_handle(name)
            for client in self.server.get_clients().values():
                self.requests.request(client)

    def _update_arrows(self):
        trajectory, step = self.inspection.trajectory, self.inspection.step
        segments = np.empty((0, 2, 3))
        if step and self.show_arrows.value and self.args.arrow_count:
            indices = self.inspection.indices(self.args.arrow_count)
            start = as_numpy(trajectory.states[step - 1]["means"])[indices]
            delta = as_numpy(trajectory.states[step]["means"])[indices] - start
            if self.arrow_mode.value == "Predicted velocity":
                delta = as_numpy(trajectory.velocities[step - 1]["means"])[indices] * self.velocity_scale.value
            end = start + delta
            # Add two arrowhead wings in a plane orthogonal to each update direction.
            length = np.linalg.norm(delta, axis=-1, keepdims=True)
            direction = delta / np.maximum(length, 1e-12)
            axis = np.eye(3)[np.argmin(np.abs(direction), axis=-1)]
            side = np.cross(direction, axis)
            side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-12)
            wing1, wing2 = end - .18 * delta + .07 * length * side, end - .18 * delta - .07 * length * side
            segments = np.concatenate([np.stack([start, end], axis=1), np.stack([end, wing1], axis=1), np.stack([end, wing2], axis=1)])
        self._lines("/updates", segments, (240, 160, 40), bool(len(segments)))

    def _update_voxels(self):
        inspection = self.inspection
        trajectory, step, index = inspection.trajectory, inspection.step, inspection.selected
        resolution = trajectory.grid_resolution
        source_idx = voxel_indices(self.source["means"], resolution)
        current_idx = voxel_indices(trajectory.states[step]["means"], resolution)
        # Overview limits apply only to graphics; summaries include every Gaussian.
        subset = inspection.indices(self.args.voxel_count) if self.args.voxel_count else np.empty(0, dtype=int)
        for name, indices, color, visible in (("/source_cells", source_idx, (30, 170, 210), self.source_voxels.value), ("/current_cells", current_idx, (240, 140, 40), self.current_voxels.value)):
            edges = voxel_edges(indices[subset], resolution) if visible else []
            self._lines(name, edges, color, visible)
        outside = np.any(source_idx != current_idx, axis=-1)
        distance = np.linalg.norm(as_numpy(trajectory.states[step]["means"]) - as_numpy(self.source["means"]), axis=-1) * resolution
        previous = trajectory.states[max(0, step - 1)]["means"]
        step_distance = np.linalg.norm(as_numpy(trajectory.states[step]["means"]) - as_numpy(previous), axis=-1) * resolution
        points, incoming, visited = selected_voxel_path(trajectory, index, step)
        visible = self.local_voxels.value
        # Exact counts remain uncapped; only extremely long local graphics are sampled.
        shown = visited[np.linspace(0, len(visited) - 1, min(len(visited), 5000), dtype=int)]
        self._lines("/visited_cells", voxel_edges(shown, resolution) if visible else [], (150, 100, 230), visible)
        for name, point, color in (("source", points[0], (30, 200, 240)), ("previous", points[max(0, step - 1)], (250, 220, 30)), ("current", points[step], (245, 80, 40))):
            self._lines(f"/selected_{name}_cell", voxel_edges(voxel_indices(point[None], resolution), resolution) if visible else [], color, visible, float(self.voxel_width.value) / resolution)
        self._lines("/selected_trajectory", np.stack([points[:-1], points[1:]], axis=1), (240, 70, 180), self.show_trajectory.value, .08 / resolution)
        selected_position = as_numpy(inspection.render_state()["means"])[index:index+1]
        self._points("/selection", selected_position, np.array([[255, 255, 255]]), self.show_trajectory.value or visible or self.pick_enabled.value)
        net = int(np.max(np.abs(current_idx[index] - source_idx[index])))
        self.voxel_info.content = f"Outside source cell: **{outside.mean():.1%}**\n\nSource distance (cells): {metric_summary(distance)}\n\nStep distance (cells): {metric_summary(step_distance)}\n\nSelected #{index}: **{len(incoming)}** incoming visits · **{len(visited)}** unique cells · **{net}** net axis shift\n\nSource `{source_idx[index].tolist()}` → current `{current_idx[index].tolist()}`\n\nShown: {len(shown):,} path cells; {len(subset):,} overview samples."


    def _update_selected(self):
        inspection = self.inspection
        trajectory, step, index = inspection.trajectory, inspection.step, inspection.selected
        rows = [f"**Gaussian #{index}** (index is stable across steps)", "", "| Attribute | Current raw value | Incoming velocity | Applied change | Target |", "|---|---|---|---|---|"]
        for key in GS_KEYS:
            value = as_numpy(trajectory.states[step][key])[index].reshape(-1)
            target = as_numpy(self.target[key])[index].reshape(-1)
            v = as_numpy(trajectory.velocities[step - 1][key])[index].reshape(-1) if step else None
            delta = value - as_numpy(trajectory.states[step - 1][key])[index].reshape(-1) if step else None
            formatted = ["N/A" if a is None else np.array2string(a, precision=4, separator=", ", max_line_width=1000) for a in (value, v, delta, target)]
            rows.append(f"| {key} | `{'` | `'.join(formatted)}` |")
        rows.extend(["", "Velocity is the raw network output at the previous time. Quaternion values are preserved exactly; angular differences are available separately."])
        self.selected_info.content = "\n".join(rows)

    def _connect(self, client):
        client.camera.on_update(self._camera_update)
        self._set_camera(client, selected=False)
        self.requests.request(client)

    def _camera_update(self, event):
        # Legacy camera callbacks receive CameraHandle; modern events may carry a client.
        client = getattr(event, "client", None)
        if client is not None:
            self.requests.request(client)
        else:
            for connected in self.server.get_clients().values():
                self.requests.request(connected)

    def _set_camera(self, client, selected):
        with self.lock:
            trajectory = self.inspection.trajectory
            if selected:
                points = np.stack([as_numpy(s["means"])[self.inspection.selected] for s in trajectory.states])
                center = as_numpy(trajectory.states[self.inspection.step]["means"])[self.inspection.selected]
                radius = max(float(np.linalg.norm(np.ptp(points, axis=0))), 8 / trajectory.grid_resolution)
            else:
                points = np.concatenate([as_numpy(self.source["means"]), as_numpy(self.target["means"])])
                center = (points.min(0) + points.max(0)) / 2
                radius = max(float(np.linalg.norm(np.ptp(points, axis=0))), .1)
            client.camera.look_at = tuple(center)
            client.camera.up_direction = (0, 0, 1)
            client.camera.position = tuple(center + np.array([1, -1, .65]) * radius)

    def _reset_cameras(self, _event):
        for client in self.server.get_clients().values():
            self._set_camera(client, selected=False)

    def _focus(self, _event):
        for client in self.server.get_clients().values():
            self._set_camera(client, selected=True)

    def _sync_picking(self):
        # Scene-wide pointer capture disables orbit dragging in legacy Viser.
        enabled = bool(self.pick_enabled.value and self.view_mode.value == "pointcloud" and self.cloud_content.value == "Sampling trajectory")
        if enabled == self.pointer_active:
            return
        if enabled:
            if hasattr(self.scene, "on_pointer_event"):
                self.scene.on_pointer_event("click")(self._pick)
            elif hasattr(self.server, "on_scene_pointer"):
                self.server.on_scene_pointer("click")(self._pick)
            else:
                self.server.on_scene_click(self._pick)
        elif hasattr(self.scene, "remove_pointer_callback"):
            self.scene.remove_pointer_callback()
        else:
            self.server.remove_scene_pointer_callback()
        self.pointer_active = enabled

    def _focus_voxels(self, _event):
        self.local_voxels.value = True
        self.refresh()
        center = as_numpy(self.inspection.trajectory.states[self.inspection.step]["means"])[self.inspection.selected]
        radius = 8 / self.inspection.trajectory.grid_resolution
        for client in self.server.get_clients().values():
            client.camera.look_at = tuple(center)
            client.camera.position = tuple(center + np.array([1, -1, .65]) * radius)
            self.requests.request(client)

    def _pick(self, event):
        if not self.pick_enabled.value or self.view_mode.value != "pointcloud" or event.ray_origin is None:
            return
        with self.lock:
            displayed = self.inspection.indices(self.args.sample_count)
            points = as_numpy(self.inspection.render_state()["means"])[displayed]
            picked = pick_gaussian(points, event.ray_origin, event.ray_direction, max(float(self.point_size.value) * 2, 1 / self.inspection.trajectory.grid_resolution))
            if picked is not None:
                self.selected_index.value = int(displayed[picked])
        # Picking is one-shot: immediately restore normal orbit/pan controls.
        self.pick_enabled.value = False
        self._sync_picking()

    def _render_loop(self):
        while not self.stop_event.is_set():
            self.requests.event.wait(.1)
            # A short coalescing window prevents a render for each individual slider event.
            if self.stop_event.wait(self.args.render_interval):
                return
            for client, _generation in self.requests.take():
                try:
                    with self.lock:
                        key = (self.inspection.revision, self.inspection.step, self.inspection.reference)
                        state = self.inspection.render_state()
                        structure = getattr(self, "structure_gs", None)
                        structure_mode = getattr(self, "cloud_content", None) is not None and self.cloud_content.value != "Sampling trajectory"
                        if structure is not None:
                            state = structure
                        colors = self.active_colors
                        mode = self.view_mode.value
                        frame_revision = self.frame_revision
                    if mode == "pointcloud" or (structure_mode and structure is None):
                        image = np.zeros((2, 2, 3), dtype=np.uint8)
                    else:
                        with self.gpu_lock:
                            if key != self.gpu_cache_key:
                                self.gpu_cache = {k: v.to(self.args.device) for k, v in state.items()}
                                self.gpu_cache_key = key
                            image = render_gaussians(client.camera, self.gpu_cache, self.args.render_height, self.args.device, colors)
                    # Publish completed camera frames during motion, but never obsolete UI states.
                    with self.lock:
                        if frame_revision == self.frame_revision:
                            client.set_background_image(image, format="jpeg", jpeg_quality=85)
                except Exception as error:
                    self.status.content = f"Render failed: `{error}`"
                    traceback.print_exc()

    def _progress(self, done, total):
        self.status.content = f"Sampling **{done}/{total}**. The previous trajectory remains available."

    def _resample(self, _event):
        with self.lock:
            if self.sampling:
                return
            steps = int(self.step_count.value)
            resolution, window = float(self.grid_resolution.value), int(self.attention_window.value)
            if steps < 1 or resolution < 1 or not np.isfinite(resolution) or window < 1:
                self.status.content = "Steps, grid resolution, and attention window must be positive."
                return
            self.sampling = True
            self.resample_button.disabled = True
            self.status.content = f"Preparing {steps} Euler steps…"
        self.sampling_thread = threading.Thread(target=self._resample_worker, args=(steps, resolution, window), daemon=True)
        self.sampling_thread.start()

    def _resample_worker(self, steps, resolution, window):
        try:
            with self.gpu_lock:
                self.gpu_cache, self.gpu_cache_key = None, None
                with sampling_geometry(self.model, resolution, window):
                    trajectory = record_trajectory(self.model, self.source, self.metadata["scene_idx"], steps, self.args.device, self._progress)
            with self.lock:
                self.inspection.replace(trajectory)
                old_layer = self.layer_select
                labels = self._layer_labels()
                with self.point_folder:
                    self.layer_select = self.control("dropdown", "Encoder layer", options=labels, initial_value=labels[0], order=old_layer.order)
                old_layer.remove()
                self.layer_select.on_update(self.refresh)
                old_attention = self.attention_select
                attention_labels = self._attention_labels()
                with self.point_folder:
                    self.attention_select = self.control("dropdown", "Attention layer", options=attention_labels, initial_value=attention_labels[0], order=old_attention.order)
                old_attention.remove()
                self.attention_select.on_update(self.refresh)
                # Legacy slider handles cannot update min/max; recreate in the same position.
                old = self.step_slider
                with self.sampling_folder:
                    self.step_slider = self.control("slider", "Recorded state", min=0, max=steps, step=1, initial_value=0, order=old.order)
                old.remove()
                self.step_slider.on_update(self.refresh)
                self.status.content = f"Ready: R={resolution:g} · window={window} · {steps} steps."
                self.refresh()
        except Exception as error:
            self.status.content = f"Resampling failed: `{error}`"
            traceback.print_exc()
        finally:
            with self.lock:
                self.sampling = False
                self.resample_button.disabled = False

    def close(self):
        self.stop_event.set()
        self.requests.event.set()
        self.render_thread.join(timeout=5)
        if hasattr(self.server, "stop"):
            self.server.stop()


def main(argv=None):
    args = parse_args(argv)
    print(f"Loading {args.scene_name} and {args.checkpoint}", flush=True)
    model, source, target, metadata = load_experiment(args.config, args.checkpoint, args.scene_name, args.split, args.gin_param, args.seed)
    trajectory = record_trajectory(model, source, metadata["scene_idx"], args.flow_steps, args.device, lambda done, total: print(f"Sampling {done}/{total}", flush=True))
    print(f"Recorded {trajectory.steps + 1} states of {trajectory.count:,} Gaussians; grid R={trajectory.grid_resolution:g}", flush=True)
    for step in range(trajectory.steps + 1):
        values = metric_values(trajectory, step, "Source difference", "distance in voxels", target=target)
        print(f"state={step} t={trajectory.times[step]:.6g} source displacement in voxels: {metric_summary(values)}", flush=True)
    if args.dry_run:
        return
    viewer = FlowViewer(model, source, target, metadata, trajectory, args)
    print(f"Flow viewer: http://{args.host}:{args.port} (model weights offloaded between sampling runs)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()

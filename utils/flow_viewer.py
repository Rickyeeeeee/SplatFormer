"""Trajectory recording and diagnostics for the GSFM sampling viewer."""

from dataclasses import dataclass, field
from functools import partial
from contextlib import contextmanager
import threading
import colorsys

import numpy as np
import torch

# Register sampling dependencies before saved Gin configuration is finalized.
from sr.flow import apply_feature_update


GS_KEYS = ("means", "features_dc", "features_rest", "opacities", "scales", "quats")
COMPARISONS = ("Previous-step change", "Predicted velocity", "Source difference", "Target error")
DERIVED_ATTRIBUTES = ("opacity probability", "linear scales", "rotation degrees", "distance in voxels", "voxel index distance", "outside source voxel")


@dataclass
class FlowTrajectory:
    states: list
    velocities: list
    times: np.ndarray
    grid_resolution: float
    encoder_layers: list = field(default_factory=list)
    attention_layers: list = field(default_factory=list)

    @property
    def steps(self):
        return len(self.velocities)

    @property
    def count(self):
        return len(self.states[0]["means"])


class EncoderStructureRecorder:
    """Snapshot actual encoder coordinates without retaining features or changing outputs."""

    def __init__(self, model):
        self.layers, self.attention_layers, self.handles = [], [], []
        backbone = getattr(getattr(model, "backbone", None), "backbone", None)
        if backbone is None or not hasattr(backbone, "enc"):
            return
        modules = [("Embedding", backbone.embedding)]
        for stage_name, stage in backbone.enc.named_children():
            modules.extend((f"{stage_name}/{name}", module) for name, module in stage.named_children())
        for name, module in modules:
            self.handles.append(module.register_forward_hook(partial(self.capture, name)))

        # Capture every encoder and decoder attention layer after its real padding is built.
        for name, module in backbone.named_modules():
            if hasattr(module, "get_padding_and_inverse") and hasattr(module, "order_index"):
                self.handles.append(module.register_forward_hook(partial(self.capture_attention, name)))

    def capture_attention(self, name, module, inputs, output):
        point = output
        order = point["serialized_order"][module.order_index][point["pad"]]
        inverse = point["unpad"][point["serialized_inverse"][module.order_index]]
        layer = attention_patch_structure(order, inverse, point["cu_seqlens_key"])
        layer.update(source_to_point=source_point_membership(point), name=name, coord=point["coord"].detach().cpu().clone(), patch_size=int(module.patch_size), order_index=int(module.order_index))
        self.attention_layers.append(layer)

    def capture(self, name, module, inputs, output):
        self.layers.append({"source_to_point": source_point_membership(output), "name": name, "coord": output["coord"].detach().cpu().clone(), "grid_coord": output["grid_coord"].detach().cpu().clone(), "channels": int(output["feat"].shape[-1])})

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def source_point_membership(point):
    """Compose real pooling inverses from original Gaussians to this layer's points."""
    inverses = []
    while "pooling_parent" in point:
        inverses.append(as_numpy(point["pooling_inverse"]).astype(np.int64, copy=True))
        point = point["pooling_parent"]
    membership = np.arange(len(point["coord"]))
    for inverse in reversed(inverses):
        membership = inverse[membership]
    return membership


def structure_colors(layer, point_colors, selected_points=None):
    """Lift layer colors onto original Gaussian indices without changing geometry."""
    membership = layer["source_to_point"]
    colors = np.asarray(point_colors, dtype=np.uint8)[membership].copy()
    if selected_points is not None:
        colors[~np.isin(membership, selected_points)] = 65
    return colors


def attention_patch_structure(order, inverse, boundaries):
    """Keep padded token membership and the window whose output each point retains."""
    order = as_numpy(order).astype(np.int64, copy=True)
    inverse = as_numpy(inverse).astype(np.int64, copy=True)
    boundaries = as_numpy(boundaries).astype(np.int64, copy=True)
    token_patches = np.repeat(np.arange(len(boundaries) - 1), np.diff(boundaries))
    return {"token_indices": order, "boundaries": boundaries, "patch_ids": token_patches[inverse]}


def attention_patch_colors(patch_ids):
    """Stable categorical colors, independent of spatial distance and patch size."""
    patch_ids = np.asarray(patch_ids, dtype=np.int64)
    palette = np.array([colorsys.hsv_to_rgb((i * .618033988749895) % 1., .72, .95) for i in range(int(patch_ids.max()) + 1)])
    return (palette[patch_ids] * 255).astype(np.uint8)


def cpu_snapshot(gaussians):
    return {key: value.detach().cpu().clone() for key, value in gaussians.items()}


@contextmanager
def sampling_geometry(model, resolution, window):
    """Apply viewer-only inference overrides, restoring model settings even on failure."""
    if resolution <= 0 or not np.isfinite(resolution) or window < 1:
        raise ValueError("Grid resolution and attention window must be positive")
    original_resolution = model.grid_resolution
    attention = [(module, module.patch_size, getattr(module, "patch_size_max", None)) for module in model.modules() if hasattr(module, "get_padding_and_inverse")]
    try:
        model.grid_resolution = resolution
        for module, _, maximum in attention:
            module.patch_size = window
            if maximum is not None:
                module.patch_size_max = window
        yield
    finally:
        model.grid_resolution = original_resolution
        for module, size, maximum in attention:
            module.patch_size = size
            if maximum is not None:
                module.patch_size_max = maximum


def record_trajectory(model, source, scene_idx, steps, device="cuda", progress=None):
    """Mirror sr.flow.sample_flow_model, retaining actual states and incoming velocities."""
    if steps < 1:
        raise ValueError("Sampling steps must be positive")
    device = torch.device(device)
    source_device = {key: value.to(device).clone() for key, value in source.items()}
    state = {key: value.clone() for key, value in source_device.items()}
    states, velocities = [cpu_snapshot(state)], []
    encoder = EncoderStructureRecorder(model)
    try:
        model.to(device).eval()
        # Explicitly disable autocast even if the caller has an active AMP context.
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for step in range(steps):
                t = torch.full((1,), float(step) / steps, device=device)
                velocity = model(batch_flow_gs=[state], batch_scene_idx=[scene_idx], batch_reference_means=[source_device["means"]], t=t)[0]
                if step == 0:
                    encoder.close()
                missing = set(GS_KEYS) - set(velocity)
                if missing:
                    raise ValueError(f"Missing predicted velocity attributes: {sorted(missing)}")
                for key in GS_KEYS:
                    if velocity[key].shape != state[key].shape:
                        raise ValueError(f"Velocity shape mismatch for {key}: {velocity[key].shape} vs {state[key].shape}")
                state = {key: apply_feature_update(model, key, state[key], velocity[key], 1.0 / steps) for key in GS_KEYS}
                if any(not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError(f"Non-finite Gaussian state after sampling step {step + 1}")
                velocities.append(cpu_snapshot(velocity))
                states.append(cpu_snapshot(state))
                if progress is not None:
                    progress(step + 1, steps)
    finally:
        encoder.close()
        model.cpu()
    del source_device, state, velocity, t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return FlowTrajectory(states, velocities, np.arange(steps + 1, dtype=np.float64) / steps, float(model.grid_resolution), encoder.layers, encoder.attention_layers)


def load_experiment(config, checkpoint, scene_name, split="test", bindings=(), seed=42):
    """Load inference settings without importing trainer entrypoints or initializing DDP."""
    import gin
    from dataset.GS_SR import SplatFactoSRDataset
    from models.feature_flow_predictor import GSFlowPredictor

    # These bindings describe training only; unknown model/dataset bindings still fail.
    ignored = ["training", "flow_matching", "loss_mixing", "feature_mse_loss", "set_seed", "build_optimizer", "build_scheduler", "FeaturePredictor"]
    gin.clear_config()
    gin.parse_config_files_and_bindings([str(config)], list(bindings), skip_unknown=ignored)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = SplatFactoSRDataset.from_gin_scope(f"{split}_dataset", load_src_gs=True, load_tgt_gs=True, load_src_images=False, load_tgt_images=False, split_across_gpus=False)
    scene_idx = dataset.scene_index(scene_name)
    scene = dataset.load_scene(scene_idx, sample_views=False, fit_alignment="fit_lr_to_hr")
    source = scene["data"][dataset.src_resolution]["gs_params"]
    target = scene["fit_lr_to_hr"]["tgt_gs"]
    model = GSFlowPredictor()
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {"scene_name": scene["scene_name"], "scene_idx": scene_idx, "split": split, "source_resolution": dataset.src_resolution, "target_resolution": dataset.tgt_resolution, "checkpoint": str(checkpoint), "config": str(config)}
    return model, cpu_snapshot(source), cpu_snapshot(target), metadata


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def voxel_indices(means, resolution):
    points = as_numpy(means)
    dtype = np.result_type(points.dtype, np.float32)
    return np.floor(points.astype(dtype) * np.asarray(resolution, dtype=dtype)).astype(np.int64)


def comparison_pair(trajectory, step, comparison, target):
    current = trajectory.states[step]
    if comparison in COMPARISONS[:2] and step == 0:
        return None, None
    if comparison == "Previous-step change":
        return current, trajectory.states[step - 1]
    if comparison == "Source difference":
        return current, trajectory.states[0]
    if comparison == "Target error":
        return current, target
    if comparison == "Predicted velocity":
        return trajectory.velocities[step - 1], None
    raise ValueError(f"Unknown comparison: {comparison}")


def metric_values(trajectory, step, comparison, attribute, component="Magnitude", target=None):
    """Return one scalar per Gaussian; None denotes an unavailable incoming step."""
    a, b = comparison_pair(trajectory, step, comparison, target)
    if a is None:
        return None
    if comparison == "Predicted velocity" and attribute not in GS_KEYS:
        raise ValueError("Interpreted differences are unavailable for raw predicted velocity")
    if attribute in GS_KEYS:
        values = as_numpy(a[attribute])
        if b is not None:
            values = values - as_numpy(b[attribute])
    elif attribute == "opacity probability":
        values = torch.sigmoid(a["opacities"]).numpy() - torch.sigmoid(b["opacities"]).numpy()
    elif attribute == "linear scales":
        values = np.exp(as_numpy(a["scales"]).astype(np.float64)) - np.exp(as_numpy(b["scales"]).astype(np.float64))
    elif attribute == "rotation degrees":
        qa, qb = as_numpy(a["quats"]).astype(np.float64), as_numpy(b["quats"]).astype(np.float64)
        qa /= np.maximum(np.linalg.norm(qa, axis=-1, keepdims=True), 1e-12)
        qb /= np.maximum(np.linalg.norm(qb, axis=-1, keepdims=True), 1e-12)
        return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, axis=-1)), 0, 1)))
    elif attribute == "distance in voxels":
        return np.linalg.norm(as_numpy(a["means"]) - as_numpy(b["means"]), axis=-1) * trajectory.grid_resolution
    elif attribute == "voxel index distance":
        return np.max(np.abs(voxel_indices(a["means"], trajectory.grid_resolution) - voxel_indices(b["means"], trajectory.grid_resolution)), axis=-1)
    elif attribute == "outside source voxel":
        return np.any(voxel_indices(trajectory.states[step]["means"], trajectory.grid_resolution) != voxel_indices(trajectory.states[0]["means"], trajectory.grid_resolution), axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown attribute: {attribute}")
    values = values.reshape(trajectory.count, -1)
    return np.linalg.norm(values, axis=-1) if component == "Magnitude" else values[:, int(component)]


def metric_summary(values):
    if values is None:
        return "No previous step"
    return "mean={:.5g}, median={:.5g}, p95={:.5g}, max={:.5g}".format(np.mean(values), np.median(values), np.percentile(values, 95), np.max(values))


@dataclass
class TrajectoryInspection:
    trajectory: FlowTrajectory
    target: dict
    seed: int = 42
    step: int = 0
    selected: int = 0
    reference: str = "Current state"
    revision: int = 0
    ranges: dict = field(default_factory=dict)

    def __post_init__(self):
        self.permutation = np.random.default_rng(self.seed).permutation(self.trajectory.count)

    def indices(self, limit):
        return self.permutation if limit == 0 else self.permutation[:limit]

    def render_state(self):
        if self.reference == "Source":
            return self.trajectory.states[0]
        if self.reference == "Fitted target":
            return self.target
        return self.trajectory.states[self.step]

    def replace(self, trajectory):
        if trajectory.count != self.trajectory.count:
            raise ValueError("Resampling must preserve Gaussian identity/count")
        self.trajectory = trajectory
        self.step = 0
        self.revision += 1
        self.ranges.clear()

    def color_range(self, comparison, attribute, component, autoscale=False):
        key = (comparison, attribute, component)
        if not autoscale and key in self.ranges:
            return self.ranges[key]
        steps = [self.step] if autoscale else range(self.trajectory.steps + 1)
        arrays = [metric_values(self.trajectory, k, comparison, attribute, component, self.target) for k in steps]
        arrays = [v for v in arrays if v is not None]
        if not arrays:
            return (0.0, 1.0)
        values = np.concatenate(arrays)
        if component == "Magnitude" or attribute in DERIVED_ATTRIBUTES[2:]:
            bounds = (0.0, max(float(np.percentile(values, 99)), 1e-12))
        else:
            high = max(float(np.percentile(np.abs(values), 99)), 1e-12)
            bounds = (-high, high)
        if not autoscale:
            self.ranges[key] = bounds
        return bounds


def heatmap_colors(values, bounds, count):
    if values is None:
        return np.full((count, 3), 160, dtype=np.uint8)
    low, high = bounds
    stops = np.array([[49, 54, 149], [116, 173, 209], [240, 240, 225], [253, 174, 97], [215, 48, 39]], dtype=np.float64)
    t = np.clip((values - low) / max(high - low, 1e-12), 0, 1) * (len(stops) - 1)
    left = np.floor(t).astype(np.int64)
    right = np.minimum(left + 1, len(stops) - 1)
    return np.round(stops[left] * (1 - (t - left)[:, None]) + stops[right] * (t - left)[:, None]).astype(np.uint8)


def traversed_voxels(start, end, resolution):
    """Half-open endpoint cells plus cells with positive-length segment intersection.

    Simultaneous face crossings advance all tied axes, excluding corner-only cells.
    Endpoints use the same floor convention as the model, including negative positions.
    """
    start_array, end_array = np.asarray(start), np.asarray(end)
    start_array = start_array.astype(np.result_type(start_array.dtype, np.float32))
    end_array = end_array.astype(np.result_type(end_array.dtype, np.float32))
    start = np.asarray(start_array * np.asarray(resolution, dtype=start_array.dtype), dtype=np.float64)
    end = np.asarray(end_array * np.asarray(resolution, dtype=end_array.dtype), dtype=np.float64)
    current, final = np.floor(start).astype(np.int64), np.floor(end).astype(np.int64)
    cells = [current.copy()]
    direction = end - start
    sign = np.sign(direction).astype(np.int64)
    active = direction != 0
    delta = np.full(3, np.inf)
    delta[active] = 1 / np.abs(direction[active])
    boundary = current + (sign > 0)
    crossing = np.full(3, np.inf)
    crossing[active] = (boundary[active] - start[active]) / direction[active]
    # Each iteration crosses at least one plane; this bound also covers endpoint faces.
    for _ in range(int(np.abs(final - current).sum()) + 3):
        if np.array_equal(current, final):
            break
        next_t = crossing.min()
        if next_t >= 1 or np.isclose(next_t, 1, rtol=0, atol=1e-12):
            if not np.array_equal(cells[-1], final):
                cells.append(final.copy())
            break
        tied = np.isclose(crossing, next_t, rtol=0, atol=1e-12)
        current = current + sign * tied
        crossing[tied] += delta[tied]
        if not np.array_equal(cells[-1], current):
            cells.append(current.copy())
    return np.asarray(cells, dtype=np.int64)


def selected_voxel_path(trajectory, selected, step):
    points = np.stack([as_numpy(state["means"])[selected] for state in trajectory.states])
    segments = [traversed_voxels(points[k], points[k + 1], trajectory.grid_resolution) for k in range(step)]
    visited = np.unique(np.concatenate(segments), axis=0) if segments else voxel_indices(points[:1], trajectory.grid_resolution)
    incoming = segments[-1] if segments else np.empty((0, 3), dtype=np.int64)
    return points, incoming, visited


def voxel_edges(cells, resolution):
    cells = np.unique(np.asarray(cells, dtype=np.int64).reshape(-1, 3), axis=0)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    pairs = np.array([(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1])
    vertices = (cells[:, None, :] + corners[None]) / resolution
    return vertices[:, pairs].reshape(-1, 2, 3).astype(np.float32)


def segment_mesh(segments, width):
    """Batch thin rectangular prisms for legacy Viser without line-segment support."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    direction = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(direction, axis=-1)
    segments, direction, lengths = segments[lengths > 1e-12], direction[lengths > 1e-12], lengths[lengths > 1e-12]
    direction /= lengths[:, None]
    axis = np.eye(3)[np.argmin(np.abs(direction), axis=-1)]
    u = np.cross(direction, axis)
    u *= (width / 2) / np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(direction, u)
    offsets = np.stack([-u - v, -u + v, u + v, u - v], axis=1)
    vertices = (segments[:, :, None, :] + offsets[:, None]).reshape(-1, 3)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    faces = (faces[None] + 8 * np.arange(len(segments))[:, None, None]).reshape(-1, 3)
    return vertices.astype(np.float32), faces.astype(np.uint32)


def add_segments(scene, name, segments, color, width, pixel_width=1.5):
    if hasattr(scene, "add_line_segments"):
        return scene.add_line_segments(name, points=np.asarray(segments, dtype=np.float32), colors=np.asarray(color, dtype=np.uint8), line_width=pixel_width)
    vertices, faces = segment_mesh(segments, width)
    return scene.add_mesh_simple(name, vertices=vertices, faces=faces, color=color, side="double")


def pick_gaussian(points, origin, direction, radius):
    """Pick the closest-to-ray displayed center within a world-space tolerance."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    relative = np.asarray(points) - np.asarray(origin)
    distance = relative @ direction
    perpendicular = np.linalg.norm(relative - distance[:, None] * direction, axis=-1)
    eligible = np.flatnonzero((distance > 0) & (perpendicular <= radius))
    if not len(eligible):
        return None
    return int(eligible[np.lexsort((distance[eligible], perpendicular[eligible]))[0]])


def camera_matrices(camera, height):
    w, x, y, z = np.asarray(camera.wxyz, dtype=np.float64)
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ np.asarray(camera.position)
    width = max(1, round(height * float(camera.aspect)))
    focal = .5 * height / np.tan(.5 * float(camera.fov))
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
    return view, intrinsics, width


def render_gaussians(camera, gaussians, height, device, colors=None, background=(0, 0, 0)):
    from gsplat.rendering import rasterization
    from utils.gs_utils import _prepare_render_inputs

    means, quats, scales, opacities, sh, degree = _prepare_render_inputs(gaussians)
    if colors is not None:
        sh = torch.as_tensor(colors, device=device, dtype=torch.float32) / 255
        degree = None
    view, intrinsics, width = camera_matrices(camera, height)
    with torch.no_grad():
        rgb, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opacities, colors=sh, viewmats=torch.as_tensor(view, device=device)[None], Ks=torch.as_tensor(intrinsics, device=device)[None], width=width, height=height, sh_degree=degree, packed=True, render_mode="RGB", rasterize_mode="classic")
        rgb = rgb + (1 - alpha) * torch.as_tensor(background, device=device, dtype=rgb.dtype).view(1, 1, 1, 3)
    return rgb[0].clamp(0, 1).mul(255).byte().cpu().numpy()


class LatestRequests:
    """Coalesce camera/UI updates to one pending render per connected client."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.generation = 0
        self.latest = {}
        self.event = threading.Event()

    def request(self, client):
        with self.lock:
            self.generation += 1
            self.latest[client.client_id] = self.generation
            self.pending[client.client_id] = (client, self.generation)
            self.event.set()

    def take(self):
        with self.lock:
            items = list(self.pending.values())
            self.pending.clear()
            self.event.clear()
            return items

    def is_current(self, client, generation):
        with self.lock:
            return self.latest.get(client.client_id) == generation

"""Analytic augmentation comparisons and the backbones' serialized patch layouts."""

import math

import gin
import numpy as np
import torch

from sr import edipt_interpolants, interpolants
from utils.data_augmentation import (GAUSSIAN_PARAMETERS, jitter_gaussian_parameter,
                                     jitter_gaussian_parameters, rotate_gaussians,
                                     sample_uniform_rotation_quaternion,
                                     sample_uniform_z_rotation_quaternion, validate_rotation_max_degrees)
from utils.rotation_flow import rotation_exp


@gin.configurable
def flow_matching(flow_steps=10, flow_noise_std=1.0, flow_t_eps=1e-4, loss_type="velocity",
                  interpolant_type="linear", loss_rollout_steps=10, eval_noise_seed=0,
                  fixed_train_noise=False, train_noise_seed=0, normalization_variance_floor=1e-8,
                  quaternion_representation=None, rotation_noise_std=0.3, rotation_loss_weight=1.0,
                  gs_statistics_path="/project/ricky/splatformer-sr-data-scaled/gs_statistics.json"):
    return {"interpolant_type": interpolant_type, "flow_noise_std": flow_noise_std,
            "eval_noise_seed": eval_noise_seed, "rotation_noise_std": rotation_noise_std,
            "quaternion_representation": quaternion_representation,
            "normalization_variance_floor": normalization_variance_floor,
            "gs_statistics_path": gs_statistics_path}


@gin.configurable
def training_augmentation(random_jitter=False, random_rotate=False, jitter_max_levels=None,
                          rotation_pivot=(0.5, 0.5, 0.5), rotation_mode="full", rotation_max_degrees=None,
                          serialization_reference="augmented"):
    return {"random_jitter": random_jitter, "random_rotate": random_rotate,
            "jitter_max_levels": dict.fromkeys(GAUSSIAN_PARAMETERS, .01) if jitter_max_levels is None else dict(jitter_max_levels),
            "rotation_pivot": tuple(rotation_pivot), "rotation_mode": rotation_mode,
            "rotation_max_degrees": rotation_max_degrees, "serialization_reference": serialization_reference}


def model_settings(backend):
    """Read registered Gin constructors without allocating model weights."""
    if backend == "edipt":
        from models.equivariant_gaussian_dipt_predictor import EquivariantGaussianDiPTPredictor as Predictor
        from models.equivariant_gaussian_dipt import EquivariantGaussianDiPT as Backbone
    else:
        from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor as Predictor
        from models.diffusion_gaussian_transformer import DiffusionGaussianTransformer as Backbone
    import inspect

    settings = {}
    for cls, names in ((Predictor, ("grid_resolution", "shift_negative_grid_coords")),
                       (Backbone, ("depth", "order", "patch_size", "shuffle_orders", "enable_flash"))):
        defaults = inspect.signature(cls.__init__).parameters
        for name in names:
            if name not in defaults:
                continue
            try:
                value = gin.query_parameter(f"{cls.__name__}.{name}")
            except ValueError:
                value = defaults[name].default
                if value is inspect.Parameter.empty:
                    raise ValueError(f"Configure {cls.__name__}.{name}") from None
            settings[name] = value
    settings["shift_negative_grid_coords"] = backend == "edipt" or settings.get("shift_negative_grid_coords", False)
    settings["enable_flash"] = settings.get("enable_flash", True)
    settings["order"] = (settings["order"],) if isinstance(settings["order"], str) else tuple(settings["order"])
    sizes = settings.pop("patch_size")
    settings["patch_sizes"] = (sizes,) * settings["depth"] if isinstance(sizes, int) else tuple(sizes)
    if len(settings["patch_sizes"]) != settings["depth"] or any(int(size) != size or size < 1 for size in settings["patch_sizes"]):
        raise ValueError("Expected one positive integer patch size per block")
    return settings


def augment_endpoints(source, target, standardizer, backend, settings, seed):
    """Rotate both prepared endpoints, then jitter only the source, as in training."""
    generator = torch.Generator(device=source["means"].device).manual_seed(int(seed))
    source_out, target_out = source, target
    rotation = source["means"].new_tensor([1., 0., 0., 0.])
    levels = dict.fromkeys(GAUSSIAN_PARAMETERS, 0.)
    manual = settings.get("control_mode", "Training samples") == "Manual"
    if settings["random_rotate"]:
        validate_rotation_max_degrees(settings["rotation_max_degrees"])
        if manual:
            axis = source["means"].new_tensor(settings.get("rotation_axis", (0., 0., 1.)))
            if settings["rotation_mode"] == "gravity_consistent":
                axis = axis.new_tensor([0., 0., 1.])
            if not torch.isfinite(axis).all() or axis.norm() == 0:
                raise ValueError("Manual rotation axis must be finite and nonzero")
            rotation = rotation_exp(axis / axis.norm() * math.radians(settings.get("rotation_degrees", 0.)))
        else:
            sampler = sample_uniform_z_rotation_quaternion if settings["rotation_mode"] == "gravity_consistent" else sample_uniform_rotation_quaternion
            rotation = sampler(source["means"].dtype, source["means"].device, generator, settings["rotation_max_degrees"])
        source_out = rotate_gaussians(source, rotation, settings["rotation_pivot"])
        target_out = rotate_gaussians(target, rotation, settings["rotation_pivot"])
    if settings["random_jitter"]:
        if manual:
            for key in GAUSSIAN_PARAMETERS:
                levels[key] = float(settings["jitter_max_levels"].get(key, 0.))
                if levels[key]:
                    source_out = jitter_gaussian_parameter(source_out, key, levels[key], generator)
        else:
            source_out, levels = jitter_gaussian_parameters(source_out, settings["jitter_max_levels"], generator)
    if backend == "edipt" and (settings["random_rotate"] or settings["random_jitter"]):
        source_out, target_out = standardizer.prepare_endpoints(source_out, target_out, align_target_sign=False)
    return source_out, target_out, rotation, levels


def analytic_state(backend, mode, t, source_flow, target_flow, noise, standardizer,
                   source_means, target_means, noise_scale=1., rotation_noise_std=.3):
    """Decode the actual training path, with finite, exact endpoint states."""
    if mode not in interpolants.MODES or not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Expected a supported interpolant and time in [0, 1]")
    if not math.isfinite(noise_scale) or noise_scale < 0 or not math.isfinite(rotation_noise_std) or rotation_noise_std < 0:
        raise ValueError("Noise scales must be finite and nonnegative")
    if t == 1:
        state = target_flow
    elif t == 0:
        if mode != "one_sided":
            state = source_flow
        elif backend == "edipt":
            state = {**noise["euclidean"], "quats": rotation_exp(rotation_noise_std * noise["rotation"])}
        else:
            state = noise
    else:
        module = edipt_interpolants if backend == "edipt" else interpolants
        kwargs = {"rotation_noise_std": rotation_noise_std} if backend == "edipt" else {}
        state = module.construct_path(None if mode == "one_sided" else source_flow, target_flow, t, mode,
                                      source_means, target_means, noise_scale, noise=noise, **kwargs)["query"]
    return standardizer.decode(state)


def serialize_reference(reference, settings):
    """Use Pointcept's serializer and the selected predictor's grid convention."""
    from pointcept.models.utils.structure import Point

    if not torch.isfinite(reference).all() or len(reference) == 0:
        raise ValueError("Serialization requires nonempty finite reference means")
    grid = torch.floor(reference.float() * settings["grid_resolution"])
    if settings["shift_negative_grid_coords"]:
        grid = grid - grid.amin(dim=0, keepdim=True).clamp(max=0)
    if (grid >= 65536).any():
        raise ValueError("Reference grid exceeds Pointcept's 16-bit coordinate range")
    point = Point({"grid_coord": grid.int(), "offset": torch.tensor([len(grid)], device=grid.device)})
    point.serialization(order=settings["order"], shuffle_orders=False)
    return point.serialized_order.clone()


def patch_layout(order, patch_size, backend, enable_flash=True):
    """Return query ownership and exact attention members, including padding."""
    if int(patch_size) != patch_size or patch_size < 1 or len(order) == 0:
        raise ValueError("Expected a positive integer patch size and nonempty order")
    size = int(patch_size)
    count = len(order)
    if backend == "dipt" and not enable_flash:
        size = min(size, count)
    primary = torch.empty_like(order)
    primary[order] = torch.arange(count, device=order.device) // size
    remainder = count % size
    padding = size - remainder if remainder else 0
    borrowed = padding if backend == "dipt" and count > size else 0
    masked = padding if backend == "edipt" else 0
    members = order
    valid = torch.ones(count, dtype=torch.bool, device=order.device)
    if borrowed:
        members = torch.cat((order, order[count - size:count - remainder]))
        valid = torch.ones(len(members), dtype=torch.bool, device=order.device)
    elif masked:
        members = torch.cat((order, order.new_zeros(masked)))
        valid = torch.cat((valid, torch.zeros(masked, dtype=torch.bool, device=order.device)))
    offsets = torch.arange(0, len(members), size, device=order.device)
    offsets = torch.cat((offsets, offsets.new_tensor([len(members)])))
    return {"ids": primary, "members": members, "valid": valid, "offsets": offsets,
            "patch_count": math.ceil(count / size), "effective_size": size,
            "borrowed": borrowed, "masked": masked}


def patch_colors(ids):
    """Use the same deterministic patch palette regardless of scene or backend."""
    palette = np.random.default_rng(0).integers(50, 256, size=(int(ids.max()) + 1, 3), dtype=np.uint8)
    return palette[ids.cpu().numpy()]


def compose_comparison(baseline, current, layouts, color_mode, pivot):
    """Translate copies for display, optionally replacing appearance with patch RGB."""
    boxes = [gs["means"].amin(0) for gs in (baseline, current)]
    tops = [gs["means"].amax(0) for gs in (baseline, current)]
    width = max(float(top[0] - box[0]) for box, top in zip(boxes, tops))
    gap = max(width * .2, .1)
    center = float(pivot[0])
    shifts = [center - gap / 2 - float(tops[0][0]), center + gap / 2 - float(boxes[1][0])]
    displayed = []
    for gs, layout, shift in zip((baseline, current), layouts, shifts):
        copy = dict(gs)
        copy["means"] = gs["means"] + gs["means"].new_tensor([shift, 0., 0.])
        if color_mode == "Patches":
            rgb = torch.from_numpy(patch_colors(layout["ids"])).to(gs["features_dc"]) / 255
            copy["features_dc"] = torch.logit(rgb.clamp(1e-6, 1 - 1e-6))
            copy["features_rest"] = gs["features_rest"].new_empty((len(rgb), 0, 3))
        displayed.append(copy)
    combined = {key: torch.cat([gs[key] for gs in displayed]) for key in baseline}
    return displayed, combined


class AugmentationComparison:
    """Keep immutable endpoints, independent RNG streams, and reusable order caches."""

    def __init__(self, source, target, standardizer, backend, settings, noise_seed=0, seed=0, mode="linear"):
        self.raw_source = {key: value.detach().cpu().clone() for key, value in source.items()}
        self.raw_target = {key: value.detach().cpu().clone() for key, value in target.items()}
        self.standardizer, self.backend, self.settings = standardizer, backend, settings
        if backend == "edipt" and standardizer.quaternion_representation != "unit_unstandardized":
            raise ValueError("EDiPT requires unit_unstandardized quaternion representation")
        self.order_cache = {}
        self.set_mode(mode)
        module = edipt_interpolants if backend == "edipt" else interpolants
        self.noise = module.seeded_noise_like(standardizer.encode(self.target), noise_seed)
        generator = torch.Generator().manual_seed(int(seed))
        self.order_rows = torch.randperm(len(settings["order"]), generator=generator).tolist() if settings["shuffle_orders"] else list(range(len(settings["order"])))

    def set_mode(self, mode):
        self.source, self.target = self.standardizer.prepare_endpoints(self.raw_source, self.raw_target, align_target_sign=mode != "one_sided")
        self.base_flow = (self.standardizer.encode(self.source), self.standardizer.encode(self.target))
        self.current_source, self.current_target = self.source, self.target
        self.current_flow = self.base_flow
        self.order_cache.clear()

    def augment(self, settings, seed):
        self.current_source, self.current_target, rotation, levels = augment_endpoints(
            self.source, self.target, self.standardizer, self.backend, settings, seed)
        self.current_flow = (self.standardizer.encode(self.current_source), self.standardizer.encode(self.current_target))
        self.order_cache = {key: value for key, value in self.order_cache.items() if key[0] == "baseline"}
        return rotation, levels

    def states(self, selection, mode, t, noise_scale, rotation_noise_std):
        endpoints = ((self.source, self.target), (self.current_source, self.current_target))
        if selection != "Interpolant":
            return [pair[0 if selection == "Source" else 1] for pair in endpoints]
        return [analytic_state(self.backend, mode, t, *flow, self.noise, self.standardizer,
                               pair[0]["means"], pair[1]["means"], noise_scale, rotation_noise_std)
                for pair, flow in zip(endpoints, (self.base_flow, self.current_flow))]

    def layouts(self, selection, mode, t, reference_mode, block, order_name, patch_size):
        use_target = selection == "Fitted target" or (selection == "Interpolant" and
                     (mode == "one_sided" or (mode == "encoding_decoding" and t >= .5)))
        endpoint = "target" if use_target else "source"
        references = [self.target if use_target else self.source,
                      self.current_target if use_target else self.current_source]
        row = self.order_rows[block % len(self.order_rows)] if order_name == "Block default" else self.settings["order"].index(order_name)
        layouts = []
        for index, reference in enumerate(references):
            variant = "baseline" if index == 0 or reference_mode == "unaugmented" else "current"
            key = (variant, endpoint)
            if key not in self.order_cache:
                actual = references[0] if variant == "baseline" else reference
                self.order_cache[key] = serialize_reference(actual["means"], self.settings)
            layouts.append(patch_layout(self.order_cache[key][row], patch_size, self.backend, self.settings["enable_flash"]))
        return layouts

"""Non-mutating augmentations for Gaussian splat parameter dictionaries."""

import math

import torch
import torch.nn.functional as F


GAUSSIAN_PARAMETERS = (
    "means",
    "scales",
    "opacities",
    "quats",
    "features_dc",
    "features_rest",
)


def _clone_gaussians(gs_params):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in gs_params.items()
    }


def quaternion_multiply(q1, q2):
    """Multiply scalar-first (wxyz) quaternions, broadcasting batch dimensions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quaternion_inverse(quaternion):
    """Invert scalar-first (wxyz) quaternions."""
    norm_squared = quaternion.square().sum(dim=-1, keepdim=True)
    if torch.any(norm_squared == 0):
        raise ValueError("Cannot invert a zero quaternion")
    conjugate = quaternion.clone()
    conjugate[..., 1:] = -conjugate[..., 1:]
    return conjugate / norm_squared


def _quaternion_to_rotation_matrix(quaternion):
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def sample_uniform_rotation_quaternion(dtype=torch.float32, device=None, generator=None):
    """Sample one uniform SO(3) rotation as a unit scalar-first quaternion."""
    quaternion = torch.randn(4, dtype=dtype, device=device, generator=generator)
    return F.normalize(quaternion, dim=-1)


def sample_uniform_z_rotation_quaternion(dtype=torch.float32, device=None, generator=None):
    """Sample a uniform yaw around world Z as a scalar-first quaternion."""
    half_angle = math.pi * torch.rand((), dtype=dtype, device=device, generator=generator)
    zero = torch.zeros_like(half_angle)
    return torch.stack((half_angle.cos(), zero, zero, half_angle.sin()))


def jitter_gaussian_parameter(gs_params, parameter, level, generator=None):
    """Jitter one parameter relative to its per-channel population deviation."""
    if parameter not in GAUSSIAN_PARAMETERS:
        raise ValueError(f"Unsupported Gaussian parameter {parameter!r}; expected one of {GAUSSIAN_PARAMETERS}")
    if parameter not in gs_params:
        raise KeyError(f"Gaussian parameters do not contain {parameter!r}")
    if not math.isfinite(float(level)) or float(level) < 0:
        raise ValueError("Jitter level must be finite and non-negative")

    augmented = _clone_gaussians(gs_params)
    if float(level) == 0:
        return augmented

    value = gs_params[parameter]
    augmented[parameter] = _jitter_value(value, float(level), generator)
    if parameter == "quats":
        augmented[parameter] = F.normalize(augmented[parameter], dim=-1)
    return augmented


def _jitter_value(value, level, generator):
    channel_std = value.std(dim=0, correction=0)
    noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
    return value + level * channel_std * noise


def jitter_gaussian_parameters(gs_params, max_levels, generator=None):
    """Jitter configured attributes with independently sampled relative levels."""
    unknown = sorted(set(max_levels) - set(GAUSSIAN_PARAMETERS))
    if unknown:
        raise ValueError(f"Unsupported Gaussian jitter parameters: {unknown}")
    missing = sorted(key for key, level in max_levels.items() if float(level) > 0 and key not in gs_params)
    if missing:
        raise KeyError(f"Gaussian parameters do not contain configured attributes: {missing}")
    for parameter, max_level in max_levels.items():
        if not math.isfinite(float(max_level)) or float(max_level) < 0:
            raise ValueError(f"Maximum jitter level for {parameter!r} must be finite and non-negative")

    augmented = _clone_gaussians(gs_params)
    sampled_levels = {}
    for parameter in GAUSSIAN_PARAMETERS:
        max_level = float(max_levels.get(parameter, 0.0))
        if max_level == 0 or parameter not in gs_params:
            sampled_levels[parameter] = 0.0
            continue
        level = torch.rand((), device=gs_params[parameter].device, generator=generator).item() * max_level
        augmented[parameter] = _jitter_value(gs_params[parameter], level, generator)
        if parameter == "quats":
            augmented[parameter] = F.normalize(augmented[parameter], dim=-1)
        sampled_levels[parameter] = level
    return augmented, sampled_levels


def _rotate_degree_one_sh(features_rest, rotation_matrix):
    """Rotate coefficients in gsplat's degree-one real-SH basis [-y, z, -x]."""
    if features_rest.ndim != 3 or features_rest.shape[1:] != (3, 3):
        raise ValueError(
            "SH rotation supports degree 1 features_rest with shape [N, 3, 3]; "
            f"got {tuple(features_rest.shape)}"
        )
    basis_change = features_rest.new_tensor(((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
    coefficient_rotation = basis_change @ rotation_matrix @ basis_change.transpose(0, 1)
    return torch.einsum("ij,njc->nic", coefficient_rotation, features_rest)


def rotate_gaussians(gs_params, rotation, pivot, rotate_sh=True):
    """Apply one global rotation to means, orientations, and degree-one SH."""
    if "means" not in gs_params or "quats" not in gs_params:
        raise KeyError("Gaussian rotation requires 'means' and 'quats'")

    means = gs_params["means"]
    rotation = torch.as_tensor(rotation, dtype=means.dtype, device=means.device)
    pivot = torch.as_tensor(pivot, dtype=means.dtype, device=means.device)
    if rotation.shape != (4,):
        raise ValueError(f"Expected one rotation quaternion with shape (4,), got {tuple(rotation.shape)}")
    if pivot.shape != (3,):
        raise ValueError(f"Expected rotation pivot with shape (3,), got {tuple(pivot.shape)}")
    if torch.linalg.vector_norm(rotation) == 0:
        raise ValueError("Rotation quaternion must be non-zero")

    rotation = F.normalize(rotation, dim=-1)
    rotation_matrix = _quaternion_to_rotation_matrix(rotation)
    augmented = _clone_gaussians(gs_params)
    augmented["means"] = (means - pivot) @ rotation_matrix.transpose(0, 1) + pivot

    quats = gs_params["quats"]
    global_quaternion = rotation.to(dtype=quats.dtype, device=quats.device).expand_as(quats)
    augmented["quats"] = F.normalize(quaternion_multiply(global_quaternion, quats), dim=-1)

    features_rest = gs_params.get("features_rest")
    if rotate_sh and features_rest is not None and features_rest.shape[1] > 0:
        augmented["features_rest"] = _rotate_degree_one_sh(
            features_rest, rotation_matrix.to(dtype=features_rest.dtype, device=features_rest.device)
        )
    return augmented


def rotate_camera_to_worlds(camera_to_worlds, rotation, pivot):
    """Apply a global world rotation to OpenGL camera-to-world matrices."""
    if camera_to_worlds.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera-to-world matrices with shape [..., 4, 4], got {tuple(camera_to_worlds.shape)}")
    rotation = torch.as_tensor(rotation, dtype=camera_to_worlds.dtype, device=camera_to_worlds.device)
    pivot = torch.as_tensor(pivot, dtype=camera_to_worlds.dtype, device=camera_to_worlds.device)
    if rotation.shape != (4,) or pivot.shape != (3,):
        raise ValueError("Camera rotation requires a quaternion with shape (4,) and pivot with shape (3,)")
    if torch.linalg.vector_norm(rotation) == 0:
        raise ValueError("Rotation quaternion must be non-zero")

    rotation_matrix = _quaternion_to_rotation_matrix(rotation)
    transformed = camera_to_worlds.clone()
    transformed[..., :3, :3] = rotation_matrix @ camera_to_worlds[..., :3, :3]
    translations = camera_to_worlds[..., :3, 3]
    transformed[..., :3, 3] = (translations - pivot) @ rotation_matrix.transpose(0, 1) + pivot
    return transformed

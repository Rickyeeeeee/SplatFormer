"""Spatial fields for reference-based and dynamic Pointcept serialization."""
import math

import torch


SPATIAL_COORD_MODES = ("reference", "dynamic")


@torch.no_grad()
def morton_order(means, bits=10):
    """Return a scene's Gaussian indices in Morton order, breaking voxel ties by position."""
    if means.ndim != 2 or means.shape[1] != 3 or not len(means) or not torch.isfinite(means).all():
        raise ValueError("Morton ordering requires nonempty finite means with shape [N, 3]")
    coordinates = means.float()
    minimum = coordinates.amin(dim=0)
    extent = coordinates.amax(dim=0) - minimum
    grid = ((coordinates - minimum) / extent.clamp_min(torch.finfo(coordinates.dtype).eps) * (2 ** bits - 1))
    grid = grid.floor().clamp(0, 2 ** bits - 1).long()
    code = torch.zeros(len(means), device=means.device, dtype=torch.long)
    # Interleave x, y, and z bits into one Morton code per Gaussian.
    for bit in range(bits):
        for axis in range(3):
            code |= ((grid[:, axis] >> bit) & 1) << (3 * bit + axis)
    order = torch.arange(len(means), device=means.device)
    for axis in (2, 1, 0):
        order = order[torch.argsort(coordinates[order, axis], stable=True)]
    return order[torch.argsort(code[order], stable=True)]


def build_spatial_point_fields(batch_flow_gs, batch_reference_means, mode, grid_resolution, dynamic_grid_size):
    """Build Pointcept coordinates while leaving dynamic grids to its shifted voxelization."""
    if mode not in SPATIAL_COORD_MODES:
        raise ValueError(f"Unknown spatial_coord_mode {mode!r}; expected {SPATIAL_COORD_MODES}")
    references = batch_reference_means or [gs["means"] for gs in batch_flow_gs]
    coord = torch.cat(references, dim=0)
    if mode == "dynamic":
        if not math.isfinite(dynamic_grid_size) or dynamic_grid_size <= 0:
            raise ValueError("dynamic_grid_size must be finite and positive")
        # coord = torch.cat([gs["means"] for gs in batch_flow_gs], dim=0)
        return {"coord": coord, "grid_size": coord.new_full((3,), dynamic_grid_size)}
    if grid_resolution <= 0:
        raise ValueError("grid_resolution must be positive")
    return {
        "coord": coord,
        "grid_coord": torch.floor(coord * grid_resolution).int(),
        "grid_size": coord.new_full((3,), 1.0 / grid_resolution),
    }

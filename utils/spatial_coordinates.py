"""Spatial fields for reference-based and dynamic Pointcept serialization."""
import math

import torch


SPATIAL_COORD_MODES = ("reference", "dynamic")


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

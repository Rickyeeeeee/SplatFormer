"""Dataset and sampling choices shared by unconditional noise-to-GS training checks."""

from pathlib import Path

import torch

from dataset.gs_io import latest_gsplat_checkpoint
from utils.spatial_coordinates import morton_order


PAD_OPACITY_LOGIT = -15.0


def select_available_scenes(dataset, splat_dir):
    """Keep listed scenes with a checkpoint in the supplied splat directory."""
    splat_dir = Path(splat_dir)
    available, skipped, seen = [], [], set()
    for entry in dataset.folders:
        name = entry["scene_name"]
        if name in seen:
            continue
        seen.add(name)
        path = splat_dir / name
        try:
            latest_gsplat_checkpoint(path)
        except FileNotFoundError:
            skipped.append(name)
            continue
        entry["resolution_paths"][dataset.tgt_resolution]["gsplat_dir"] = str(path)
        available.append(entry)
    if not available:
        raise ValueError(f"No listed scenes have GS checkpoints under {splat_dir}")
    dataset.folders = available
    return skipped


def prepare_fixed_count_target(gaussians, count, coordinate_mode="morton_rank"):
    """Complete a normalized target and order it only for Morton-rank DiT."""
    size = len(gaussians["means"])
    if not 1 <= size <= count:
        raise ValueError(f"Expected 1 to {count} selected Gaussians, got {size}")
    missing = count - size
    if missing:
        padded = {}
        for key, value in gaussians.items():
            fill = value.new_zeros((missing,) + value.shape[1:])
            if key == "means":
                fill.fill_(0.5)
            elif key == "opacities":
                fill.fill_(PAD_OPACITY_LOGIT)
            elif key == "scales" or key == "features_dc":
                fill[:] = value.median(dim=0).values
            elif key == "quats":
                fill[:, 0] = 1.0
            padded[key] = torch.cat((value, fill), dim=0)
        gaussians = padded
    if coordinate_mode == "morton_rank":
        order = morton_order(gaussians["means"])
        return {key: value[order] for key, value in gaussians.items()}
    if coordinate_mode != "current":
        raise ValueError(f"Unsupported unconditional coordinate mode: {coordinate_mode}")
    return gaussians


def evenly_spaced_camera_indices(total_views, count=8):
    if total_views < count:
        raise ValueError(f"Evaluation scene needs at least {count} cameras, got {total_views}")
    return [round(index * (total_views - 1) / (count - 1)) for index in range(count)]


def evaluation_seed(base_seed, slot):
    return int(base_seed) + int(slot)

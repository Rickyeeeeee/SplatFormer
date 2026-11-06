"""Dataset and sampling choices shared by unconditional noise-to-GS training checks."""

from pathlib import Path

import torch

from dataset.gs_io import latest_gsplat_checkpoint
from utils.spatial_coordinates import morton_order
from utils.gs_normalization import STATISTIC_KEYS


PAD_OPACITY_LOGIT = -15.0


def mean_feature_losses(feature_loss_sum, scene_count, weights):
    """Summarize feature sums after microbatch and distributed reduction."""
    raw = dict(zip(STATISTIC_KEYS, (feature_loss_sum / scene_count).tolist()))
    weighted = {key: value * weights.get(key, 1.0) for key, value in raw.items()}
    return raw, weighted


def select_available_scenes(dataset, splat_dir, scene_count=-1):
    """Keep listed scenes with a checkpoint in the supplied splat directory."""
    if scene_count != -1 and (scene_count < 1 or scene_count > len(dataset.folders)):
        raise ValueError(f"scene_count must be -1 or between 1 and {len(dataset.folders)}")
    splat_dir = Path(splat_dir)
    available, skipped, seen = [], [], set()
    entries = dataset.folders if scene_count == -1 else dataset.folders[:scene_count]
    for entry in entries:
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
    if scene_count != -1 and len(available) != scene_count:
        raise ValueError(f"First {scene_count} listed scenes must be distinct and have GS checkpoints; "
                         f"found {len(available)} available, missing={skipped}")
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

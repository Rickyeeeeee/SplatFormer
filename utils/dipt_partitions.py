"""Shared serialized attention partitions for DiPT and its viewer."""

import torch

from pointcept.models.utils.serialization import encode


PARTITION_MODES = ("single", "rotated", "shifted", "both")
ROTATION_TURNS = (0, 1, 2, 3)


def partition_views(mode, patch_size):
    if mode not in PARTITION_MODES:
        raise ValueError(f"Unknown DiPT partition mode: {mode}")
    if mode != "single" and patch_size % 4:
        raise ValueError("Multi-partition DiPT requires patch sizes divisible by four")
    views = [("0°", 0, 0)]
    if mode in ("rotated", "both"):
        views.extend((f"{turn * 90}°", turn, 0) for turn in ROTATION_TURNS[1:])
    if mode in ("shifted", "both"):
        views.extend((f"shift {part}/4 K", 0, part * patch_size // 4) for part in (1, 2, 3))
    return views


def rotated_grid(reference, offset, resolution, shift_negative, turn, pivot=(0.5, 0.5, 0.5)):
    """Quantize quarter-turn reference positions without changing the model geometry."""
    centered = reference - reference.new_tensor(pivot)
    if turn == 1:
        xy = torch.stack((-centered[:, 1], centered[:, 0]), dim=1)
    elif turn == 2:
        xy = -centered[:, :2]
    elif turn == 3:
        xy = torch.stack((centered[:, 1], -centered[:, 0]), dim=1)
    else:
        xy = centered[:, :2]
    rotated = torch.cat((xy, centered[:, 2:]), dim=1) + reference.new_tensor(pivot)
    grid = torch.floor(rotated * resolution).int()
    if shift_negative:
        left = 0
        for right in offset.tolist():
            grid[left:right] -= grid[left:right].amin(0).clamp(max=0)
            left = right
    if (grid >= 65536).any():
        raise ValueError("Rotated reference grid exceeds Pointcept's 16-bit coordinate range")
    return grid


@torch.no_grad()
def rotated_orders(reference, offset, resolution, shift_negative, orders, depth, pivot=(0.5, 0.5, 0.5)):
    """Return original-point permutations for the three non-identity turns."""
    counts = torch.diff(torch.cat((offset.new_zeros(1), offset)))
    batch = torch.repeat_interleave(torch.arange(len(offset), device=offset.device), counts)
    result = {}
    for turn in ROTATION_TURNS[1:]:
        grid = rotated_grid(reference, offset, resolution, shift_negative, turn, pivot)
        rotated_depth = max(depth, int(grid.max()).bit_length())
        if rotated_depth > 16 or rotated_depth * 3 + len(offset).bit_length() > 63:
            raise ValueError("Rotated reference exceeds Pointcept's serialization range")
        result[turn] = torch.stack([torch.argsort(encode(grid, batch, rotated_depth, order=name)) for name in orders])
    return result


def attention_layout(order, offset, patch_size, shift=0):
    """Build exact Flash Attention members, query restoration, and scene-local patches."""
    if patch_size < 1 or shift < 0 or shift >= patch_size:
        raise ValueError("Invalid patch size or boundary shift")
    members, restore, ids, boundaries = [], [], [], [0]
    borrowed = 0
    left = 0
    patch_index = 0
    for right in offset.tolist():
        scene_order = order[left:right]
        count = right - left
        starts = [0] if shift == 0 else [0, min(count, patch_size - shift)]
        while starts[-1] < count:
            starts.append(min(count, starts[-1] + patch_size))
        if len(starts) == 1:
            starts.append(count)
        scene_members = scene_order
        if shift == 0 and count > patch_size and count % patch_size:
            extra = patch_size - count % patch_size
            scene_members = torch.cat((scene_order, scene_order[count - patch_size:count - count % patch_size]))
            starts[-1] += extra
            borrowed += extra
        members.append(scene_members)
        inverse = torch.empty(count, dtype=torch.long, device=order.device)
        inverse[scene_order - left] = torch.arange(count, device=order.device)
        restore.append(inverse + boundaries[-1])
        scene_ids = torch.empty(count, dtype=torch.long, device=order.device)
        for begin, end in zip(starts[:-1], starts[1:]):
            scene_ids[scene_order[begin:min(end, count)] - left] = patch_index
            patch_index += 1
            boundaries.append(boundaries[-1] + end - begin)
        ids.append(scene_ids)
        left = right
    return {"members": torch.cat(members), "restore": torch.cat(restore), "ids": torch.cat(ids),
            "offsets": torch.tensor(boundaries, dtype=torch.int32, device=order.device),
            "patch_count": patch_index, "effective_size": patch_size, "borrowed": borrowed, "masked": 0,
            "valid": torch.ones(boundaries[-1], dtype=torch.bool, device=order.device)}


def neighborhood(layout, index):
    """Unique members of the selected query's actual attention patch."""
    if index < 0 or index >= len(layout["ids"]):
        raise ValueError("Gaussian index is outside the scene")
    patch = int(layout["ids"][index])
    begin, end = layout["offsets"][patch:patch + 2].tolist()
    return torch.unique(layout["members"][begin:end][layout["valid"][begin:end]])

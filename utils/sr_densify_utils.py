import os

import numpy as np
import pointops
import torch

from utils import gpu_utils, gs_utils


def _as_scaler_tensor(scaler, name, device, dtype):
    value = getattr(scaler, name)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def _as_scale_tensor(scaler, device, dtype):
    return _as_scaler_tensor(scaler, "scale_", device, dtype)


def _scaler_transform(scaler, value):
    device = value.device
    dtype = value.dtype
    scale = _as_scaler_tensor(scaler, "scale_", device, dtype)
    trans = _as_scaler_tensor(scaler, "trans_", device, dtype)
    return value * scale + trans


def _scaler_inverse_transform(scaler, value):
    device = value.device
    dtype = value.dtype
    scale = _as_scaler_tensor(scaler, "scale_", device, dtype)
    trans = _as_scaler_tensor(scaler, "trans_", device, dtype)
    return (value - trans) / scale


def convert_gs_to_target_frame(input_gs, input_scaler, target_scaler):
    target_gs = {}
    device = input_gs["means"].device
    dtype = input_gs["means"].dtype

    raw_means = _scaler_inverse_transform(input_scaler, input_gs["means"])
    target_gs["means"] = _scaler_transform(target_scaler, raw_means)

    input_scale = _as_scale_tensor(input_scaler, device, dtype)
    target_scale = _as_scale_tensor(target_scaler, device, dtype)
    for key, value in input_gs.items():
        if key == "means":
            continue
        if key == "scales":
            target_gs[key] = value - torch.log(input_scale) + torch.log(target_scale)
        else:
            target_gs[key] = value.clone()
    return target_gs


def chunked_knn_indices(points, centers, k, chunk_size=512):
    if points.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("Cannot query kNN on empty point sets")
    k = min(int(k), points.shape[0])
    idx_chunks = []
    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        dist = torch.cdist(centers[start:end].float(), points.float())
        idx_chunks.append(torch.topk(dist, k=k, dim=1, largest=False).indices)
    return torch.cat(idx_chunks, dim=0)


def chunked_nearest_indices(source_means, target_means, chunk_size=512):
    return chunked_knn_indices(source_means, target_means, 1, chunk_size=chunk_size).squeeze(1)


def midpoint_interpolate_gs(input_gs, target_count):
    source_count = input_gs["means"].shape[0]
    if source_count <= 0:
        raise ValueError("Cannot densify an empty input GS")
    if target_count < source_count:
        raise ValueError(
            f"Cannot preserve {source_count} source Gaussians when target_count={target_count}"
        )

    new_count = target_count - source_count
    if new_count == 0:
        return {key: value.clone() for key, value in input_gs.items()}
    if source_count == 1:
        raise ValueError("Cannot create midpoint Gaussians from a single source Gaussian")

    means = input_gs["means"].contiguous()
    if not means.is_cuda:
        raise ValueError("Pointcept pointops midpoint interpolation requires CUDA tensors")

    up_rate = float(target_count) / float(source_count)
    k = min(source_count, int(2 * up_rate))
    if k < 2:
        raise ValueError(f"Need at least 2 neighbors for midpoint interpolation, got k={k}")

    offset = torch.tensor([source_count], device=means.device, dtype=torch.int32)
    nn_idx, _ = pointops.knn_query(k, means, offset, means, offset)
    nn_idx = nn_idx.long()
    src_idx = torch.arange(source_count, device=means.device).unsqueeze(1).expand(-1, k).reshape(-1)
    nbr_idx = nn_idx.reshape(-1)
    non_self = src_idx != nbr_idx
    src_idx = src_idx[non_self]
    nbr_idx = nbr_idx[non_self]

    candidate_count = src_idx.shape[0]
    if candidate_count < new_count:
        raise ValueError(
            f"PUFM midpoint interpolation produced {candidate_count} new candidates, "
            f"but target requires {new_count}. source_count={source_count}, k={k}"
        )
    if candidate_count > new_count:
        candidate_means = ((means[src_idx] + means[nbr_idx]) * 0.5).contiguous()
        candidate_offset = torch.tensor([candidate_count], device=means.device, dtype=torch.int32)
        new_offset = torch.tensor([new_count], device=means.device, dtype=torch.int32)
        keep_idx = pointops.farthest_point_sampling(candidate_means, candidate_offset, new_offset).long()
        src_idx = src_idx[keep_idx]
        nbr_idx = nbr_idx[keep_idx]

    interpolated = {}
    for key, value in input_gs.items():
        src_value = value[src_idx]
        nbr_value = value[nbr_idx]

        if key in ["means", "features_dc", "features_rest"]:
            midpoint_value = (src_value + nbr_value) * 0.5
        elif key == "quats":
            midpoint_value = torch.zeros_like(src_value)
            midpoint_value[:, 0] = 1
        elif key == "opacities":
            low_opacity = torch.full_like(src_value, 0.01)
            midpoint_value = torch.logit(low_opacity)
        elif key == "scales":
            pair_min_scale = torch.minimum(src_value, nbr_value).min(dim=-1, keepdim=True).values
            midpoint_value = pair_min_scale.repeat(1, src_value.shape[-1])
        else:
            midpoint_value = (src_value + nbr_value) * 0.5

        interpolated[key] = torch.cat([value.clone(), midpoint_value], dim=0)
    return interpolated


def align_nearest_target_to_source(source_means, target_means):
    return chunked_nearest_indices(source_means, target_means)


def align_emd_target_to_source(source_means, target_means, eps, iters):
    try:
        from emd_assignment import emd_module
    except Exception as exc:
        raise ImportError(
            "Could not import local EMD package. Run "
            "`cd /home/ricky/SplatFormer/emd_assignment && python setup.py install`, "
            "or rerun with `--alignment=nearest`."
        ) from exc

    if source_means.shape[0] != target_means.shape[0]:
        raise ValueError(
            f"EMD alignment requires equal counts, got {source_means.shape[0]} and {target_means.shape[0]}"
        )

    count = source_means.shape[0]
    padded_count = int(np.ceil(count / 128.0) * 128)
    source_pad = source_means
    target_pad = target_means
    if padded_count != count:
        pad = padded_count - count
        source_pad = torch.cat([source_means, source_means[-1:].expand(pad, -1)], dim=0)
        target_pad = torch.cat([target_means, target_means[-1:].expand(pad, -1)], dim=0)

    aligner = emd_module.emdModule()
    with torch.no_grad():
        _, assignment = aligner(
            source_pad.unsqueeze(0).contiguous(),
            target_pad.unsqueeze(0).contiguous(),
            float(eps),
            int(iters),
        )
    assignment = assignment[0, :count].detach().long()
    assigned_target = assignment.cpu()
    source_cpu = torch.arange(count, dtype=torch.long)
    dist_cpu = ((source_means - target_pad[assignment.clamp(min=0)].to(source_means.device)) ** 2).sum(dim=1).detach().cpu()

    best_source = torch.full((count,), -1, dtype=torch.long)
    best_dist = torch.full((count,), float("inf"))
    valid = (assigned_target >= 0) & (assigned_target < count)
    for src_i, tgt_i, dist_i in zip(source_cpu[valid].tolist(), assigned_target[valid].tolist(), dist_cpu[valid].tolist()):
        if dist_i < best_dist[tgt_i].item():
            best_dist[tgt_i] = dist_i
            best_source[tgt_i] = src_i

    missing = best_source < 0
    if missing.any():
        missing_idx = missing.nonzero(as_tuple=False).squeeze(1).to(target_means.device)
        nearest = align_nearest_target_to_source(source_means, target_means[missing_idx]).detach().cpu()
        best_source[missing] = nearest

    return best_source.to(source_means.device)


def nearest_neighbor_dist2(means):
    count = means.shape[0]
    if count <= 1:
        return torch.full((count,), 1e-7, device=means.device, dtype=means.dtype)
    nn_idx = chunked_knn_indices(means, means, 2)
    nearest = means[nn_idx[:, 1]]
    dist2 = ((means - nearest) ** 2).sum(dim=-1)
    return torch.clamp_min(dist2, 1e-7)


def apply_3dgs_attribute_init(densified_gs):
    means = densified_gs["means"]
    count = means.shape[0]
    device = means.device
    dtype = means.dtype

    if "scales" in densified_gs:
        dist2 = nearest_neighbor_dist2(means)
        densified_gs["scales"] = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    if "quats" in densified_gs:
        quats = torch.zeros((count, 4), device=device, dtype=dtype)
        quats[:, 0] = 1
        densified_gs["quats"] = quats

    if "opacities" in densified_gs:
        opacity = 0.1 * torch.ones((count, 1), device=device, dtype=dtype)
        densified_gs["opacities"] = torch.logit(opacity)

    return densified_gs


def apply_gt_attribute_overrides(densified_gs, target_gs, gt_attribute_keys):
    for key in gt_attribute_keys:
        if key in target_gs and key in densified_gs:
            densified_gs[key] = target_gs[key].clone()
    return densified_gs


def densify_stage_gs(low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs):
    return {
        "00_low_res_gs.ply": low_res_gs,
        "01_interpolated_high_res_gs.ply": interpolated_gs,
        "02_gt_high_res_gs.ply": gt_high_res_gs,
        "03_input_high_res_gs.ply": input_high_res_gs,
    }


def save_densify_stage_plys(output_dir, low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs):
    stage_dir = os.path.join(output_dir, "densify_init")
    for ply_name, gs in densify_stage_gs(
        low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs
    ).items():
        gs_utils.export_ply_forviewer(gs, os.path.join(stage_dir, ply_name))


def build_densified_input_gs(
    input_factor_dict,
    target_factor_dict,
    alignment,
    attribute_init,
    emd_eps,
    emd_iters,
    device,
    output_dir=None,
    gt_attribute_keys=None,
    return_stages=False,
):
    input_gs = gpu_utils.move_to_device(input_factor_dict["gs_params"], device)
    target_gs = gpu_utils.move_to_device(target_factor_dict["gs_params"], device)
    target_count = target_gs["means"].shape[0]

    input_in_target_frame = convert_gs_to_target_frame(
        input_gs,
        input_factor_dict["scaler"],
        target_factor_dict["scaler"],
    )
    interpolated_gs = midpoint_interpolate_gs(input_in_target_frame, target_count)

    if alignment == "emd":
        source_idx = align_emd_target_to_source(
            interpolated_gs["means"], target_gs["means"], emd_eps, emd_iters
        )
    elif alignment == "nearest":
        source_idx = align_nearest_target_to_source(interpolated_gs["means"], target_gs["means"])
    else:
        raise ValueError(f"Unsupported alignment method: {alignment}")

    densified_gs = {}
    for key, value in target_gs.items():
        if key in interpolated_gs:
            densified_gs[key] = interpolated_gs[key][source_idx].clone()
        else:
            densified_gs[key] = value.clone()

    if attribute_init == "3dgs":
        densified_gs = apply_3dgs_attribute_init(densified_gs)
    elif attribute_init != "aligned":
        raise ValueError(f"Unsupported attribute initialization: {attribute_init}")

    if gt_attribute_keys is None:
        gt_attribute_keys = []
    densified_gs = apply_gt_attribute_overrides(densified_gs, target_gs, gt_attribute_keys)

    if output_dir is not None:
        save_densify_stage_plys(
            output_dir=output_dir,
            low_res_gs=input_in_target_frame,
            interpolated_gs=interpolated_gs,
            gt_high_res_gs=target_gs,
            input_high_res_gs=densified_gs,
        )

    if return_stages:
        return densified_gs, densify_stage_gs(
            low_res_gs=input_in_target_frame,
            interpolated_gs=interpolated_gs,
            gt_high_res_gs=target_gs,
            input_high_res_gs=densified_gs,
        )
    return densified_gs

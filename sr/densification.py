import os

import numpy as np
import pointops
import torch

from utils import gpu_utils, gs_utils


def chunked_knn_indices(points, centers, k, chunk_size=512):
    if points.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("Cannot query kNN on empty point sets")
    k = min(int(k), points.shape[0])
    index_chunks = []
    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        distances = torch.cdist(centers[start:end].float(), points.float())
        index_chunks.append(
            torch.topk(distances, k=k, dim=1, largest=False).indices
        )
    return torch.cat(index_chunks, dim=0)


def midpoint_interpolate_gaussians(input_gs, target_count):
    source_count = input_gs["means"].shape[0]
    if source_count <= 0:
        raise ValueError("Cannot densify an empty input GS")
    if target_count < source_count:
        return gs_utils.clone_gaussians(input_gs)

    new_count = target_count - source_count
    if new_count == 0:
        return gs_utils.clone_gaussians(input_gs)
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
    neighbor_indices, _ = pointops.knn_query(k, means, offset, means, offset)
    neighbor_indices = neighbor_indices.long()
    source_indices = (
        torch.arange(source_count, device=means.device)
        .unsqueeze(1)
        .expand(-1, k)
        .reshape(-1)
    )
    neighbor_indices = neighbor_indices.reshape(-1)
    non_self = source_indices != neighbor_indices
    source_indices = source_indices[non_self]
    neighbor_indices = neighbor_indices[non_self]

    candidate_count = source_indices.shape[0]
    if candidate_count < new_count:
        raise ValueError(
            f"PUFM midpoint interpolation produced {candidate_count} new candidates, "
            f"but target requires {new_count}. source_count={source_count}, k={k}"
        )
    if candidate_count > new_count:
        candidate_means = (
            (means[source_indices] + means[neighbor_indices]) * 0.5
        ).contiguous()
        candidate_offset = torch.tensor(
            [candidate_count], device=means.device, dtype=torch.int32
        )
        new_offset = torch.tensor(
            [new_count], device=means.device, dtype=torch.int32
        )
        keep_indices = pointops.farthest_point_sampling(
            candidate_means, candidate_offset, new_offset
        ).long()
        source_indices = source_indices[keep_indices]
        neighbor_indices = neighbor_indices[keep_indices]

    midpoint_means = (
        (means[source_indices] + means[neighbor_indices]) * 0.5
    ).contiguous()
    interpolated = {}
    for key, value in input_gs.items():
        source_value = value[source_indices]
        neighbor_value = value[neighbor_indices]
        if key == "means":
            midpoint_value = midpoint_means + 0.01 * (
                torch.rand_like(source_value) - 0.5
            )
        elif key in ("features_dc", "features_rest"):
            midpoint_value = (source_value + neighbor_value) * 0.5
        elif key == "quats":
            midpoint_value = torch.zeros_like(source_value)
            midpoint_value[:, 0] = 1
        elif key == "opacities":
            midpoint_value = torch.log(torch.full_like(source_value, 0.99))
        elif key == "scales":
            pair_scale = torch.maximum(source_value, neighbor_value)
            pair_scale = pair_scale.min(dim=-1, keepdim=True).values
            midpoint_value = pair_scale.repeat(1, source_value.shape[-1])
        else:
            midpoint_value = (source_value + neighbor_value) * 0.5
        interpolated[key] = torch.cat([value.clone(), midpoint_value], dim=0)
    return interpolated


def align_nearest_target_to_source(source_means, target_means, chunk_size=512):
    return chunked_knn_indices(
        source_means, target_means, 1, chunk_size=chunk_size
    ).squeeze(1)


def align_emd_target_to_source(source_means, target_means, eps, iters):
    try:
        from emd_assignment import emd_module
    except Exception as exc:
        raise ImportError(
            "Could not import local EMD package. Run "
            "cd /home/ricky/SplatFormer/emd_assignment && python setup.py install, "
            "or rerun with --alignment=nearest."
        ) from exc

    if source_means.shape[0] != target_means.shape[0]:
        raise ValueError(
            f"EMD alignment requires equal counts, got "
            f"{source_means.shape[0]} and {target_means.shape[0]}"
        )

    count = source_means.shape[0]
    padded_count = int(np.ceil(count / 128.0) * 128)
    source_pad = source_means
    target_pad = target_means
    if padded_count != count:
        pad = padded_count - count
        source_pad = torch.cat(
            [source_means, source_means[-1:].expand(pad, -1)], dim=0
        )
        target_pad = torch.cat(
            [target_means, target_means[-1:].expand(pad, -1)], dim=0
        )

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
    target_for_source = target_pad[assignment.clamp(min=0)].to(source_means.device)
    distance_cpu = (
        (source_means - target_for_source).square().sum(dim=1).detach().cpu()
    )

    best_source = torch.full((count,), -1, dtype=torch.long)
    best_distance = torch.full((count,), float("inf"))
    valid = (assigned_target >= 0) & (assigned_target < count)
    for source_index, target_index, distance in zip(
        source_cpu[valid].tolist(),
        assigned_target[valid].tolist(),
        distance_cpu[valid].tolist(),
    ):
        if distance < best_distance[target_index].item():
            best_distance[target_index] = distance
            best_source[target_index] = source_index

    missing = best_source < 0
    if missing.any():
        missing_indices = missing.nonzero(as_tuple=False).squeeze(1).to(
            target_means.device
        )
        nearest = align_nearest_target_to_source(
            source_means, target_means[missing_indices]
        ).detach().cpu()
        best_source[missing] = nearest
    return best_source.to(source_means.device)


def nearest_neighbor_dist2(means):
    count = means.shape[0]
    if count <= 1:
        return torch.full((count,), 1e-7, device=means.device, dtype=means.dtype)
    neighbor_indices = chunked_knn_indices(means, means, 2)
    nearest = means[neighbor_indices[:, 1]]
    return torch.clamp_min((means - nearest).square().sum(dim=-1), 1e-7)


def initialize_3dgs_attributes(densified_gs):
    means = densified_gs["means"]
    count = means.shape[0]
    if "scales" in densified_gs:
        distance2 = nearest_neighbor_dist2(means)
        densified_gs["scales"] = torch.log(torch.sqrt(distance2))[
            ..., None
        ].repeat(1, 3)
    if "quats" in densified_gs:
        quats = torch.zeros((count, 4), device=means.device, dtype=means.dtype)
        quats[:, 0] = 1
        densified_gs["quats"] = quats
    if "opacities" in densified_gs:
        opacity = torch.full(
            (count, 1), 0.1, device=means.device, dtype=means.dtype
        )
        densified_gs["opacities"] = torch.logit(opacity)
    return densified_gs


def densification_stages(
    low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs
):
    return {
        "00_low_res_gs.ply": low_res_gs,
        "01_interpolated_high_res_gs.ply": interpolated_gs,
        "02_gt_high_res_gs.ply": gt_high_res_gs,
        "03_input_high_res_gs.ply": input_high_res_gs,
    }


def save_densification_stages(
    output_dir, low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs
):
    stage_dir = os.path.join(output_dir, "densify_init")
    stages = densification_stages(
        low_res_gs, interpolated_gs, gt_high_res_gs, input_high_res_gs
    )
    for ply_name, gs in stages.items():
        gs_utils.export_ply_forviewer(gs, os.path.join(stage_dir, ply_name))


def build_densified_input(
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
    input_in_target_frame = gs_utils.convert_gaussian_frame(
        input_gs,
        input_factor_dict["scaler"],
        target_factor_dict["scaler"],
    )
    interpolated_gs = midpoint_interpolate_gaussians(
        input_in_target_frame, target_count
    )

    if alignment == "emd":
        source_indices = align_emd_target_to_source(
            interpolated_gs["means"], target_gs["means"], emd_eps, emd_iters
        )
    elif alignment == "nearest":
        source_indices = align_nearest_target_to_source(
            interpolated_gs["means"], target_gs["means"]
        )
    elif alignment == "none":
        source_indices = torch.arange(
            target_count, device=interpolated_gs["means"].device
        )
    else:
        raise ValueError(f"Unsupported alignment method: {alignment}")

    densified_gs = {
        key: (
            interpolated_gs[key][source_indices].clone()
            if key in interpolated_gs
            else value.clone()
        )
        for key, value in target_gs.items()
    }
    if attribute_init == "3dgs":
        densified_gs = initialize_3dgs_attributes(densified_gs)
    elif attribute_init != "aligned":
        raise ValueError(f"Unsupported attribute initialization: {attribute_init}")

    if gt_attribute_keys:
        densified_gs = gs_utils.copy_gt_attributes(
            densified_gs, target_gs, gt_attribute_keys
        )

    stages = densification_stages(
        input_in_target_frame,
        interpolated_gs,
        target_gs,
        densified_gs,
    )
    if output_dir is not None:
        save_densification_stages(
            output_dir,
            input_in_target_frame,
            interpolated_gs,
            target_gs,
            densified_gs,
        )
    if return_stages:
        return densified_gs, stages
    return densified_gs

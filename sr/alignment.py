import torch

from sr import densification, matching
from utils import gpu_utils, gs_utils
from utils.metrics import write_densify_stage_render_metrics

def prepare_alignment(
    dataset,
    scene,
    input_resolution_entry,
    target_resolution_entry,
    target_images,
    target_cameras,
    target_gs,
    output_dir,
    logger,
    device,
    eval_chunk_size,
    alignment,
    attribute_init,
    emd_eps,
    emd_iters,
    input_resolution,
    target_resolution,
    matching_cache_root=None,
    force_matching_fit=False,
    matching_config=None,
    matching_optimizer_factory=None,
    input_images=None,
    input_cameras=None,
    write_artifacts=True,
):
    if alignment in {"emd", "random"}:
        densification_result = densification.build_densified_input(
            input_factor_dict=input_resolution_entry,
            target_factor_dict=target_resolution_entry,
            alignment="emd" if alignment == "emd" else "none",
            attribute_init=attribute_init,
            emd_eps=emd_eps,
            emd_iters=emd_iters,
            device=device,
            return_stages=write_artifacts,
        )
        if write_artifacts:
            aligned_input_gs, stage_gs = densification_result
        else:
            aligned_input_gs = densification_result
        if alignment == "random":
            permutation = torch.randperm(
                aligned_input_gs["means"].shape[0], device=device
            )
            aligned_input_gs = {
                key: value[permutation].clone()
                for key, value in aligned_input_gs.items()
            }

        if not write_artifacts:
            return aligned_input_gs, target_gs, {
                "cache_status": "not_applicable"
            }

        stage_gs["03_input_high_res_gs.ply"] = aligned_input_gs
        densification.save_densification_stages(
            output_dir=output_dir,
            low_res_gs=stage_gs["00_low_res_gs.ply"],
            interpolated_gs=stage_gs["01_interpolated_high_res_gs.ply"],
            gt_high_res_gs=stage_gs["02_gt_high_res_gs.ply"],
            input_high_res_gs=stage_gs["03_input_high_res_gs.ply"],
        )
        stage_metrics = write_densify_stage_render_metrics(
            output_dir=output_dir,
            stage_gs=stage_gs,
            images=target_images,
            cameras=target_cameras,
            chunk_size=eval_chunk_size,
            device=device,
        )
        for ply_name, metrics in stage_metrics.items():
            logger.info(
                "Alignment init %s: %s",
                ply_name,
                " ".join(
                    f"{key}: {value:.4f}"
                    for key, value in metrics.items()
                ),
            )
        return aligned_input_gs, target_gs, {
            "cache_status": "not_applicable"
        }

    if alignment == "fit_lr_to_hr":
        fit_source_gs = matching.build_matching_source(
            input_resolution_entry, target_resolution_entry, device
        )
        aligned_target_gs, cache = matching.get_or_fit_matching_target(
            source_gs=fit_source_gs,
            target_images=target_images,
            target_cameras=target_cameras,
            pre_matching_root=matching_cache_root,
            scene_name=scene["scene_name"],
            input_resolution=input_resolution,
            target_resolution=target_resolution,
            logger=logger,
            config=matching_config,
            optimizer_factory=matching_optimizer_factory,
            force_pre_matching=True,
        )
        aligned_input_gs = fit_source_gs
        artifact_images = target_images
        artifact_cameras = target_cameras
        fitted_artifact_gs = aligned_target_gs
    elif alignment == "fit_hr_to_lr":
        if (input_images is None) != (input_cameras is None):
            raise ValueError(
                "input_images and input_cameras must be supplied together"
            )
        if input_images is None:
            input_images, _, input_cameras = dataset.load_resolution_views(
                input_resolution_entry
            )
        high_res_in_low_res_frame = matching.build_matching_source(
            target_resolution_entry, input_resolution_entry, device
        )
        fitted_high_res_in_low_res_frame, cache = (
            matching.get_or_fit_matching_target(
                source_gs=high_res_in_low_res_frame,
                target_images=input_images,
                target_cameras=input_cameras,
                pre_matching_root=matching_cache_root,
                scene_name=scene["scene_name"],
                input_resolution=target_resolution,
                target_resolution=input_resolution,
                logger=logger,
                config=matching_config,
                optimizer_factory=matching_optimizer_factory,
                force_pre_matching=True,
            )
        )
        if "scaler" not in input_resolution_entry and "scaler" not in target_resolution_entry:
            aligned_input_gs = fitted_high_res_in_low_res_frame
        else:
            aligned_input_gs = gs_utils.convert_gaussian_frame(
                fitted_high_res_in_low_res_frame,
                input_resolution_entry["scaler"],
                target_resolution_entry["scaler"],
            )
        aligned_target_gs = target_gs
        fit_source_gs = high_res_in_low_res_frame
        fitted_artifact_gs = fitted_high_res_in_low_res_frame
        artifact_images = input_images
        artifact_cameras = input_cameras
    else:
        raise ValueError(f"Unsupported alignment mode: {alignment}")

    alignment_info = {
        "cache_status": cache["status"],
        "cache_path": cache["checkpoint_path"],
        "matching_steps": matching_config["total_steps"],
        "matching_images_per_step": matching_config["image_per_step"],
    }
    if not write_artifacts:
        return aligned_input_gs, aligned_target_gs, alignment_info

    artifact_chunk_size = dataset.image_per_scene or len(artifact_images)
    artifact_metrics = matching.save_matching_artifacts(
        output_dir,
        fit_source_gs,
        fitted_artifact_gs,
        artifact_images,
        artifact_cameras,
        artifact_chunk_size,
        device,
    )
    for ply_name, metrics in artifact_metrics.items():
        logger.info(
            "Fitted alignment %s: %s",
            ply_name,
            " ".join(
                f"{key}: {value:.4f}" for key, value in metrics.items()
            ),
        )
    return aligned_input_gs, aligned_target_gs, alignment_info

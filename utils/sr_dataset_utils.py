import gin
import torch

from dataset.GS_SR import SplatFactoSRDataset
from utils import gpu_utils


def build_dataset(scope="train_dataset") -> SplatFactoSRDataset:
    with gin.config_scope(scope):
        return SplatFactoSRDataset()


def scene_name_from_dataset(dataset, idx):
    return dataset.folders[idx]["scene_name"]


def find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for idx in range(len(dataset.folders)):
        if scene_name_from_dataset(dataset, idx) == scene_name:
            return idx
    raise ValueError(f"Scene {scene_name!r} is absent or was filtered from the dataset")


def _scaler_tensor(scaler, name, reference):
    return torch.as_tensor(
        getattr(scaler, name), device=reference.device, dtype=reference.dtype
    )


def _move_source_to_target_frame(source_entry, target_entry, device):
    source = gpu_utils.move_to_device(source_entry["gs_params"], device)
    source_scaler = source_entry["scaler"]
    target_scaler = target_entry["scaler"]
    source_scale = _scaler_tensor(source_scaler, "scale_", source["means"])
    source_translation = _scaler_tensor(source_scaler, "trans_", source["means"])
    target_scale = _scaler_tensor(target_scaler, "scale_", source["means"])
    target_translation = _scaler_tensor(target_scaler, "trans_", source["means"])

    raw_means = (source["means"] - source_translation) / source_scale
    converted = {
        "means": raw_means * target_scale + target_translation,
    }
    for key, value in source.items():
        if key == "means":
            continue
        if key == "scales":
            converted[key] = value - torch.log(source_scale) + torch.log(target_scale)
        else:
            converted[key] = value.clone()
    return converted


def build_precomputed_fit_pair(
    scene,
    input_resolution_entry,
    target_resolution_entry,
    input_resolution,
    target_resolution,
    device,
):
    source_gs = _move_source_to_target_frame(
        input_resolution_entry, target_resolution_entry, device
    )
    fit_entry = scene["fit_lr_to_hr"]
    available_pair = (
        int(fit_entry["source_resolution"]),
        int(fit_entry["target_resolution"]),
    )
    requested_pair = (int(input_resolution), int(target_resolution))
    if available_pair != requested_pair:
        raise ValueError(
            f"Precomputed fit supports {available_pair[0]}->{available_pair[1]}, "
            f"but {requested_pair[0]}->{requested_pair[1]} was requested"
        )
    target_gs = gpu_utils.move_to_device(fit_entry["gs_params"], device)
    if set(source_gs) != set(target_gs):
        raise ValueError("Precomputed fit attributes do not match the source")
    for key, source_value in source_gs.items():
        if target_gs[key].shape != source_value.shape:
            raise ValueError(f"Precomputed fit is not identity-paired for {key}")
    provenance = {
        "status": "dataset_precomputed",
        "checkpoint_path": fit_entry["checkpoint_path"],
    }
    return source_gs, target_gs, provenance

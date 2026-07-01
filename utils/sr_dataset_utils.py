import gin
import numpy as np
import torch

from dataset.GS_multi import SplatFactoMultiLevelDataset


def build_dataset() -> SplatFactoMultiLevelDataset:
    with gin.config_scope("train_dataset"):
        return SplatFactoMultiLevelDataset()


def scene_name_from_dataset(dataset, idx):
    return dataset.folders[idx]["scene_name"]


def find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for idx in range(len(dataset.folders)):
        if scene_name_from_dataset(dataset, idx) == scene_name:
            return idx
    return 0


def build_split_payload(dataset, scene_idx, scene_name, factor_entry, split):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    total_num = len(meta["camera_to_worlds"])
    if split in ["train", "test"]:
        cam_ids = np.arange(total_num)
    else:
        raise ValueError(f"Unsupported split: {split}")

    images = [dataset.read_image(imgs_path[i], background=background) for i in cam_ids]
    images_name = [imgs_name[i] for i in cam_ids]
    camera_to_worlds = meta["camera_to_worlds"][cam_ids]

    cameras = {
        "camera_to_worlds": torch.as_tensor(camera_to_worlds).float(),
        "fx": torch.as_tensor(meta["fx"]).float(),
        "fy": torch.as_tensor(meta["fy"]).float(),
        "cx": torch.as_tensor(meta["cx"]).float(),
        "cy": torch.as_tensor(meta["cy"]).float(),
        "width": torch.as_tensor(meta["width"]).float(),
        "height": torch.as_tensor(meta["height"]).float(),
        "background_color": background,
    }

    return {
        "gs_params": factor_entry["gs_params"],
        "images": images,
        "images_name": images_name,
        "cameras": cameras,
        "scene_idx": scene_idx,
        "scene_name": scene_name,
    }

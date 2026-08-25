import csv
import glob
import os
import re

import numpy as np
import torch
from PIL import Image

from dataset import colmap_utils


GSPLAT_KEY_MAP = {
    "means": "means",
    "sh0": "features_dc",
    "shN": "features_rest",
    "opacities": "opacities",
    "scales": "scales",
    "quats": "quats",
}
GSPLAT_CHECKPOINT_PATTERN = re.compile(r"ckpt_(\d+)_rank0\.pt$")


def read_scene_names(scene_list):
    with open(scene_list, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "scene_id" not in reader.fieldnames:
            raise ValueError(
                f"Scene manifest must be a CSV with a scene_id column: {scene_list}"
            )
        scene_names = [
            row["scene_id"].strip()
            for row in reader
            if row.get("scene_id", "").strip()
        ]
    if not scene_names:
        raise ValueError(f"Scene manifest is empty: {scene_list}")
    return scene_names


def resolution_paths(split_root, scene_name, resolution):
    resolution_root = os.path.join(split_root, str(resolution))
    colmap_dir = os.path.join(resolution_root, "colmap", scene_name)
    gsplat_dir = os.path.join(resolution_root, "gsplat", scene_name)
    return {
        "colmap_dir": colmap_dir,
        "image_dir": os.path.join(colmap_dir, "images"),
        "sparse_dir": os.path.join(colmap_dir, "sparse", "0"),
        "gsplat_dir": gsplat_dir,
    }


def fitted_paths(fitted_root, source_resolution, scene_name):
    return {
        "gsplat_dir": os.path.join(
            fitted_root,
            str(source_resolution),
            "gsplat",
            scene_name,
        )
    }


def latest_gsplat_checkpoint(gsplat_dir):
    candidates = []
    pattern = os.path.join(gsplat_dir, "ckpts", "ckpt_*_rank0.pt")
    for path in glob.glob(pattern):
        match = GSPLAT_CHECKPOINT_PATTERN.match(os.path.basename(path))
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"{gsplat_dir} does not have ckpts/ckpt_*_rank0.pt"
        )
    return max(candidates, key=lambda item: item[0])[1]


def colmap_model_paths(sparse_dir):
    text_paths = (
        os.path.join(sparse_dir, "cameras.txt"),
        os.path.join(sparse_dir, "images.txt"),
    )
    binary_paths = (
        os.path.join(sparse_dir, "cameras.bin"),
        os.path.join(sparse_dir, "images.bin"),
    )
    if all(os.path.isfile(path) for path in text_paths):
        return text_paths
    if all(os.path.isfile(path) for path in binary_paths):
        return binary_paths
    raise FileNotFoundError(f"Missing COLMAP cameras/images model in {sparse_dir}")


def scene_problem(scene_info, resolutions):
    try:
        for resolution in resolutions:
            paths = scene_info["resolution_paths"][resolution]
            if not os.path.isdir(paths["image_dir"]):
                return f"resolution={resolution}:missing_image_dir"
            colmap_model_paths(paths["sparse_dir"])
            latest_gsplat_checkpoint(paths["gsplat_dir"])
        latest_gsplat_checkpoint(scene_info["fit_lr_to_hr_paths"]["gsplat_dir"])
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    return None


def load_gsplat(gsplat_dir):
    checkpoint_path = latest_gsplat_checkpoint(gsplat_dir)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load gsplat checkpoint {checkpoint_path}"
        ) from exc
    splats = checkpoint.get("splats") if isinstance(checkpoint, dict) else None
    if not isinstance(splats, dict):
        raise ValueError(f"Checkpoint has no splats mapping: {checkpoint_path}")
    missing = sorted(set(GSPLAT_KEY_MAP) - set(splats))
    if missing:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing {missing}")

    gs_params = {
        target_key: splats[source_key].detach().float().cpu()
        for source_key, target_key in GSPLAT_KEY_MAP.items()
    }
    gs_params["features_dc"] = gs_params["features_dc"].squeeze(1)
    if gs_params["opacities"].ndim == 1:
        gs_params["opacities"] = gs_params["opacities"].unsqueeze(-1)
    count = gs_params["means"].shape[0]
    for key, value in gs_params.items():
        if value.shape[0] != count:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has inconsistent {key} count"
            )
    return gs_params, checkpoint_path


def load_colmap_views(colmap_dir):
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    camera_path, image_path = colmap_model_paths(sparse_dir)
    if camera_path.endswith(".txt"):
        cameras = colmap_utils.read_cameras_text(camera_path)
        images = colmap_utils.read_images_text(image_path)
    else:
        cameras = colmap_utils.read_cameras_binary(camera_path)
        images = colmap_utils.read_images_binary(image_path)
    if len(cameras) != 1:
        raise ValueError(f"Only one COLMAP camera is supported: {colmap_dir}")
    camera = colmap_utils.parse_colmap_camera_params(next(iter(cameras.values())))
    if camera["camera_model"] not in ("SIMPLE_PINHOLE", "PINHOLE"):
        raise ValueError(
            f"Unsupported COLMAP camera model {camera['camera_model']}: {colmap_dir}"
        )

    camera_to_worlds = []
    image_paths = []
    for image in sorted(images.values(), key=lambda item: item.name):
        rotation = colmap_utils.qvec2rotmat(image.qvec)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = image.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        camera_to_world[:3, 1:3] *= -1
        camera_to_worlds.append(camera_to_world.astype(np.float32))
        image_paths.append(os.path.join(colmap_dir, "images", image.name))
    if not camera_to_worlds:
        raise ValueError(f"COLMAP scene has zero registered images: {colmap_dir}")
    missing_images = [path for path in image_paths if not os.path.isfile(path)]
    if missing_images:
        raise FileNotFoundError(f"Missing registered image {missing_images[0]}")
    return {
        "camera_to_worlds": torch.from_numpy(np.stack(camera_to_worlds)),
        "fx": torch.tensor(camera["fl_x"], dtype=torch.float32),
        "fy": torch.tensor(camera["fl_y"], dtype=torch.float32),
        "cx": torch.tensor(camera["cx"], dtype=torch.float32),
        "cy": torch.tensor(camera["cy"], dtype=torch.float32),
        "width": torch.tensor(camera["w"], dtype=torch.float32),
        "height": torch.tensor(camera["h"], dtype=torch.float32),
    }, image_paths


def read_image(path, background):
    image = np.asarray(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0
    mask = None
    if "real" in path.lower():
        mask_path = path.replace("images", "masks")
        if os.path.exists(mask_path):
            mask = torch.from_numpy(
                np.asarray(Image.open(mask_path)).astype(image.dtype) / 255.0
            )
    image = torch.from_numpy(image)
    if image.shape[2] == 4:
        alpha = image[:, :, -1:]
        return image[:, :, :3] * alpha + background * (1.0 - alpha)
    if mask is not None:
        image_rgb = image * mask[..., None] + background * (1.0 - mask[..., None])
        return torch.concat([image_rgb, mask[..., None]], axis=-1)
    return image

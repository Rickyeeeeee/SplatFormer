import csv
import glob
import os
import random
import re
from typing import Optional, Sequence

import gin
import numpy as np
import torch
from PIL import Image

from dataset import colmap_utils
from utils.transform_utils import MinMaxScaler, remove_outliers


GSPLAT_KEY_MAP = {
    "means": "means",
    "sh0": "features_dc",
    "shN": "features_rest",
    "opacities": "opacities",
    "scales": "scales",
    "quats": "quats",
}
GSPLAT_CHECKPOINT_PATTERN = re.compile(r"ckpt_(\d+)_rank0\.pt$")


@gin.configurable
class SplatFactoSRDataset(torch.utils.data.IterableDataset):
    """Load native-resolution and identity-paired fitted gsplat scenes."""

    def __init__(
        self,
        train_or_test: str,
        dataset_root: str,
        scene_list: str,
        image_per_scene: Optional[int],
        remove_outlier_ndevs: float,
        max_gs_num: int,
        split_across_gpus: bool,
        resolutions: Sequence[int] = (128, 512),
        background_color: Sequence[int] = (0, 0, 0),
        dataset_name: str = "objaverse",
        fit_source_resolution: int = 128,
        fit_target_resolution: int = 512,
    ):
        if train_or_test not in ("train", "test"):
            raise ValueError("train_or_test must be either 'train' or 'test'")
        if not resolutions:
            raise ValueError("resolutions must contain at least one resolution")

        self.train_or_test = train_or_test
        self.dataset_root = os.fspath(dataset_root)
        self.scene_list = os.fspath(scene_list)
        self.image_per_scene = image_per_scene
        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.max_gs_num = max_gs_num
        self.split_across_gpus = split_across_gpus
        self.resolutions = [int(resolution) for resolution in resolutions]
        self.primary_resolution = self.resolutions[0]
        self.background_color = background_color
        self.dataset_name = dataset_name
        self.fit_source_resolution = int(fit_source_resolution)
        self.fit_target_resolution = int(fit_target_resolution)
        if self.fit_source_resolution not in self.resolutions:
            raise ValueError("fit_source_resolution must be included in resolutions")
        if self.fit_target_resolution not in self.resolutions:
            raise ValueError("fit_target_resolution must be included in resolutions")

        self.filtered_scenes = []
        self.folders = self._load_scene_manifest()
        if self.train_or_test == "test":
            self.remaining_scenes = self._test_process_split()
        else:
            self.counter = 0

    @property
    def split_root(self):
        return os.path.join(
            self.dataset_root, f"{self.train_or_test}-set", self.dataset_name
        )

    @property
    def fitted_root(self):
        return os.path.join(
            self.dataset_root,
            f"{self.train_or_test}-set-4x-up",
            self.dataset_name,
        )

    def _read_scene_names(self):
        with open(self.scene_list, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "scene_id" not in reader.fieldnames:
                raise ValueError(
                    f"Scene manifest must be a CSV with a scene_id column: {self.scene_list}"
                )
            scene_names = [
                row["scene_id"].strip()
                for row in reader
                if row.get("scene_id", "").strip()
            ]
        if not scene_names:
            raise ValueError(f"Scene manifest is empty: {self.scene_list}")
        return scene_names

    def _resolution_paths(self, scene_name, resolution):
        resolution_root = os.path.join(self.split_root, str(resolution))
        colmap_dir = os.path.join(resolution_root, "colmap", scene_name)
        gsplat_dir = os.path.join(resolution_root, "gsplat", scene_name)
        return {
            "colmap_dir": colmap_dir,
            "image_dir": os.path.join(colmap_dir, "images"),
            "sparse_dir": os.path.join(colmap_dir, "sparse", "0"),
            "gsplat_dir": gsplat_dir,
        }

    def _fitted_paths(self, scene_name):
        return {
            "gsplat_dir": os.path.join(
                self.fitted_root,
                str(self.fit_source_resolution),
                "gsplat",
                scene_name,
            )
        }

    @staticmethod
    def _latest_checkpoint(gsplat_dir):
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

    @staticmethod
    def _camera_model_paths(sparse_dir):
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

    def _scene_problem(self, scene_info):
        try:
            for resolution in self.resolutions:
                paths = scene_info["resolution_paths"][resolution]
                if not os.path.isdir(paths["image_dir"]):
                    return f"resolution={resolution}:missing_image_dir"
                self._camera_model_paths(paths["sparse_dir"])
                self._latest_checkpoint(paths["gsplat_dir"])
            self._latest_checkpoint(scene_info["fit_lr_to_hr_paths"]["gsplat_dir"])
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        return None

    def _load_scene_manifest(self):
        folders = []
        for scene_name in self._read_scene_names():
            scene_info = {
                "scene_name": scene_name,
                "resolution_paths": {
                    resolution: self._resolution_paths(scene_name, resolution)
                    for resolution in self.resolutions
                },
                "fit_lr_to_hr_paths": self._fitted_paths(scene_name),
            }
            problem = self._scene_problem(scene_info)
            if problem is None:
                folders.append(scene_info)
            else:
                self.filtered_scenes.append(
                    {"scene_name": scene_name, "reason": problem}
                )

        print(
            f"[SplatFactoSRDataset] using {len(folders)} scenes; "
            f"filtered {len(self.filtered_scenes)} incomplete scenes",
            flush=True,
        )
        for item in self.filtered_scenes[:20]:
            print(
                f"[SplatFactoSRDataset] filtered {item['scene_name']}: "
                f"{item['reason']}",
                flush=True,
            )
        if len(self.filtered_scenes) > 20:
            print(
                f"[SplatFactoSRDataset] ... and "
                f"{len(self.filtered_scenes) - 20} more",
                flush=True,
            )
        if not folders:
            raise ValueError(
                f"No complete scenes remain after filtering {self.scene_list}"
            )
        return folders

    @staticmethod
    def _distributed_context():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _test_process_split(self):
        remaining_scenes = list(range(len(self.folders)))
        world_size, rank = self._distributed_context()
        chunk_size = len(remaining_scenes) // world_size
        if rank == world_size - 1:
            return remaining_scenes[rank * chunk_size :]
        return remaining_scenes[rank * chunk_size : (rank + 1) * chunk_size]

    def get_thisworker_split(self, count):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return list(range(count))
        per_worker = count // worker_info.num_workers
        if worker_info.id == worker_info.num_workers - 1:
            return list(range(worker_info.id * per_worker, count))
        return list(
            range(worker_info.id * per_worker, (worker_info.id + 1) * per_worker)
        )

    def random_split_to_remaining(self):
        world_size, rank = self._distributed_context()
        permutation = np.random.RandomState(self.counter).permutation(len(self.folders))
        pad_num = (world_size - len(self.folders) % world_size) % world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])
        chunk_size = len(permutation) // world_size
        if rank == world_size - 1:
            process_scenes = permutation[rank * chunk_size :]
        else:
            process_scenes = permutation[rank * chunk_size : (rank + 1) * chunk_size]
        worker_indices = self.get_thisworker_split(len(process_scenes))
        self.remaining_scenes = [int(process_scenes[i]) for i in worker_indices]

    def refresh_remaining_training(self):
        if self.split_across_gpus:
            self.random_split_to_remaining()
        else:
            self.remaining_scenes = self.get_thisworker_split(len(self.folders))
            random.shuffle(self.remaining_scenes)
        self.counter += 1

    @gin.configurable
    def read_image(self, path, background):
        image = np.asarray(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0
        mask = None
        if "real" in path.lower():
            possible_mask_filename = path.replace("images", "masks")
            if os.path.exists(possible_mask_filename):
                mask = torch.from_numpy(
                    np.asarray(Image.open(possible_mask_filename)).astype(image.dtype)
                    / 255.0
                )
        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            alpha = image[:, :, -1:]
            image = image[:, :, :3] * alpha + background * (1.0 - alpha)
        elif mask is not None:
            image_rgb = image * mask[..., None] + background * (1.0 - mask[..., None])
            image = torch.concat([image_rgb, mask[..., None]], axis=-1)
        return image

    def _load_raw_gs(self, gsplat_dir):
        checkpoint_path = self._latest_checkpoint(gsplat_dir)
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

    @staticmethod
    def _finite_mask(gs_params):
        count = gs_params["means"].shape[0]
        mask = torch.ones(count, dtype=torch.bool)
        for value in gs_params.values():
            finite = torch.isfinite(value)
            for dim in range(value.ndim - 1, 0, -1):
                finite = finite.all(dim=dim)
            mask &= finite
        return mask

    @staticmethod
    def _select_gs(gs_params, mask):
        return {key: value[mask] for key, value in gs_params.items()}

    def _base_mask(self, gs_params):
        mask = self._finite_mask(gs_params)
        if self.remove_outlier_ndevs > 0:
            valid_indices = mask.nonzero(as_tuple=False).squeeze(1)
            _, inlier_mask = remove_outliers(
                gs_params["means"][mask], n_devs=self.remove_outlier_ndevs
            )
            outlier_mask = torch.zeros_like(mask)
            outlier_mask[valid_indices[inlier_mask]] = True
            mask &= outlier_mask
        selected_indices = mask.nonzero(as_tuple=False).squeeze(1)
        if selected_indices.numel() > self.max_gs_num:
            keep = torch.zeros_like(mask)
            keep[selected_indices[: self.max_gs_num]] = True
            mask &= keep
        if not mask.any():
            raise ValueError("Gaussian filtering removed every splat")
        return mask

    @staticmethod
    def _normalize_gs(gs_params, scaler=None):
        normalized = {key: value.clone() for key, value in gs_params.items()}
        if scaler is None:
            scaler = MinMaxScaler()
            normalized["means"] = scaler.fit_transform(normalized["means"])
        else:
            normalized["means"] = scaler.transform(normalized["means"])
        normalized["scales"] = normalized["scales"] + torch.log(scaler.scale_)
        return normalized, scaler

    def _process_native_gs(self, raw_gs):
        selected = self._select_gs(raw_gs, self._base_mask(raw_gs))
        return self._normalize_gs(selected)

    def _process_fitted_pair(self, source_raw, fitted_raw, target_scaler):
        if set(source_raw) != set(fitted_raw):
            raise ValueError("Source and fitted GS attributes do not match")
        for key in source_raw:
            if source_raw[key].shape != fitted_raw[key].shape:
                raise ValueError(
                    f"Source/fitted shape mismatch for {key}: "
                    f"{tuple(source_raw[key].shape)} vs {tuple(fitted_raw[key].shape)}"
                )
        shared_mask = self._base_mask(source_raw) & self._finite_mask(fitted_raw)
        if not shared_mask.any():
            raise ValueError("Source/fitted shared filtering removed every splat")
        source = self._select_gs(source_raw, shared_mask)
        fitted = self._select_gs(fitted_raw, shared_mask)
        source, source_scaler = self._normalize_gs(source)
        fitted, _ = self._normalize_gs(fitted, scaler=target_scaler)
        return source, source_scaler, fitted

    def load_images_cameras_fromcolmap(self, colmap_dir):
        sparse_dir = os.path.join(colmap_dir, "sparse", "0")
        camera_path, image_path = self._camera_model_paths(sparse_dir)
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
        meta = {
            "camera_to_worlds": torch.from_numpy(np.stack(camera_to_worlds)),
            "fx": torch.tensor(camera["fl_x"], dtype=torch.float32),
            "fy": torch.tensor(camera["fl_y"], dtype=torch.float32),
            "cx": torch.tensor(camera["cx"], dtype=torch.float32),
            "cy": torch.tensor(camera["cy"], dtype=torch.float32),
            "width": torch.tensor(camera["w"], dtype=torch.float32),
            "height": torch.tensor(camera["h"], dtype=torch.float32),
        }
        return meta, image_paths

    def load_scene(self, scene_idx):
        scene_info = self.folders[scene_idx]
        raw_by_resolution = {}
        checkpoint_by_resolution = {}
        for resolution in self.resolutions:
            raw_by_resolution[resolution], checkpoint_by_resolution[resolution] = (
                self._load_raw_gs(
                    scene_info["resolution_paths"][resolution]["gsplat_dir"]
                )
            )
        fitted_raw, fitted_checkpoint = self._load_raw_gs(
            scene_info["fit_lr_to_hr_paths"]["gsplat_dir"]
        )

        processed = {}
        target_gs, target_scaler = self._process_native_gs(
            raw_by_resolution[self.fit_target_resolution]
        )
        source_gs, source_scaler, fitted_gs = self._process_fitted_pair(
            raw_by_resolution[self.fit_source_resolution],
            fitted_raw,
            target_scaler,
        )
        processed[self.fit_source_resolution] = (source_gs, source_scaler)
        processed[self.fit_target_resolution] = (target_gs, target_scaler)
        for resolution in self.resolutions:
            if resolution not in processed:
                processed[resolution] = self._process_native_gs(
                    raw_by_resolution[resolution]
                )

        resolution_data = {}
        reference_names = None
        for resolution in self.resolutions:
            paths = scene_info["resolution_paths"][resolution]
            gs_params, scaler = processed[resolution]
            meta, image_paths = self.load_images_cameras_fromcolmap(paths["colmap_dir"])
            meta["camera_to_worlds"][:, :3, -1] = scaler.transform(
                meta["camera_to_worlds"][:, :3, -1]
            )
            image_names = [os.path.basename(path) for path in image_paths]
            if len(image_names) != len(meta["camera_to_worlds"]):
                raise ValueError(
                    f"Scene {scene_info['scene_name']} resolution {resolution}: "
                    "image count does not match pose count"
                )
            if reference_names is None:
                reference_names = image_names
            elif image_names != reference_names:
                raise ValueError(
                    f"Scene {scene_info['scene_name']}: image names do not match "
                    "across resolutions"
                )
            resolution_data[resolution] = {
                "gs_params": gs_params,
                "meta": meta,
                "imgs_path": image_paths,
                "imgs_name": image_names,
                "scaler": scaler,
                "checkpoint_path": checkpoint_by_resolution[resolution],
            }

        return {
            "idx": scene_idx,
            "scene_name": scene_info["scene_name"],
            "resolution_data": resolution_data,
            "fit_lr_to_hr": {
                "source_resolution": self.fit_source_resolution,
                "target_resolution": self.fit_target_resolution,
                "gs_params": fitted_gs,
                "scaler": target_scaler,
                "coordinate_frame": "target_resolution",
                "checkpoint_path": fitted_checkpoint,
            },
        }

    def build_background(self):
        if self.background_color == "random":
            return torch.rand(3)
        return torch.tensor(self.background_color, dtype=torch.float32) / 255.0

    def load_resolution_views(
        self, resolution_entry, camera_ids=None, background=None
    ):
        meta = resolution_entry["meta"]
        total_views = len(meta["camera_to_worlds"])
        if total_views == 0:
            raise ValueError("Resolution entry has zero views")
        camera_ids = list(range(total_views)) if camera_ids is None else list(camera_ids)
        if background is None:
            background = self.build_background()
        images = [
            self.read_image(resolution_entry["imgs_path"][i], background)
            for i in camera_ids
        ]
        image_names = [resolution_entry["imgs_name"][i] for i in camera_ids]
        cameras = {
            "camera_to_worlds": meta["camera_to_worlds"][camera_ids],
            "fx": meta["fx"],
            "fy": meta["fy"],
            "cx": meta["cx"],
            "cy": meta["cy"],
            "width": meta["width"],
            "height": meta["height"],
            "background_color": background,
        }
        return images, image_names, cameras

    def _prepare_resolution_payload(self, resolution_entry, background, camera_ids):
        images, image_names, cameras = self.load_resolution_views(
            resolution_entry, camera_ids=camera_ids, background=background
        )
        return {
            "gs_params": resolution_entry["gs_params"],
            "scaler": resolution_entry["scaler"],
            "images": images,
            "cameras": cameras,
            "images_name": image_names,
            "checkpoint_path": resolution_entry["checkpoint_path"],
        }

    def __iter__(self):
        if self.train_or_test == "train":
            self.refresh_remaining_training()
        while self.remaining_scenes:
            scene_idx = self.remaining_scenes.pop(0)
            if self.train_or_test == "train" and not self.remaining_scenes:
                self.refresh_remaining_training()
            scene = self.load_scene(scene_idx)
            primary_entry = scene["resolution_data"][self.primary_resolution]
            total_views = len(primary_entry["meta"]["camera_to_worlds"])
            background = self.build_background()
            if self.train_or_test == "train":
                sample_count = total_views
                if self.image_per_scene is not None:
                    sample_count = min(self.image_per_scene, total_views)
                camera_ids = np.random.permutation(total_views)[:sample_count]
            else:
                if self.background_color == "random":
                    raise ValueError("Test background_color cannot be random")
                camera_ids = np.arange(total_views)
            multiresolution = {
                resolution: self._prepare_resolution_payload(
                    scene["resolution_data"][resolution], background, camera_ids
                )
                for resolution in self.resolutions
            }
            yield {
                "scene_idx": scene["idx"],
                "scene_name": scene["scene_name"],
                "resolutions": self.resolutions,
                "multiresolution": multiresolution,
                "fit_lr_to_hr": scene["fit_lr_to_hr"],
            }

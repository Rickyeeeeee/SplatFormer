import glob
import os
import pickle
import random
from typing import Optional, Sequence

import gin
import numpy as np
import torch
from PIL import Image

from utils.transform_utils import MinMaxScaler, remove_outliers


CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"


@gin.configurable
class SplatFactoSRDataset(torch.utils.data.IterableDataset):
    """Load matching native-resolution Gaussian splatting scenes."""

    def __init__(
        self,
        train_or_test: str,
        dataset_root: str,
        scene_list: str,
        image_per_scene: Optional[int],
        remove_outlier_ndevs: float,
        max_gs_num: int,
        split_across_gpus: bool,
        load_pose_src: str = "nerfstudio",
        resolutions: Sequence[int] = (512, 128),
        background_color: Sequence[int] = (0, 0, 0),
    ):
        if train_or_test not in ("train", "test"):
            raise ValueError("train_or_test must be either 'train' or 'test'")
        if load_pose_src != "nerfstudio":
            raise ValueError("SplatFactoSRDataset only supports load_pose_src='nerfstudio'")
        if len(resolutions) == 0:
            raise ValueError("resolutions must contain at least one resolution")

        self.train_or_test = train_or_test
        self.dataset_root = os.fspath(dataset_root)
        self.scene_list = os.fspath(scene_list)
        self.image_per_scene = image_per_scene
        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.max_gs_num = max_gs_num
        self.split_across_gpus = split_across_gpus
        self.load_pose_src = load_pose_src
        self.resolutions = list(resolutions)
        self.primary_resolution = self.resolutions[0]
        self.background_color = background_color

        self.folders = self._load_scene_manifest()

        if self.train_or_test == "test":
            self.remaining_scenes = self._test_process_split()
        else:
            self.counter = 0

    def _load_scene_manifest(self):
        with open(self.scene_list, "r", encoding="utf-8") as f:
            scene_names = [line.strip() for line in f if line.strip()]
        if len(scene_names) == 0:
            raise ValueError(f"Scene list is empty: {self.scene_list}")

        folders = []
        for scene_name in scene_names:
            resolution_paths = {}
            for resolution in self.resolutions:
                resolution_root = os.path.join(self.dataset_root, str(resolution))
                colmap_dir = os.path.join(resolution_root, "colmap", scene_name)
                nerfstudio_dir = os.path.join(
                    resolution_root, "nerfstudio", scene_name, "splatfacto"
                )
                resolution_paths[resolution] = {
                    "colmap_dir": colmap_dir,
                    "image_dir": os.path.join(colmap_dir, "images"),
                    "nerfstudio_dir": nerfstudio_dir,
                }
            folders.append(
                {"scene_name": scene_name, "resolution_paths": resolution_paths}
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
        try:
            pil_image = Image.open(path)
        except Exception:
            print(f"Warning: {path} cannot be opened")
            raise

        image = np.asarray(pil_image, dtype=np.uint8).astype(np.float32) / 255.0
        mask = None
        if "real" in path.lower():
            possible_mask_filename = path.replace("images", "masks")
            if os.path.exists(possible_mask_filename):
                mask = np.asarray(Image.open(possible_mask_filename)).astype(image.dtype) / 255.0
                mask = torch.from_numpy(mask)

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            alpha = image[:, :, -1:]
            image = image[:, :, :3] * alpha + background * (1.0 - alpha)
        elif mask is not None:
            image_rgb = image * mask[..., None] + background * (1.0 - mask[..., None])
            image = torch.concat([image_rgb, mask[..., None]], axis=-1)
        return image

    @staticmethod
    def _checkpoint_sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.split("-")[-1])

    def load_gs_params_fromnerfstudio(self, nerfstudio_dir, scene_idx):
        input_features = list(gin.query_parameter("FeaturePredictor.input_features"))
        if gin.query_parameter("training.pretrain_steps") > 0:
            input_features += list(
                gin.query_parameter("create_pseudo_target.take_from_input")
            )

        checkpoint_paths = sorted(
            glob.glob(os.path.join(nerfstudio_dir, "nerfstudio_models", "step-*.ckpt")),
            key=self._checkpoint_sort_key,
        )
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(
                f"{nerfstudio_dir} does not have nerfstudio_models/step-*.ckpt"
            )
        checkpoint_path = checkpoint_paths[-1]

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            scene_name = self.folders[scene_idx]["scene_name"]
            raise RuntimeError(
                f"Failed to load GS checkpoint for scene_idx={scene_idx} "
                f"scene_name={scene_name} checkpoint={checkpoint_path}"
            ) from exc

        checkpoint = {
            key.replace("_model.gauss_params.", ""): value
            for key, value in checkpoint.items()
            if "gauss_params" in key
        }
        gs_params = {key: checkpoint[key] for key in set(input_features)}

        select = torch.ones(gs_params["means"].shape[0], dtype=torch.bool)
        for key, value in gs_params.items():
            if key == "features_rest":
                select &= ~torch.isnan(value.sum(dim=1)).any(dim=1)
            else:
                select &= ~torch.isnan(value).any(dim=1)
        gs_params = {key: value[select] for key, value in gs_params.items()}

        if self.remove_outlier_ndevs > 0:
            _, inlier_mask = remove_outliers(
                gs_params["means"], n_devs=self.remove_outlier_ndevs
            )
            gs_params = {
                key: value[inlier_mask] for key, value in gs_params.items()
            }

        if gs_params["means"].shape[0] > self.max_gs_num:
            gs_params = {
                key: value[: self.max_gs_num] for key, value in gs_params.items()
            }

        scaler = MinMaxScaler()
        gs_params["means"] = scaler.fit_transform(gs_params["means"])
        gs_params["scales"] = gs_params["scales"] + torch.log(scaler.scale_)

        finite_scales = ~torch.isinf(gs_params["scales"]).any(dim=1)
        in_range = torch.all(
            (gs_params["means"] >= 0) & (gs_params["means"] <= 1), dim=1
        )
        valid_mask = finite_scales & in_range
        for key in gs_params:
            gs_params[key] = gs_params[key][valid_mask]
            if torch.isnan(gs_params[key]).any():
                print(f"Warning: {key} contains nan", nerfstudio_dir)
        return gs_params, scaler

    @staticmethod
    def _tensorize_camera_meta(meta):
        for key in (
            "train_camera_to_worlds",
            "test_camera_to_worlds",
            "camera_to_worlds",
            "fx",
            "fy",
            "cx",
            "cy",
            "width",
            "height",
        ):
            if key in meta:
                meta[key] = torch.as_tensor(meta[key], dtype=torch.float32)
        return meta

    def load_images_cameras_fromnerfstudio(self, nerfstudio_dir, image_dir):
        camera_path = os.path.join(nerfstudio_dir, CAMERA_METADATA_NAME)
        with open(camera_path, "rb") as f:
            meta = pickle.load(f)

        image_names = sorted(
            name for name in os.listdir(image_dir) if name.lower().endswith(".png")
        )
        image_paths = [os.path.join(image_dir, name) for name in image_names]
        meta["camera_to_worlds"] = meta["train_camera_to_worlds"]
        return self._tensorize_camera_meta(meta), image_paths

    def load_scene(self, scene_idx):
        scene_info = self.folders[scene_idx]
        resolution_data = {}
        reference_names = None

        for resolution in self.resolutions:
            paths = scene_info["resolution_paths"][resolution]
            gs_params, scaler = self.load_gs_params_fromnerfstudio(
                paths["nerfstudio_dir"], scene_idx
            )
            meta, image_paths = self.load_images_cameras_fromnerfstudio(
                paths["nerfstudio_dir"], paths["image_dir"]
            )
            meta["camera_to_worlds"][:, :3, -1] = scaler.transform(
                meta["camera_to_worlds"][:, :3, -1]
            )
            image_names = [os.path.basename(path) for path in image_paths]
            if len(image_names) != len(meta["camera_to_worlds"]):
                raise ValueError(
                    f"Scene {scene_info['scene_name']} resolution {resolution}: "
                    f"image count {len(image_names)} does not match pose count "
                    f"{len(meta['camera_to_worlds'])}"
                )
            if reference_names is None:
                reference_names = image_names
            elif image_names != reference_names:
                raise ValueError(
                    f"Scene {scene_info['scene_name']}: image names do not match "
                    f"between resolution {self.primary_resolution} and {resolution}"
                )

            resolution_data[resolution] = {
                "gs_params": gs_params,
                "meta": meta,
                "imgs_path": image_paths,
                "imgs_name": image_names,
                "scaler": scaler,
            }

        return {
            "idx": scene_idx,
            "scene_name": scene_info["scene_name"],
            "resolution_data": resolution_data,
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
        if camera_ids is None:
            camera_ids = list(range(total_views))
        else:
            camera_ids = list(camera_ids)
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
        }

    def __iter__(self):
        if self.train_or_test == "train":
            self.refresh_remaining_training()

        while len(self.remaining_scenes) > 0:
            scene_idx = self.remaining_scenes.pop(0)
            if self.train_or_test == "train" and len(self.remaining_scenes) == 0:
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
            }

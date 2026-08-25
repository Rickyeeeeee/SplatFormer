import os
import random
from typing import Optional, Sequence

import gin
import numpy as np
import torch

from dataset import gs_io
from dataset.gs_processing import GaussianProcessor


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

        self.processor = GaussianProcessor(remove_outlier_ndevs, max_gs_num)
        self.filtered_scenes = []
        self.folders = self.load_scene_manifest()
        if self.train_or_test == "test":
            self.remaining_scenes = self.test_process_split()
        else:
            self.counter = 0

    @classmethod
    def from_gin_scope(cls, scope):
        with gin.config_scope(scope):
            return cls()

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

    def scene_index(self, scene_name):
        if scene_name == "":
            return 0
        for index, scene in enumerate(self.folders):
            if scene["scene_name"] == scene_name:
                return index
        raise ValueError(
            f"Scene {scene_name!r} is absent or was filtered from the dataset"
        )

    def load_scene_manifest(self):
        folders = []
        for scene_name in gs_io.read_scene_names(self.scene_list):
            scene_info = {
                "scene_name": scene_name,
                "resolution_paths": {
                    resolution: gs_io.resolution_paths(
                        self.split_root, scene_name, resolution
                    )
                    for resolution in self.resolutions
                },
                "fit_lr_to_hr_paths": gs_io.fitted_paths(
                    self.fitted_root,
                    self.fit_source_resolution,
                    scene_name,
                ),
            }
            problem = gs_io.scene_problem(scene_info, self.resolutions)
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
    def distributed_context():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def test_process_split(self):
        remaining_scenes = list(range(len(self.folders)))
        world_size, rank = self.distributed_context()
        chunk_size = len(remaining_scenes) // world_size
        if rank == world_size - 1:
            return remaining_scenes[rank * chunk_size :]
        return remaining_scenes[rank * chunk_size : (rank + 1) * chunk_size]

    def worker_split(self, count):
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
        world_size, rank = self.distributed_context()
        permutation = np.random.RandomState(self.counter).permutation(len(self.folders))
        pad_num = (world_size - len(self.folders) % world_size) % world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])
        chunk_size = len(permutation) // world_size
        if rank == world_size - 1:
            process_scenes = permutation[rank * chunk_size :]
        else:
            process_scenes = permutation[rank * chunk_size : (rank + 1) * chunk_size]
        worker_indices = self.worker_split(len(process_scenes))
        self.remaining_scenes = [int(process_scenes[i]) for i in worker_indices]

    def refresh_remaining_training(self):
        if self.split_across_gpus:
            self.random_split_to_remaining()
        else:
            self.remaining_scenes = self.worker_split(len(self.folders))
            random.shuffle(self.remaining_scenes)
        self.counter += 1

    def load_scene(self, scene_idx):
        scene_info = self.folders[scene_idx]
        raw_by_resolution = {}
        checkpoint_by_resolution = {}
        for resolution in self.resolutions:
            gsplat_dir = scene_info["resolution_paths"][resolution]["gsplat_dir"]
            raw_by_resolution[resolution], checkpoint_by_resolution[resolution] = (
                gs_io.load_gsplat(gsplat_dir)
            )
        fitted_raw, fitted_checkpoint = gs_io.load_gsplat(
            scene_info["fit_lr_to_hr_paths"]["gsplat_dir"]
        )

        processed = {}
        target_gs, target_scaler = self.processor.process_native(
            raw_by_resolution[self.fit_target_resolution]
        )
        source_gs, source_scaler, fitted_gs = self.processor.process_fitted_pair(
            raw_by_resolution[self.fit_source_resolution],
            fitted_raw,
            target_scaler,
        )
        processed[self.fit_source_resolution] = (source_gs, source_scaler)
        processed[self.fit_target_resolution] = (target_gs, target_scaler)
        for resolution in self.resolutions:
            if resolution not in processed:
                processed[resolution] = self.processor.process_native(
                    raw_by_resolution[resolution]
                )

        resolution_data = {}
        reference_names = None
        for resolution in self.resolutions:
            paths = scene_info["resolution_paths"][resolution]
            gs_params, scaler = processed[resolution]
            meta, image_paths = gs_io.load_colmap_views(paths["colmap_dir"])
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
            gs_io.read_image(resolution_entry["imgs_path"][index], background)
            for index in camera_ids
        ]
        image_names = [
            resolution_entry["imgs_name"][index] for index in camera_ids
        ]
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

            multiresolution = {}
            for resolution in self.resolutions:
                resolution_entry = scene["resolution_data"][resolution]
                images, image_names, cameras = self.load_resolution_views(
                    resolution_entry,
                    camera_ids=camera_ids,
                    background=background,
                )
                multiresolution[resolution] = {
                    "gs_params": resolution_entry["gs_params"],
                    "scaler": resolution_entry["scaler"],
                    "images": images,
                    "cameras": cameras,
                    "images_name": image_names,
                    "checkpoint_path": resolution_entry["checkpoint_path"],
                }
            yield {
                "scene_idx": scene["idx"],
                "scene_name": scene["scene_name"],
                "resolutions": self.resolutions,
                "multiresolution": multiresolution,
                "fit_lr_to_hr": scene["fit_lr_to_hr"],
            }

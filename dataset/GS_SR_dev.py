import os
import random
from typing import Optional, Sequence

import gin
import numpy as np
import torch

from dataset import gs_io
from dataset.gs_processing import GaussianProcessor
from utils import gs_utils


COORDINATE_FRAME = "input_resolution"
COORDINATE_FRAME_VERSION = 1


@gin.configurable
class SplatFactoSRDevDataset(torch.utils.data.IterableDataset):
    """Load SR scenes in one coordinate frame derived from the input GS."""

    def __init__(
        self,
        train_or_test: str,
        dataset_root: str,
        train_scene_list: str,
        test_scene_list: str,
        image_per_scene: Optional[int],
        remove_outlier_ndevs: float,
        max_gs_num: int,
        split_across_gpus: bool,
        fit_lr_to_hr_root: str,
        fit_hr_to_lr_root: str,
        background_color: Sequence[int] = (0, 0, 0),
        dataset_name: str = "objaverse",
        src_resolution: int = 128,
        tgt_resolution: int = 512,
        load_src_gs: bool = True,
        load_tgt_gs: bool = True,
        load_src_images: bool = True,
        load_tgt_images: bool = True,
        alignment: Optional[str] = None,
    ):
        if train_or_test not in ("train", "test"):
            raise ValueError("train_or_test must be either 'train' or 'test'")
        self.train_or_test = train_or_test
        self.dataset_root = os.fspath(dataset_root)
        self.fit_lr_to_hr_root = os.fspath(fit_lr_to_hr_root)
        self.fit_hr_to_lr_root = os.fspath(fit_hr_to_lr_root)
        self.scene_list = os.fspath(
            train_scene_list if train_or_test == "train" else test_scene_list
        )
        self.image_per_scene = image_per_scene
        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.max_gs_num = max_gs_num
        self.split_across_gpus = split_across_gpus
        self.background_color = background_color
        self.dataset_name = dataset_name
        self.src_resolution = int(src_resolution)
        self.tgt_resolution = int(tgt_resolution)
        self.load_src_gs = bool(load_src_gs)
        self.load_tgt_gs = bool(load_tgt_gs)
        self.load_src_images = bool(load_src_images)
        self.load_tgt_images = bool(load_tgt_images)
        self.alignment = alignment
        self.coordinate_frame = COORDINATE_FRAME
        self.coordinate_frame_version = COORDINATE_FRAME_VERSION
        if self.src_resolution == self.tgt_resolution:
            raise ValueError("src_resolution and tgt_resolution must be different")

        self.processor = GaussianProcessor(remove_outlier_ndevs, max_gs_num)
        self.filtered_scenes = []
        self.folders = self.load_scene_manifest()
        if self.train_or_test == "test":
            self.remaining_scenes = self.test_process_split()
        else:
            self.counter = 0

    @classmethod
    def from_gin_scope(cls, scope, **kwargs):
        with gin.config_scope(scope):
            return cls(**kwargs)

    @property
    def split_root(self):
        return os.path.join(
            self.dataset_root, f"{self.train_or_test}-set", self.dataset_name
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
                    for resolution in (self.src_resolution, self.tgt_resolution)
                },
            }
            folders.append(scene_info)

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

    def load_camera_meta(self, scene_info, resolution, coordinate_scaler):
        paths = scene_info["resolution_paths"][resolution]
        meta, image_paths = gs_io.load_colmap_views(paths["colmap_dir"])
        meta["camera_to_worlds"][:, :3, -1] = coordinate_scaler.transform(
            meta["camera_to_worlds"][:, :3, -1]
        )
        image_names = [os.path.basename(path) for path in image_paths]
        if len(image_names) != len(meta["camera_to_worlds"]):
            raise ValueError(
                f"Scene {scene_info['scene_name']} resolution {resolution}: "
                "image count does not match pose count"
            )
        return {
            "meta": meta,
            "imgs_path": image_paths,
            "imgs_name": image_names,
        }

    def pretrained_path(self, fit_alignment, scene_name, source_resolution):
        if fit_alignment == "fit_lr_to_hr":
            root = self.fit_lr_to_hr_root
        elif fit_alignment == "fit_hr_to_lr":
            root = self.fit_hr_to_lr_root
        else:
            raise ValueError(f"Unsupported pretrained alignment: {fit_alignment}")
        return gs_io.fitted_paths(root, source_resolution, scene_name)["gsplat_dir"]

    def load_pretrained_fit(
        self,
        fit_alignment,
        scene_name,
        source_resolution,
        source_gs_raw,
        source_gs_mask,
        target_scaler,
        coordinate_scaler,
    ):
        gsplat_dir = self.pretrained_path(fit_alignment, scene_name, source_resolution)
        try:
            target_gs_raw, _ = gs_io.load_gsplat(gsplat_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing preloaded {fit_alignment} GS: {gsplat_dir}") from exc
        if set(target_gs_raw) != set(source_gs_raw):
            raise ValueError(f"Preloaded {fit_alignment} attributes do not match source GS: {gsplat_dir}")
        for key, source_value in source_gs_raw.items():
            if target_gs_raw[key].shape != source_value.shape:
                raise ValueError(f"Preloaded {fit_alignment} is not identity-paired for {key}: {gsplat_dir}")
        target_gs_raw = {key: value[source_gs_mask] for key, value in target_gs_raw.items()}
        if not self.processor.finite_mask(target_gs_raw).all():
            raise ValueError(f"Preloaded {fit_alignment} GS contains non-finite values: {gsplat_dir}")
        target_gs, _ = self.processor.normalize(target_gs_raw, scaler=target_scaler)
        return gs_utils.convert_gaussian_frame(target_gs, target_scaler, coordinate_scaler)

    def load_scene(self, scene_idx, sample_views=False, fit_alignment=None):
        scene_info = self.folders[scene_idx]
        valid_fit_alignments = (None, "emd", "random", "fit_lr_to_hr", "fit_hr_to_lr")
        if fit_alignment not in valid_fit_alignments:
            raise ValueError(f"Unsupported fit_alignment: {fit_alignment}")
        # 1. Calculate coordinate scaler from source GS
        src_gs_paths = scene_info["resolution_paths"][self.src_resolution]
        src_gs_raw, _ = gs_io.load_gsplat(src_gs_paths["gsplat_dir"])
        src_gs_mask = self.processor.selection_mask(src_gs_raw)
        selected_src_gs = {
            key: value[src_gs_mask] for key, value in src_gs_raw.items()
        }
        normalized_src_gs, coordinate_scaler = self.processor.normalize(selected_src_gs)

        # 2. Load GS
        normalized_gs = {}
        resolution_scalers = {self.src_resolution: coordinate_scaler}
        fitted_gs_pair = None
        needs_target_gs = self.load_tgt_gs or fit_alignment in (
            "fit_lr_to_hr", "fit_hr_to_lr"
        )
        if self.load_src_gs:
            normalized_gs[self.src_resolution] = normalized_src_gs
        if needs_target_gs:
            tgt_gs_paths = scene_info["resolution_paths"][self.tgt_resolution]
            tgt_gs_raw, _ = gs_io.load_gsplat(tgt_gs_paths["gsplat_dir"])
            tgt_gs_mask = self.processor.selection_mask(tgt_gs_raw)
            selected_tgt_gs = {
                key: value[tgt_gs_mask] for key, value in tgt_gs_raw.items()
            }
            _, resolution_scalers[self.tgt_resolution] = self.processor.normalize(
                selected_tgt_gs
            )
            normalized_gs[self.tgt_resolution], _ = self.processor.normalize(
                selected_tgt_gs, scaler=coordinate_scaler
            )

            if fit_alignment in ("fit_lr_to_hr", "fit_hr_to_lr"):
                if fit_alignment == "fit_lr_to_hr":
                    fit_source_resolution = self.src_resolution
                    fit_target_resolution = self.tgt_resolution
                    fit_source_gs_raw = src_gs_raw
                    fit_source_gs_mask = src_gs_mask
                else:
                    fit_source_resolution = self.tgt_resolution
                    fit_target_resolution = self.src_resolution
                    fit_source_gs_raw = tgt_gs_raw
                    fit_source_gs_mask = tgt_gs_mask
                fitted_tgt_gs = self.load_pretrained_fit(
                    fit_alignment,
                    scene_info["scene_name"],
                    fit_source_resolution,
                    fit_source_gs_raw,
                    fit_source_gs_mask,
                    resolution_scalers[fit_target_resolution],
                    coordinate_scaler,
                )
                fitted_gs_pair = {
                    "src_resolution": fit_source_resolution,
                    "tgt_resolution": fit_target_resolution,
                    "tgt_gs": fitted_tgt_gs,
                    "coordinate_frame": self.coordinate_frame,
                    "coordinate_resolution": self.src_resolution,
                }

        # 3. Build the final resolution payloads
        data = {}
        camera_ids = None
        background = self.build_background()
        for resolution in (self.src_resolution, self.tgt_resolution):
            res_data = {}
            load_gs = (
                self.load_src_gs if resolution == self.src_resolution
                else self.load_tgt_gs
            )
            load_images = (
                self.load_src_images if resolution == self.src_resolution
                else self.load_tgt_images
            )
            if load_gs:
                res_data["gs_params"] = normalized_gs[resolution]
            if load_images:
                camera_data = self.load_camera_meta(
                    scene_info, resolution, coordinate_scaler
                )
                if camera_ids is None:
                    total_views = len(camera_data["meta"]["camera_to_worlds"])
                    camera_ids = np.arange(total_views)
                    if sample_views and self.train_or_test == "train":
                        sample_count = total_views
                        if self.image_per_scene is not None:
                            sample_count = min(self.image_per_scene, total_views)
                        camera_ids = np.random.permutation(total_views)[:sample_count]
                    elif sample_views and self.background_color == "random":
                        raise ValueError("Test background_color cannot be random")
                images, image_names, cameras = self.load_resolution_views(
                    camera_data, camera_ids=camera_ids, background=background
                )
                res_data.update(
                    {
                        "images": images,
                        "cameras": cameras,
                        "images_name": image_names,
                    }
                )
            data[resolution] = res_data

        sample = {
            "scene_idx": scene_idx,
            "scene_name": scene_info["scene_name"],
            "coordinate_frame": self.coordinate_frame,
            "coordinate_frame_version": self.coordinate_frame_version,
            "coordinate_resolution": self.src_resolution,
            "data": data,
        }
        if fitted_gs_pair is not None:
            sample[fit_alignment] = fitted_gs_pair
        return sample

    def build_background(self):
        if self.background_color == "random":
            return torch.rand(3)
        return torch.tensor(self.background_color, dtype=torch.float32) / 255.0

    def load_resolution_views(
        self, resolution_entry, camera_ids=None, background=None
    ):
        meta = resolution_entry["meta"]
        total_views = len(meta["camera_to_worlds"])
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
            yield self.load_scene(
                scene_idx, sample_views=True, fit_alignment=self.alignment
            )

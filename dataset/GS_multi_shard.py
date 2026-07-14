"""Bounded local-shard variant of :mod:`dataset.GS_multi`.

Set ``local_shard_root`` to stage the next scheduled group of scenes into a
local directory.  Leave it as ``None`` to use the original dataset behavior.
"""

import copy
import glob
from collections import deque
import os
import shutil
from pathlib import Path
import threading
from typing import Optional

import gin
import numpy as np
import torch

from dataset.GS_multi import (
    CAMERA_METADATA_NAME,
    FACTOR_TO_IMAGE_DIR,
    _GSCheckpointLoadError,
    SplatFactoMultiLevelDataset,
)


@gin.configurable
class SplatFactoMultiLevelShardDataset(SplatFactoMultiLevelDataset):
    """Multi-level dataset that optionally stages one scheduled shard locally.

    Shard membership intentionally follows the parent dataset's existing
    scheduling order.  No extra seed or deterministic ordering is introduced.
    """

    def __init__(
        self,
        train_or_test,
        nerfstudio_folder,
        colmap_folder,
        load_pose_src,
        sample_ratio_test: Optional[float],
        image_per_scene: Optional[int],
        remove_outlier_ndevs: float,
        max_gs_num: int,
        split_across_gpus: bool,
        factors: list = [1, 2, 4],
        background_color: list = [0, 0, 0],
        skip_invalid_scenes: bool = True,
        cache_steps: Optional[int] = None,
        cache_num_scenes: Optional[int] = None,
        local_shard_root: Optional[str] = None,
        local_queue_capacity: int = 500,
        local_queue_initial: int = 500,
        local_shard_enabled_for_test: bool = False,
    ):
        super().__init__(
            train_or_test=train_or_test,
            nerfstudio_folder=nerfstudio_folder,
            colmap_folder=colmap_folder,
            load_pose_src=load_pose_src,
            sample_ratio_test=sample_ratio_test,
            image_per_scene=image_per_scene,
            remove_outlier_ndevs=remove_outlier_ndevs,
            max_gs_num=max_gs_num,
            split_across_gpus=split_across_gpus,
            factors=factors,
            background_color=background_color,
            skip_invalid_scenes=skip_invalid_scenes,
            cache_steps=cache_steps,
            cache_num_scenes=cache_num_scenes,
        )
        if local_queue_capacity < 1 or local_queue_initial < 1:
            raise ValueError("local queue capacity and initial preload must be positive")
        if local_queue_initial > local_queue_capacity:
            raise ValueError("local_queue_initial cannot exceed local_queue_capacity")
        self.local_shard_root = local_shard_root
        self.local_queue_capacity = local_queue_capacity
        self.local_queue_initial = local_queue_initial
        self.local_shard_enabled = local_shard_root is not None and (
            train_or_test == "train" or local_shard_enabled_for_test
        )
        self._source_folders = copy.deepcopy(self.folders)
        self._active_local_scenes = {}
        self._active_shard_indices = []
        self._shard_ready = False
        self._pending_shard_scenes = []
        self._queue_condition = threading.Condition()
        self._queue_ready = deque()
        self._queue_loading = 0
        self._queue_in_use = 0
        self._queue_error = None
        self._queue_stop = threading.Event()
        self._queue_thread = None
        self._queue_ticket = 0

        if self.local_shard_enabled and train_or_test == "test":
            # The test set is intentionally small; stage it once rather than rotate it.
            self._stage_scene_indices(list(range(len(self._source_folders))))

    def _stage_directory(self):
        """Use a process-specific directory so DDP ranks never overwrite each other."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        return Path(self.local_shard_root) / f"rank-{rank}" / f"worker-{worker_id}"

    def _latest_checkpoint(self, nerfstudio_dir):
        checkpoints = sorted(
            glob.glob(os.path.join(nerfstudio_dir, "nerfstudio_models", "step-*.ckpt")),
            key=lambda path: int(os.path.splitext(os.path.basename(path))[0].split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found under {nerfstudio_dir}")
        return checkpoints[-1]

    def _copy_tree(self, source, destination):
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)

    def _copy_scene(self, scene_idx, partial_root):
        source = self._source_folders[scene_idx]
        scene_name = source["scene_name"]
        local = copy.deepcopy(source)
        for factor in self.factors:
            source_paths = source["factor_paths"][factor]
            destination_ns = (
                partial_root / "nerfstudio" / scene_name / f"df-{factor}" / "splatfacto"
            )
            destination_models = destination_ns / "nerfstudio_models"
            destination_models.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                os.path.join(source_paths["nerfstudio_dir"], CAMERA_METADATA_NAME),
                destination_ns / CAMERA_METADATA_NAME,
            )
            checkpoint = self._latest_checkpoint(source_paths["nerfstudio_dir"])
            shutil.copy2(checkpoint, destination_models / os.path.basename(checkpoint))

            image_name = FACTOR_TO_IMAGE_DIR.get(factor, f"images_{factor}")
            destination_colmap = partial_root / "colmap" / scene_name
            self._copy_tree(source_paths["image_dir"], destination_colmap / image_name)
            mask_dir = source_paths["image_dir"].replace("images", "masks")
            self._copy_tree(mask_dir, destination_colmap / os.path.basename(mask_dir))
            local["factor_paths"][factor] = {
                "nerfstudio_dir": str(destination_ns),
                "image_dir": str(destination_colmap / image_name),
            }
        local["colmap_dir"] = str(partial_root / "colmap" / scene_name)
        return local

    def _stage_scene_indices(self, scene_indices):
        if not scene_indices:
            raise RuntimeError("Cannot stage an empty shard")
        stage_dir = self._stage_directory()
        partial = stage_dir / "active.partial"
        active = stage_dir / "active"
        previous = stage_dir / "active.previous"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True, exist_ok=True)
        local_scenes = {}
        try:
            for scene_idx in scene_indices:
                local_scenes[scene_idx] = self._copy_scene(scene_idx, partial)
            (partial / "READY").touch()
            shutil.rmtree(previous, ignore_errors=True)
            if active.exists():
                active.rename(previous)
            partial.rename(active)
            shutil.rmtree(previous, ignore_errors=True)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise

        # Rebase paths from the renamed partial root to its active final location.
        self._active_local_scenes = {}
        for scene_idx, scene in local_scenes.items():
            rebased = copy.deepcopy(scene)
            for factor in self.factors:
                for key in ("nerfstudio_dir", "image_dir"):
                    rebased["factor_paths"][factor][key] = rebased["factor_paths"][factor][key].replace(
                        str(partial), str(active), 1
                    )
            rebased["colmap_dir"] = rebased["colmap_dir"].replace(str(partial), str(active), 1)
            self._active_local_scenes[scene_idx] = rebased
        self._active_shard_indices = list(scene_indices)
        self._shard_ready = True

    def _rebase_local_scene(self, scene, partial, active):
        rebased = copy.deepcopy(scene)
        for factor in self.factors:
            for key in ("nerfstudio_dir", "image_dir"):
                rebased["factor_paths"][factor][key] = rebased["factor_paths"][factor][key].replace(
                    str(partial), str(active), 1
                )
        rebased["colmap_dir"] = rebased["colmap_dir"].replace(str(partial), str(active), 1)
        return rebased

    def _stage_queue_scene(self, scene_idx, ticket):
        slots = self._stage_directory() / "queue"
        ready_root = slots / f"{ticket:08d}-{scene_idx}"
        partial_root = slots / f"{ticket:08d}-{scene_idx}.partial"
        shutil.rmtree(partial_root, ignore_errors=True)
        partial_root.mkdir(parents=True, exist_ok=True)
        try:
            local_scene = self._copy_scene(scene_idx, partial_root)
            (partial_root / "READY").touch()
            shutil.rmtree(ready_root, ignore_errors=True)
            partial_root.rename(ready_root)
        except Exception:
            shutil.rmtree(partial_root, ignore_errors=True)
            raise
        return self._rebase_local_scene(local_scene, partial_root, ready_root), ready_root

    def _next_scheduled_scene(self):
        if not self._pending_shard_scenes:
            self.refresh_remaining_training()
            self._pending_shard_scenes = self.remaining_scenes
            self.remaining_scenes = []
        return self._pending_shard_scenes.pop(0)

    def _queue_loader(self):
        while not self._queue_stop.is_set():
            with self._queue_condition:
                while (
                    not self._queue_stop.is_set()
                    and self._queue_loading + self._queue_in_use + len(self._queue_ready)
                    >= self.local_queue_capacity
                ):
                    self._queue_condition.wait()
                if self._queue_stop.is_set():
                    return
                scene_idx = self._next_scheduled_scene()
                ticket = self._queue_ticket
                self._queue_ticket += 1
                self._queue_loading += 1
            try:
                local_scene, slot_root = self._stage_queue_scene(scene_idx, ticket)
            except BaseException as exc:
                with self._queue_condition:
                    self._queue_loading -= 1
                    self._queue_error = exc
                    self._queue_condition.notify_all()
                return
            with self._queue_condition:
                self._queue_loading -= 1
                self._queue_ready.append((scene_idx, local_scene, slot_root))
                self._queue_condition.notify_all()

    def _start_queue_loader(self):
        if self._queue_thread is not None and self._queue_thread.is_alive():
            return
        self._queue_stop.clear()
        self._queue_error = None
        self._queue_thread = threading.Thread(target=self._queue_loader, daemon=True)
        self._queue_thread.start()
        with self._queue_condition:
            while len(self._queue_ready) < self.local_queue_initial:
                if self._queue_error is not None:
                    raise RuntimeError("Local queue staging failed") from self._queue_error
                self._queue_condition.wait()

    def _take_ready_scene(self):
        with self._queue_condition:
            while not self._queue_ready:
                if self._queue_error is not None:
                    raise RuntimeError("Local queue staging failed") from self._queue_error
                self._queue_condition.wait()
            self._queue_in_use += 1
            return self._queue_ready.popleft()

    def _release_queue_scene(self, slot_root):
        shutil.rmtree(slot_root, ignore_errors=True)
        with self._queue_condition:
            self._queue_in_use -= 1
            self._queue_condition.notify_all()

    def _stop_queue_loader(self):
        self._queue_stop.set()
        with self._queue_condition:
            self._queue_condition.notify_all()

    def load_scene(self, idx):
        if not self.local_shard_enabled:
            return super().load_scene(idx)
        return self._load_scene_from_info(idx, self._active_local_scenes[idx])

    def _load_scene_from_info(self, idx, scene_info):
        factor_data = {}
        for factor in self.factors:
            paths = scene_info["factor_paths"][factor]
            gs_params, scaler = self.load_gs_params_fromnerfstudio(paths["nerfstudio_dir"], idx)
            meta, imgs_path = self.load_images_cameras_fromnerfstudio(
                paths["nerfstudio_dir"], scene_info["colmap_dir"], paths["image_dir"]
            )
            meta["camera_to_worlds"][:, :3, -1] = scaler.transform(meta["camera_to_worlds"][:, :3, -1])
            entry = {
                "gs_params": gs_params, "meta": meta, "imgs_path": imgs_path,
                "imgs_name": [os.path.basename(path) for path in imgs_path], "scaler": scaler,
            }
            self._validate_pose_image_count(factor, entry)
            factor_data[factor] = entry
        self._validate_factor_alignment(factor_data)
        return {"idx": idx, "scene_name": scene_info["scene_name"], "factor_data": factor_data}

    def __iter__(self):
        if not self.local_shard_enabled:
            yield from super().__iter__()
            return
        if self.train_or_test == "test":
            yield from super().__iter__()
            return

        self._start_queue_loader()
        slot_root = None
        try:
            while True:
                scene_idx, scene_info, slot_root = self._take_ready_scene()
                try:
                    scene = self._load_scene_from_info(scene_idx, scene_info)
                except _GSCheckpointLoadError as exc:
                    self._skip_checkpoint_load_failure(scene_idx, exc)
                    self._release_queue_scene(slot_root)
                    slot_root = None
                    continue
                factor_data = scene["factor_data"]
                total_num = len(factor_data[self.primary_factor]["meta"]["camera_to_worlds"])
                background = self.build_background()
                sample_num = total_num if self.image_per_scene is None else min(self.image_per_scene, total_num)
                cam_ids = np.random.permutation(total_num)[:sample_num]
                multilevel = {
                    factor: self._prepare_factor_payload(factor_data[factor], background, cam_ids)
                    for factor in self.factors
                }
                yield {
                    "scene_idx": scene["idx"], "scene_name": scene["scene_name"],
                    "factors": self.factors, "multilevel": multilevel,
                }
                self._release_queue_scene(slot_root)
                slot_root = None
        finally:
            if slot_root is not None:
                self._release_queue_scene(slot_root)
            self._stop_queue_loader()

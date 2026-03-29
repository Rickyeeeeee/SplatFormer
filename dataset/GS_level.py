import glob
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import gin
import numpy as np
import torch
from PIL import Image

from dataset import colmap_utils
from utils.transform_utils import MinMaxScaler, remove_outliers


@gin.configurable
class SplatfactoLevelDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        train_or_test,
        nerfstudio_folder,
        colmap_folder,
        load_pose_src,  # [colmap or nerfstudio]
        sample_ratio_test: Optional[float],
        image_per_scene: Optional[int],
        remove_outlier_ndevs: float,
        max_gs_num: int,
        cache_steps: int,
        cache_num_scenes: int,  # Default: cache_num_scenes=1, cache_steps=1
        split_across_gpus: bool,
        level: int,
        background_color: list = [0, 0, 0],
    ):
        self.train_or_test = train_or_test
        self.image_per_scene = image_per_scene
        self.sample_ratio_test = sample_ratio_test
        self.level = int(level)
        self.level_str = f"level-{self.level}"
        self.images_dirname = "images" if self.level == 1 else f"images_{self.level}"

        # Explicitly disable COLMAP camera loading for now (requested behavior).
        if load_pose_src != "nerfstudio":
            raise ValueError(
                "SplatfactoLevelDataset currently only supports "
                "load_pose_src='nerfstudio'."
            )
        self.load_pose_src = load_pose_src

        self.folders = self._build_scene_pairs(nerfstudio_folder, colmap_folder)

        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.cache_steps, self.cache_num_scenes = cache_steps, cache_num_scenes
        self.split_across_gpus = split_across_gpus
        self.max_gs_num = max_gs_num
        self.cache_scenes = []
        self.background_color = background_color

        if train_or_test in ["test"]:
            # For test set, we need to split data across device deterministically
            self.remaining_scenes = list(range(len(self.folders)))
            assert self.cache_num_scenes == 1 and self.cache_steps == 1, "For test, we do not cache"
            # For DDP evaluation, we need to chunk the data
            try:
                world_size = torch.cuda.device_count()
                rank = torch.distributed.get_rank()
            except Exception:
                world_size, rank = 1, 0
            chunk_size = len(self.remaining_scenes) // world_size
            if rank == world_size - 1:
                self.remaining_scenes = self.remaining_scenes[rank * chunk_size :]
            else:
                self.remaining_scenes = self.remaining_scenes[rank * chunk_size : (rank + 1) * chunk_size]
        else:
            self.counter = 0

    def _load_scene_roots(self, root_or_txt: str, tag: str):
        if root_or_txt.endswith(".txt"):
            scene_roots = []
            with open(root_or_txt, "r") as f:
                for line in f:
                    path = line.strip()
                    if path:
                        scene_roots.append(path)
        elif os.path.isdir(root_or_txt):
            scene_roots = sorted(
                [
                    os.path.join(root_or_txt, name)
                    for name in os.listdir(root_or_txt)
                    if os.path.isdir(os.path.join(root_or_txt, name))
                ]
            )
        else:
            raise ValueError(f"Invalid {tag} path: {root_or_txt}")

        for p in scene_roots:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"{tag} scene root does not exist: {p}")
        return scene_roots

    def _scene_map(self, scene_roots, tag: str):
        scene_map = {}
        for path in scene_roots:
            scene_name = os.path.basename(os.path.normpath(path))
            if scene_name in scene_map:
                raise ValueError(f"Duplicate scene name '{scene_name}' in {tag}: {path}")
            scene_map[scene_name] = path
        return scene_map

    def _build_scene_pairs(self, nerfstudio_folder: str, colmap_folder: str):
        ns_roots = self._load_scene_roots(nerfstudio_folder, "nerfstudio")
        colmap_roots = self._load_scene_roots(colmap_folder, "colmap")
        ns_map = self._scene_map(ns_roots, "nerfstudio")
        colmap_map = self._scene_map(colmap_roots, "colmap")

        ns_names = set(ns_map.keys())
        colmap_names = set(colmap_map.keys())
        if ns_names != colmap_names:
            missing_in_colmap = sorted(ns_names - colmap_names)
            missing_in_nerfstudio = sorted(colmap_names - ns_names)
            raise ValueError(
                "Scene mismatch between nerfstudio and colmap folders. "
                f"Missing in colmap: {missing_in_colmap[:10]}, "
                f"missing in nerfstudio: {missing_in_nerfstudio[:10]}"
            )

        folders = []
        for scene_name in sorted(ns_names):
            ns_scene_root = ns_map[scene_name]
            colmap_scene_root = colmap_map[scene_name]

            ns_level_dir = os.path.join(ns_scene_root, self.level_str)
            ns_models_dir = os.path.join(ns_level_dir, "nerfstudio_models")
            if not os.path.isdir(ns_level_dir):
                raise FileNotFoundError(
                    f"Missing nerfstudio level folder for scene '{scene_name}': {ns_level_dir}"
                )
            if not os.path.isdir(ns_models_dir):
                raise FileNotFoundError(
                    f"Missing nerfstudio_models for scene '{scene_name}': {ns_models_dir}"
                )

            image_dir = os.path.join(colmap_scene_root, self.images_dirname)
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(
                    f"Missing image folder for scene '{scene_name}': {image_dir}"
                )

            folders.append((ns_level_dir, colmap_scene_root, image_dir))

        if len(folders) == 0:
            raise ValueError("No scenes found for SplatfactoLevelDataset.")

        return folders

    def refresh_remaining_training(self):
        if self.split_across_gpus:
            self.random_split_to_remaining()
        else:
            self.remaining_scenes = self.get_thisworker_split(N=len(self.folders))
            random.shuffle(self.remaining_scenes)
        self.counter += 1
        return

    def get_thisworker_split(self, N):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return list(range(N))
        per_worker = N // worker_info.num_workers
        worker_id = worker_info.id
        if worker_id == worker_info.num_workers - 1:
            return list(range(worker_id * per_worker, N))
        else:
            return list(range(worker_id * per_worker, (worker_id + 1) * per_worker))

    def random_split_to_remaining(self):
        """
        Generate a new permutation of the folders.
        """
        # 1. Split self.folders across processes
        np.random.seed(torch.distributed.get_rank())
        torch.manual_seed(torch.distributed.get_rank())
        rng_state = np.random.get_state()
        world_size = torch.cuda.device_count()
        rank = torch.distributed.get_rank()
        np.random.seed(self.counter)
        # Pad to world_size*k
        permutation = np.random.permutation(len(self.folders))
        pad_num = world_size - len(self.folders) % world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])
        np.random.set_state(rng_state)

        chunk_size = len(permutation) // world_size
        if rank == world_size - 1:
            remaining_scenes_for_thisprocess = permutation[rank * chunk_size :]
        else:
            remaining_scenes_for_thisprocess = permutation[rank * chunk_size : (rank + 1) * chunk_size]

        # 2. Split scenes across workers
        split_id = self.get_thisworker_split(N=len(remaining_scenes_for_thisprocess))
        self.remaining_scenes = [remaining_scenes_for_thisprocess[i] for i in split_id]
        return

    @gin.configurable
    def read_image(self, path, background):
        mask = None
        try:
            pil_image = Image.open(path)
        except Exception:
            print(f"Warning: {path} cannot be opened")
            raise

        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255.0
        if "real" in path.lower():
            possible_mask_filename = path.replace("images", "masks")  # TODO: hardcoded
            if os.path.exists(possible_mask_filename):
                mask = np.array(Image.open(possible_mask_filename)).astype(image.dtype) / 255.0
                mask = torch.from_numpy(mask)

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + background * (1.0 - image[:, :, -1:])
        elif mask is not None:
            image_rgb = image * mask[..., None] + background * (1.0 - mask[..., None])
            # As we need to preserve the mask for evaluation, we save the image RGBA.
            image = torch.concat([image_rgb, mask[..., None]], axis=-1)  # Hardcoded for real dataset
        return image

    def load_gs_params_fromnerfstudio(self, nerfstudio_dir, idx):
        skip_params = gin.query_parameter("FeaturePredictor.input_features")
        if gin.query_parameter("training.pretrain_steps") > 0:
            skip_params = skip_params + gin.query_parameter("create_pseudo_target.take_from_input")
        try:
            ckpt_file = glob.glob(nerfstudio_dir + "/nerfstudio_models/step-*.ckpt")[-1]  # Take last checkpoint
        except Exception:
            raise FileNotFoundError(
                f"{nerfstudio_dir} does not have nerfstudio_models/step-*.ckpt"
            )

        ckpt = torch.load(ckpt_file, map_location="cpu")
        ckpt = {k.replace("_model.gauss_params.", ""): v for k, v in ckpt.items() if "gauss_params" in k}
        gs_params = {k: ckpt[k] for k in set(skip_params)}

        # Remove inf or nan
        select = torch.ones(gs_params["means"].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key == "features_rest":
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        # Filter outliers
        if self.remove_outlier_ndevs > 0:
            _, inlier_mask = remove_outliers(gs_params["means"], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # Truncate if num > max_gs_num
        N = gs_params["means"].shape[0]
        if N > self.max_gs_num:
            inlier_mask = torch.zeros(N, dtype=torch.bool)
            inlier_mask[: self.max_gs_num] = True
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # Normalize means/scales (camera translation will be transformed with same scaler)
        scaler = MinMaxScaler()
        gs_params["means"] = scaler.fit_transform(gs_params["means"])
        gs_params["scales"] = gs_params["scales"] + torch.log(scaler.scale_)

        inf_mask = torch.isinf(gs_params["scales"]).sum(dim=1).bool()
        valid_mask = (~inf_mask).bool()
        inrange_mask = torch.all((gs_params["means"] >= 0) & (gs_params["means"] <= 1), dim=1)
        valid_mask = valid_mask & inrange_mask
        for key in gs_params:
            gs_params[key] = gs_params[key][valid_mask]
            if torch.isnan(gs_params[key]).any():
                print(f"Warning: {key} contains nan", nerfstudio_dir)

        return gs_params, scaler

    def load_images_cameras_fromnerfstudio(self, nerfstudio_dir, colmap_dir, image_dir):
        camera_pkl = os.path.join(nerfstudio_dir, "camera_for-3d-denoise.pkl")
        if not os.path.isfile(camera_pkl):
            raise FileNotFoundError(f"Missing camera metadata file: {camera_pkl}")
        with open(camera_pkl, "rb") as f:
            meta = pickle.load(f)

        train_imgs_path, test_imgs_path = [], []
        image_names = os.listdir(image_dir)

        # Hard coded, only used for real-world dataset
        split_file = os.path.join(colmap_dir, "ood-test_split.txt")
        if os.path.isfile(split_file):
            ood_test_img_names = []
            with open(split_file, "r") as f:
                for line in f.readlines():
                    ood_test_img_names.append(line.strip())
        else:
            ood_test_img_names = None

        testset_elevation = False
        for name in sorted(image_names):  # TODO: order may not align to camera_to_worlds
            if "elevation" in name:  # Hardcoded: test set
                assert self.train_or_test == "test"
                testset_elevation = True
                if "elevation90" in name or "elevation80" in name or "elevation70" in name:
                    test_imgs_path.append(os.path.join(image_dir, name))
            else:
                if name.startswith("test") or name.startswith("frame_eval"):
                    test_imgs_path.append(os.path.join(image_dir, name))
                else:
                    train_imgs_path.append(os.path.join(image_dir, name))

        if testset_elevation:
            meta["test_camera_to_worlds"] = meta["test_camera_to_worlds"][-3 * 3 :]
        if ood_test_img_names is not None:
            ood_ids = [i for i, path in enumerate(test_imgs_path) if os.path.basename(path) in ood_test_img_names]
            test_imgs_path = [test_imgs_path[i] for i in ood_ids]
            meta["test_camera_to_worlds"] = meta["test_camera_to_worlds"][ood_ids]
        return meta, train_imgs_path, test_imgs_path

    def load_scene(self, idx):
        nerfstudio_dir, colmap_dir, image_dir = self.folders[idx]
        gs_params, scaler = self.load_gs_params_fromnerfstudio(nerfstudio_dir, idx)
        meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromnerfstudio(
            nerfstudio_dir, colmap_dir, image_dir
        )
        meta["train_camera_to_worlds"][:, :3, -1] = scaler.transform(meta["train_camera_to_worlds"][:, :3, -1])
        meta["test_camera_to_worlds"][:, :3, -1] = scaler.transform(meta["test_camera_to_worlds"][:, :3, -1])

        outputs = {
            "gs_params": gs_params,
            "meta": meta,
            "idx": idx,
            "scene_name": nerfstudio_dir.split("/")[-2],  # nerfstudio/<scene>/level-k
            "train_imgs_path": train_imgs_path,
            "test_imgs_path": test_imgs_path,
        }
        return outputs

    def get_scene_from_cache(self):
        # 1) If cache is not full, append a new scene
        if len(self.cache_scenes) < self.cache_num_scenes:
            idx = self.remaining_scenes.pop(0)
            if self.train_or_test == "train" and len(self.remaining_scenes) == 0:
                self.refresh_remaining_training()
            new_scene = self.load_scene(idx)
            if self.cache_steps != 1:  # cache_steps=-1: forever, >1: multiple steps
                self.cache_scenes.append([new_scene, 1])
            return new_scene
        else:
            # 2) If cache is full, sample one cached scene
            scene_i = random.randint(0, len(self.cache_scenes) - 1)
            scene = self.cache_scenes[scene_i][0]
            self.cache_scenes[scene_i][1] += 1
            if self.cache_scenes[scene_i][1] == self.cache_steps:  # cache_steps=-1 means forever
                self.cache_scenes.pop(scene_i)
                self.get_scene_from_cache()
            return scene

    def __iter__(self):
        if self.train_or_test == "train":
            self.refresh_remaining_training()
        if len(self.remaining_scenes) < self.cache_num_scenes:
            print(
                f"Warning: The number of scenes is less than cache_num_scenes, "
                f"{len(self.remaining_scenes)} < {self.cache_num_scenes}"
            )
            self.cache_num_scenes = len(self.remaining_scenes)
            print(f"cache_num_scenes is set to {self.cache_num_scenes}")
        while len(self.remaining_scenes) > 0:
            scene = self.get_scene_from_cache()
            gs_params, meta, scene_name = scene["gs_params"], scene["meta"], scene["scene_name"]
            train_imgs_path, test_imgs_path = scene["train_imgs_path"], scene["test_imgs_path"]
            train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
            test_imgs_name = [os.path.basename(path) for path in test_imgs_path]

            total_train_num, total_test_num = len(meta["train_camera_to_worlds"]), len(meta["test_camera_to_worlds"])
            cameras = {}
            if self.train_or_test == "train":
                sample_test = np.random.rand(self.image_per_scene) < self.sample_ratio_test
                sample_test_num = min(np.sum(sample_test), total_test_num)
                sample_train_num = self.image_per_scene - sample_test_num
                sample_train_num = min(sample_train_num, total_train_num)
                images, images_names = [], []
                cameras["camera_to_worlds"] = []
                # decide background_color
                if self.background_color == "random":
                    background = torch.rand(3)
                else:
                    background = torch.tensor(self.background_color) / 255.0
                if sample_train_num > 0:
                    train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
                    images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
                    images_names.extend([train_imgs_name[i] for i in train_cam_ids])
                    cameras["camera_to_worlds"].append(meta["train_camera_to_worlds"][train_cam_ids])
                if sample_test_num > 0:
                    test_cam_ids = np.random.permutation(total_test_num)[:sample_test_num]
                    images.extend([self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
                    images_names.extend([test_imgs_name[i] for i in test_cam_ids])
                    cameras["camera_to_worlds"].append(meta["test_camera_to_worlds"][test_cam_ids])
                cameras["camera_to_worlds"] = torch.concatenate(cameras["camera_to_worlds"], axis=0)
            elif self.train_or_test == "test":
                assert self.background_color != "random", "For test set, background_color cannot be random"
                background = torch.tensor(self.background_color) / 255.0
                test_cam_ids = np.arange(total_test_num)  # We take all test images
                images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]
                images_names = [test_imgs_name[i] for i in test_cam_ids]
                cameras["camera_to_worlds"] = meta["test_camera_to_worlds"][test_cam_ids]
            else:
                raise ValueError

            for key in ["fx", "fy", "cx", "cy", "width", "height"]:
                cameras[key] = meta[key]
            cameras["background_color"] = background
            output_dict = {
                "gs_params": gs_params,
                "images": images,
                "cameras": cameras,
                "scene_idx": scene["idx"],
                "scene_name": scene_name,
            }
            output_dict["images_name"] = images_names
            yield output_dict

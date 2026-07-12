import glob
import os
import pickle
import random
from typing import Optional

import gin
import numpy as np
import torch
from PIL import Image

from utils.transform_utils import MinMaxScaler, remove_outliers


EXPECTED_IMAGE_COUNT = 128
FACTOR_TO_IMAGE_DIR = {
    1: "images",
    2: "images_2",
    4: "images_4",
}
CAMERA_METADATA_NAME = "camera_for-3d-denoise.pkl"


class _GSCheckpointLoadError(RuntimeError):
    def __init__(self, message, nerfstudio_dir):
        super().__init__(message)
        self.nerfstudio_dir = nerfstudio_dir


@gin.configurable
class SplatFactoMultiLevelDataset(torch.utils.data.IterableDataset):
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
        split_across_gpus: bool,
        factors: list = [1, 2, 4],
        background_color: list = [0, 0, 0],
        skip_invalid_scenes: bool = True,
        cache_steps: Optional[int] = None,
        cache_num_scenes: Optional[int] = None,
    ):
        self.train_or_test = train_or_test
        self.image_per_scene = image_per_scene
        self.sample_ratio_test = sample_ratio_test
        self.factors = factors
        self.primary_factor = self.factors[0]
        self.skip_invalid_scenes = skip_invalid_scenes
        self.skipped_scenes = []
        self._runtime_skipped_scene_indices = set()

        if load_pose_src != "nerfstudio":
            raise ValueError(
                "SplatFactoMultiLevelDataset currently only supports "
                "load_pose_src='nerfstudio'."
            )
        self.load_pose_src = load_pose_src

        self.folders = self._build_scene_pairs(nerfstudio_folder, colmap_folder)

        self.remove_outlier_ndevs = remove_outlier_ndevs
        # Accepted for existing gin configs; scenes are loaded directly each iteration.
        self.split_across_gpus = split_across_gpus
        self.cache_steps = cache_steps
        self.cache_num_scenes = cache_num_scenes
        self.max_gs_num = max_gs_num
        self.background_color = background_color

        if train_or_test in ["test"]:
            # For test set, we need to split data across device deterministically
            self.remaining_scenes = list(range(len(self.folders)))
            # For DDP evaluation, we need to chunk the data
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            else:
                world_size, rank = 1, 0
            chunk_size = len(self.remaining_scenes) // world_size
            if rank == world_size - 1:
                self.remaining_scenes = self.remaining_scenes[rank * chunk_size :]
            else:
                self.remaining_scenes = self.remaining_scenes[rank * chunk_size : (rank + 1) * chunk_size]
        else:
            self.counter = 0


    # ----------- Parsing scene images, cameras and gs ---------------
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

    def _image_dir_for_factor(self, colmap_scene_root: str, factor: int) -> str:
        image_dir_name = FACTOR_TO_IMAGE_DIR.get(factor, f"images_{factor}")
        return os.path.join(colmap_scene_root, image_dir_name)

    def _list_pngs(self, image_dir: str):
        return sorted(
            [
                os.path.join(image_dir, name)
                for name in os.listdir(image_dir)
                if name.lower().endswith(".png")
            ]
        )

    def _validate_scene_structure(self, scene_info):
        scene_name = scene_info["scene_name"]
        colmap_dir = scene_info["colmap_dir"]
        if not os.path.isdir(colmap_dir):
            return f"{scene_name}: missing_colmap_scene:{colmap_dir}"

        factor_paths = scene_info["factor_paths"]
        for factor in self.factors:
            if factor not in FACTOR_TO_IMAGE_DIR:
                return f"{scene_name}: unsupported_factor:{factor}"

            paths = factor_paths[factor]
            nerfstudio_dir = paths["nerfstudio_dir"]
            image_dir = paths["image_dir"]
            expected_image_dir = self._image_dir_for_factor(colmap_dir, factor)

            if image_dir != expected_image_dir:
                return (
                    f"{scene_name}: factor={factor}: unexpected_image_dir: "
                    f"got={image_dir} expected={expected_image_dir}"
                )
            if not os.path.isdir(nerfstudio_dir):
                return f"{scene_name}: factor={factor}: missing_nerfstudio_dir:{nerfstudio_dir}"
            if not os.path.isdir(image_dir):
                return f"{scene_name}: factor={factor}: missing_image_dir:{image_dir}"

            image_paths = self._list_pngs(image_dir)
            if len(image_paths) != EXPECTED_IMAGE_COUNT:
                return (
                    f"{scene_name}: factor={factor}: bad_image_count: "
                    f"got={len(image_paths)} expected={EXPECTED_IMAGE_COUNT} dir={image_dir}"
                )

            models_dir = os.path.join(nerfstudio_dir, "nerfstudio_models")
            if not os.path.isdir(models_dir):
                return f"{scene_name}: factor={factor}: missing_nerfstudio_models:{models_dir}"
            if len(glob.glob(os.path.join(models_dir, "step-*.ckpt"))) == 0:
                return f"{scene_name}: factor={factor}: missing_ckpt:{models_dir}"

            camera_path = os.path.join(nerfstudio_dir, CAMERA_METADATA_NAME)
            if not os.path.isfile(camera_path):
                return f"{scene_name}: factor={factor}: missing_camera_metadata:{camera_path}"

        return None

    def _print_skip_summary(self, messages):
        for message in messages:
            print(f"[SplatFactoMultiLevelDataset] {message}")

    def _build_scene_pairs(self, nerfstudio_folder: str, colmap_folder: str):
        ns_roots = self._load_scene_roots(nerfstudio_folder, "nerfstudio")
        colmap_roots = self._load_scene_roots(colmap_folder, "colmap")
        ns_map = self._scene_map(ns_roots, "nerfstudio")
        colmap_map = self._scene_map(colmap_roots, "colmap")

        ns_names = set(ns_map.keys())
        colmap_names = set(colmap_map.keys())
        common_names = sorted(ns_names & colmap_names)
        missing_in_colmap = sorted(ns_names - colmap_names)
        missing_in_nerfstudio = sorted(colmap_names - ns_names)

        messages = []
        if missing_in_colmap:
            preview = missing_in_colmap[:10]
            message = f"skipping {len(missing_in_colmap)} scenes missing in colmap, e.g. {preview}"
            if not self.skip_invalid_scenes:
                raise ValueError(message)
            self.skipped_scenes.extend(
                {
                    "scene_name": scene_name,
                    "reason": "missing_in_colmap",
                    "skip_reason": "missing_in_colmap",
                    "exception_reason": None,
                    "nerfstudio_dir": ns_map[scene_name],
                    "colmap_dir": None,
                }
                for scene_name in missing_in_colmap
            )
            messages.append(message)
        if missing_in_nerfstudio:
            preview = missing_in_nerfstudio[:10]
            message = f"skipping {len(missing_in_nerfstudio)} scenes missing in nerfstudio, e.g. {preview}"
            if not self.skip_invalid_scenes:
                raise ValueError(message)
            self.skipped_scenes.extend(
                {
                    "scene_name": scene_name,
                    "reason": "missing_in_nerfstudio",
                    "skip_reason": "missing_in_nerfstudio",
                    "exception_reason": None,
                    "nerfstudio_dir": None,
                    "colmap_dir": colmap_map[scene_name],
                }
                for scene_name in missing_in_nerfstudio
            )
            messages.append(message)

        folders = []
        invalid_messages = []
        for scene_name in common_names:
            ns_scene_root = ns_map[scene_name]
            colmap_scene_root = colmap_map[scene_name]
            factor_paths = {}
            for factor in self.factors:
                factor_paths[factor] = {
                    "nerfstudio_dir": os.path.join(ns_scene_root, f"df-{factor}", "splatfacto"),
                    "image_dir": self._image_dir_for_factor(colmap_scene_root, factor),
                }

            scene_info = {
                "scene_name": scene_name,
                "colmap_dir": colmap_scene_root,
                "factor_paths": factor_paths,
            }
            problem = self._validate_scene_structure(scene_info)
            if problem is not None:
                if not self.skip_invalid_scenes:
                    raise ValueError(problem)
                invalid_messages.append(problem)
                self.skipped_scenes.append(
                    {
                        "scene_name": scene_name,
                        "reason": problem,
                        "skip_reason": problem,
                        "exception_reason": None,
                        "nerfstudio_dir": ns_scene_root,
                        "colmap_dir": colmap_scene_root,
                    }
                )
                continue

            folders.append(scene_info)

        if invalid_messages:
            preview = invalid_messages[:20]
            messages.append(
                f"skipping {len(invalid_messages)} structurally invalid scenes:\n"
                + "\n".join(preview)
            )
            if len(invalid_messages) > 20:
                messages.append(f"... and {len(invalid_messages) - 20} more invalid scenes")

        if len(folders) == 0:
            details = "\n".join(messages) if messages else "No shared valid scenes found."
            raise ValueError(f"No scenes found for SplatFactoMultiLevelDataset.\n{details}")

        messages.append(f"using {len(folders)} shared valid scenes")
        self._print_skip_summary(messages)
        return folders
    # ----------------------------------------------------------------------------

    def refresh_remaining_training(self):
        if self.split_across_gpus:
            self.random_split_to_remaining()
        else:
            self.remaining_scenes = self.get_thisworker_split(N=len(self.folders))
            random.shuffle(self.remaining_scenes)
        self.remaining_scenes = [
            idx for idx in self.remaining_scenes if idx not in self._runtime_skipped_scene_indices
        ]
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
        """Generate a new permutation of the folders."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size, rank = 1, 0
        permutation = np.random.RandomState(self.counter).permutation(len(self.folders))
        pad_num = (world_size - len(self.folders) % world_size) % world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])

        chunk_size = len(permutation) // world_size
        if rank == world_size - 1:
            remaining_scenes_for_thisprocess = permutation[rank * chunk_size :]
        else:
            remaining_scenes_for_thisprocess = permutation[rank * chunk_size : (rank + 1) * chunk_size]

        split_id = self.get_thisworker_split(N=len(remaining_scenes_for_thisprocess))
        self.remaining_scenes = [remaining_scenes_for_thisprocess[i] for i in split_id]
        return

    @gin.configurable
    def read_image(self, path, background):
        try:
            pil_image = Image.open(path)
        except Exception:
            print(f"Warning: {path} cannot be opened")
            raise

        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255.0
        if "real" in path.lower():
            possible_mask_filename = path.replace("images", "masks")
            if os.path.exists(possible_mask_filename):
                mask = np.array(Image.open(possible_mask_filename)).astype(image.dtype) / 255.0
                mask = torch.from_numpy(mask)
            else:
                mask = None
        else:
            mask = None

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + background * (1.0 - image[:, :, -1:])
        elif mask is not None:
            image_rgb = image * mask[..., None] + background * (1.0 - mask[..., None])
            image = torch.concat([image_rgb, mask[..., None]], axis=-1)
        return image

    def load_gs_params_fromnerfstudio(self, nerfstudio_dir, idx):
        skip_params = gin.query_parameter("FeaturePredictor.input_features")
        if gin.query_parameter("training.pretrain_steps") > 0:
            skip_params = skip_params + gin.query_parameter("create_pseudo_target.take_from_input")

        ckpt_list = sorted(
            glob.glob(nerfstudio_dir + "/nerfstudio_models/step-*.ckpt"),
            key=lambda path: int(os.path.splitext(os.path.basename(path))[0].split("-")[-1]),
        )
        if len(ckpt_list) == 0:
            raise FileNotFoundError(
                f"{nerfstudio_dir} does not have nerfstudio_models/step-*.ckpt"
            )
        ckpt_file = ckpt_list[-1]

        try:
            ckpt = torch.load(ckpt_file, map_location="cpu")
        except Exception as exc:
            scene_name = self.folders[idx]["scene_name"] if idx < len(self.folders) else str(idx)
            raise _GSCheckpointLoadError(
                f"Failed to load GS checkpoint for scene_idx={idx} "
                f"scene_name={scene_name} ckpt_file={ckpt_file}",
                nerfstudio_dir,
            ) from exc
        ckpt = {k.replace("_model.gauss_params.", ""): v for k, v in ckpt.items() if "gauss_params" in k}
        gs_params = {k: ckpt[k] for k in set(skip_params)}

        select = torch.ones(gs_params["means"].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key == "features_rest":
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        if self.remove_outlier_ndevs > 0:
            _, inlier_mask = remove_outliers(gs_params["means"], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        N = gs_params["means"].shape[0]
        if N > self.max_gs_num:
            inlier_mask = torch.zeros(N, dtype=torch.bool)
            inlier_mask[: self.max_gs_num] = True
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

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


    def _tensorize_camera_meta(self, meta):
        for key in [
            "train_camera_to_worlds",
            "test_camera_to_worlds",
            "camera_to_worlds",
            "fx",
            "fy",
            "cx",
            "cy",
            "width",
            "height",
        ]:
            if key in meta:
                meta[key] = torch.as_tensor(meta[key], dtype=torch.float32)
        return meta


    def load_images_cameras_fromnerfstudio(self, nerfstudio_dir, colmap_dir, image_dir):
        del colmap_dir
        with open(os.path.join(nerfstudio_dir, "camera_for-3d-denoise.pkl"), "rb") as f:
            meta = pickle.load(f)

        imgs_name = sorted(os.listdir(image_dir))
        imgs_path = [os.path.join(image_dir, name) for name in imgs_name]
        meta["camera_to_worlds"] = meta["train_camera_to_worlds"]
        meta = self._tensorize_camera_meta(meta)
        return meta, imgs_path


    def _validate_pose_image_count(self, factor, factor_entry):
        meta = factor_entry["meta"]
        if len(factor_entry["imgs_name"]) != len(meta["camera_to_worlds"]):
            raise ValueError(
                f"Factor {factor}: images count ({len(factor_entry['imgs_name'])}) "
                f"!= camera_to_worlds ({len(meta['camera_to_worlds'])})"
            )


    def _validate_factor_alignment(self, factor_data):
        key_name = "imgs_name"
        key_path = "imgs_path"
        key_pose = "camera_to_worlds"

        ref_names = factor_data[self.primary_factor][key_name]
        common_names = []
        for name in ref_names:
            keep = True
            for factor in self.factors:
                if name not in factor_data[factor][key_name]:
                    keep = False
                    break
            if keep:
                common_names.append(name)

        for factor in self.factors:
            names = factor_data[factor][key_name]
            name_to_idx = {name: i for i, name in enumerate(names)}
            keep_idx = [name_to_idx[name] for name in common_names]
            factor_data[factor][key_path] = [factor_data[factor][key_path][i] for i in keep_idx]
            factor_data[factor][key_name] = [factor_data[factor][key_name][i] for i in keep_idx]
            factor_data[factor]["meta"][key_pose] = factor_data[factor]["meta"][key_pose][keep_idx]

    def load_scene(self, idx):
        scene_info = self.folders[idx]
        colmap_dir = scene_info["colmap_dir"]
        factor_data = {}

        for factor in self.factors:
            paths = scene_info["factor_paths"][factor]
            nerfstudio_dir = paths["nerfstudio_dir"]
            image_dir = paths["image_dir"]

            gs_params, scaler = self.load_gs_params_fromnerfstudio(nerfstudio_dir, idx)
            meta, imgs_path = self.load_images_cameras_fromnerfstudio(
                nerfstudio_dir, colmap_dir, image_dir
            )
            meta["camera_to_worlds"][:, :3, -1] = scaler.transform(meta["camera_to_worlds"][:, :3, -1])

            factor_entry = {
                "gs_params": gs_params,
                "meta": meta,
                "imgs_path": imgs_path,
                "imgs_name": [os.path.basename(path) for path in imgs_path],
                "scaler": scaler,
            }
            self._validate_pose_image_count(factor, factor_entry)
            factor_data[factor] = factor_entry

        self._validate_factor_alignment(factor_data)

        return {
            "idx": idx,
            "scene_name": scene_info["scene_name"],
            "factor_data": factor_data,
        }


    def build_background(self):
        if self.background_color == "random":
            return torch.rand(3)
        return torch.tensor(self.background_color, dtype=torch.float32) / 255.0


    def load_factor_views(self, factor_entry, cam_ids=None, background=None):
        meta = factor_entry["meta"]
        total_num = len(meta["camera_to_worlds"])
        if total_num == 0:
            raise ValueError("Factor entry has zero views")

        if cam_ids is None:
            cam_ids = list(range(total_num))
        else:
            cam_ids = list(cam_ids)
        if background is None:
            background = self.build_background()

        imgs_path = factor_entry["imgs_path"]
        imgs_name = factor_entry["imgs_name"]
        images = [self.read_image(imgs_path[i], background=background) for i in cam_ids]
        images_names = [imgs_name[i] for i in cam_ids]

        cameras = {
            "camera_to_worlds": meta["camera_to_worlds"][cam_ids],
            "fx": meta["fx"],
            "fy": meta["fy"],
            "cx": meta["cx"],
            "cy": meta["cy"],
            "width": meta["width"],
            "height": meta["height"],
            "background_color": background,
        }
        return images, images_names, cameras


    def _prepare_factor_payload(self, factor_entry, background, cam_ids):
        images, images_names, cameras = self.load_factor_views(
            factor_entry, cam_ids=cam_ids, background=background
        )
        return {
            "gs_params": factor_entry["gs_params"],
            "scaler": factor_entry["scaler"],
            "images": images,
            "cameras": cameras,
            "images_name": images_names,
        }


    def _skip_checkpoint_load_failure(self, scene_idx, exc):
        self._runtime_skipped_scene_indices.add(scene_idx)
        self.remaining_scenes = [idx for idx in self.remaining_scenes if idx != scene_idx]

        cause = exc.__cause__
        exception_reason = (
            f"{type(cause).__name__}: {cause}" if cause is not None else "unknown"
        )
        scene_info = self.folders[scene_idx]
        skip_reason = "checkpoint_load_error"
        self.skipped_scenes.append(
            {
                "scene_name": scene_info["scene_name"],
                "reason": skip_reason,
                "skip_reason": skip_reason,
                "exception_reason": exception_reason,
                "nerfstudio_dir": exc.nerfstudio_dir,
                "colmap_dir": scene_info["colmap_dir"],
            }
        )
        print(
            f"[SplatFactoMultiLevelDataset] skipping scene_idx={scene_idx} "
            f"scene_name={scene_info['scene_name']}: {exc}; cause={exception_reason}",
            flush=True,
        )


    def __iter__(self):
        if self.train_or_test == "train":
            self.refresh_remaining_training()

        while len(self.remaining_scenes) > 0:
            scene_idx = self.remaining_scenes.pop(0)
            if self.train_or_test == "train" and len(self.remaining_scenes) == 0:
                self.refresh_remaining_training()
            try:
                scene = self.load_scene(scene_idx)
            except _GSCheckpointLoadError as exc:
                self._skip_checkpoint_load_failure(scene_idx, exc)
                continue
            factor_data = scene["factor_data"]
            primary_entry = factor_data[self.primary_factor]
            total_num = len(primary_entry["meta"]["camera_to_worlds"])

            background = self.build_background()

            if self.train_or_test == "train":
                if self.image_per_scene is None:
                    sample_num = total_num
                else:
                    sample_num = min(self.image_per_scene, total_num)
                cam_ids = np.random.permutation(total_num)[:sample_num]
            elif self.train_or_test == "test":
                assert self.background_color != "random", "For test set, background_color cannot be random"
                cam_ids = np.arange(total_num)
            else:
                raise ValueError

            multilevel = {}
            for factor in self.factors:
                multilevel[factor] = self._prepare_factor_payload(
                    factor_data[factor],
                    background=background,
                    cam_ids=cam_ids,
                )

            yield {
                "scene_idx": scene["idx"],
                "scene_name": scene["scene_name"],
                "factors": self.factors,
                "multilevel": multilevel,
            }


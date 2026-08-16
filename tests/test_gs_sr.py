from pathlib import Path

import numpy as np
import pytest
import torch

from dataset.GS_SR import SplatFactoSRDataset


class _IdentityScaler:
    def transform(self, value):
        return value


def _make_dataset(tmp_path, scene_lines="scene_b\n\nscene_a\n", **overrides):
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene_lines, encoding="utf-8")
    kwargs = {
        "train_or_test": "test",
        "dataset_root": str(tmp_path / "dataset"),
        "scene_list": str(scene_list),
        "image_per_scene": None,
        "remove_outlier_ndevs": -1,
        "max_gs_num": 1000,
        "split_across_gpus": False,
        "resolutions": [512, 128],
        "background_color": [0, 0, 0],
    }
    kwargs.update(overrides)
    return SplatFactoSRDataset(**kwargs)


def _resolution_entry(resolution, image_names=("000.png", "001.png", "002.png")):
    return {
        "gs_params": {"resolution": resolution},
        "scaler": _IdentityScaler(),
        "meta": {
            "camera_to_worlds": torch.arange(36, dtype=torch.float32).reshape(3, 3, 4),
            "fx": torch.tensor(float(resolution)),
            "fy": torch.tensor(float(resolution)),
            "cx": torch.tensor(float(resolution) / 2),
            "cy": torch.tensor(float(resolution) / 2),
            "width": torch.tensor(float(resolution)),
            "height": torch.tensor(float(resolution)),
        },
        "imgs_path": [f"/{resolution}/images/{name}" for name in image_names],
        "imgs_name": list(image_names),
    }


def _loaded_scene(scene_idx=0):
    return {
        "idx": scene_idx,
        "scene_name": "scene_b",
        "resolution_data": {
            512: _resolution_entry(512),
            128: _resolution_entry(128),
        },
    }


def test_manifest_order_blank_lines_and_native_resolution_paths(tmp_path):
    dataset = _make_dataset(tmp_path)

    assert [entry["scene_name"] for entry in dataset.folders] == ["scene_b", "scene_a"]
    paths = dataset.folders[0]["resolution_paths"]
    assert Path(paths[512]["image_dir"]) == (
        tmp_path / "dataset" / "512" / "colmap" / "scene_b" / "images"
    )
    assert Path(paths[128]["nerfstudio_dir"]) == (
        tmp_path / "dataset" / "128" / "nerfstudio" / "scene_b" / "splatfacto"
    )


def test_missing_manifest_raises_naturally(tmp_path):
    with pytest.raises(FileNotFoundError):
        SplatFactoSRDataset(
            train_or_test="test",
            dataset_root=str(tmp_path),
            scene_list=str(tmp_path / "missing.txt"),
            image_per_scene=None,
            remove_outlier_ndevs=-1,
            max_gs_num=1000,
            split_across_gpus=False,
        )


def test_training_sample_uses_shared_views_background_and_new_schema(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(
        tmp_path,
        scene_lines="scene_b\n",
        train_or_test="train",
        image_per_scene=2,
        background_color="random",
    )
    monkeypatch.setattr(dataset, "load_scene", lambda _: _loaded_scene())
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([2, 0, 1]))
    monkeypatch.setattr(torch, "rand", lambda _: torch.tensor([0.2, 0.3, 0.4]))
    monkeypatch.setattr(
        dataset,
        "read_image",
        lambda path, background: (Path(path).name, background),
    )

    sample = next(iter(dataset))

    assert set(sample) == {"scene_idx", "scene_name", "resolutions", "multiresolution"}
    assert sample["resolutions"] == [512, 128]
    high = sample["multiresolution"][512]
    low = sample["multiresolution"][128]
    assert high["images_name"] == ["002.png", "000.png"]
    assert low["images_name"] == high["images_name"]
    assert high["cameras"]["background_color"] is low["cameras"]["background_color"]
    torch.testing.assert_close(
        high["cameras"]["camera_to_worlds"], low["cameras"]["camera_to_worlds"]
    )


def test_test_sample_returns_every_view(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, scene_lines="scene_b\n")
    monkeypatch.setattr(dataset, "load_scene", lambda _: _loaded_scene())
    monkeypatch.setattr(dataset, "read_image", lambda path, background: Path(path).name)

    sample = next(iter(dataset))

    assert sample["multiresolution"][512]["images_name"] == [
        "000.png",
        "001.png",
        "002.png",
    ]
    assert sample["multiresolution"][128]["images_name"] == [
        "000.png",
        "001.png",
        "002.png",
    ]


def test_test_scenes_are_partitioned_deterministically(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    dataset = _make_dataset(
        tmp_path,
        scene_lines="scene_0\nscene_1\nscene_2\nscene_3\nscene_4\n",
    )

    assert dataset.remaining_scenes == [2, 3, 4]


def test_load_scene_rejects_cross_resolution_name_mismatch(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, scene_lines="scene_b\n")
    monkeypatch.setattr(
        dataset,
        "load_gs_params_fromnerfstudio",
        lambda nerfstudio_dir, scene_idx: ({}, _IdentityScaler()),
    )

    def fake_camera_loader(nerfstudio_dir, image_dir):
        resolution = 512 if "/512/" in image_dir else 128
        names = ["000.png", "001.png"]
        if resolution == 128:
            names[-1] = "different.png"
        meta = {
            "camera_to_worlds": torch.zeros((2, 3, 4)),
            "fx": torch.tensor(1.0),
            "fy": torch.tensor(1.0),
            "cx": torch.tensor(1.0),
            "cy": torch.tensor(1.0),
            "width": torch.tensor(float(resolution)),
            "height": torch.tensor(float(resolution)),
        }
        return meta, [str(Path(image_dir) / name) for name in names]

    monkeypatch.setattr(dataset, "load_images_cameras_fromnerfstudio", fake_camera_loader)

    with pytest.raises(ValueError, match="image names do not match"):
        dataset.load_scene(0)


def test_load_scene_rejects_image_pose_count_mismatch(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, scene_lines="scene_b\n")
    monkeypatch.setattr(
        dataset,
        "load_gs_params_fromnerfstudio",
        lambda nerfstudio_dir, scene_idx: ({}, _IdentityScaler()),
    )

    def fake_camera_loader(nerfstudio_dir, image_dir):
        meta = {
            "camera_to_worlds": torch.zeros((2, 3, 4)),
            "fx": torch.tensor(1.0),
            "fy": torch.tensor(1.0),
            "cx": torch.tensor(1.0),
            "cy": torch.tensor(1.0),
            "width": torch.tensor(512.0),
            "height": torch.tensor(512.0),
        }
        image_paths = [
            str(Path(image_dir) / name)
            for name in ("000.png", "001.png", "002.png")
        ]
        return meta, image_paths

    monkeypatch.setattr(dataset, "load_images_cameras_fromnerfstudio", fake_camera_loader)

    with pytest.raises(ValueError, match="does not match pose count"):
        dataset.load_scene(0)


def test_missing_and_corrupt_checkpoints_raise(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, scene_lines="scene_b\n")
    monkeypatch.setattr(
        "dataset.GS_SR.gin.query_parameter",
        lambda name: [] if name != "training.pretrain_steps" else 0,
    )
    nerfstudio_dir = tmp_path / "splatfacto"
    model_dir = nerfstudio_dir / "nerfstudio_models"
    model_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        dataset.load_gs_params_fromnerfstudio(str(nerfstudio_dir), 0)

    (model_dir / "step-000000001.ckpt").write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Failed to load GS checkpoint"):
        dataset.load_gs_params_fromnerfstudio(str(nerfstudio_dir), 0)

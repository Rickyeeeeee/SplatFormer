from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset.GS_SR import SplatFactoSRDataset


def _write_manifest(path, scenes):
    path.write_text(
        "scene_id,score\n" + "".join(f"{scene},1\n" for scene in scenes),
        encoding="utf-8",
    )


def _splats(count, offset=0.0):
    means = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10
    return {
        "means": means + offset,
        "opacities": torch.arange(count, dtype=torch.float32) + offset,
        "quats": torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
        "scales": torch.full((count, 3), -2.0 + offset),
        "sh0": torch.full((count, 1, 3), offset),
        "shN": torch.full((count, 3, 3), offset),
    }


def _write_checkpoint(gsplat_dir, splats, step):
    checkpoint_dir = gsplat_dir / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"ckpt_{step}_rank0.pt"
    torch.save({"step": step, "splats": splats}, path)
    return path


def _write_colmap_scene(root, split, scene, resolution, names):
    scene_dir = (
        root
        / f"{split}-set"
        / "objaverse"
        / str(resolution)
        / "colmap"
        / scene
    )
    image_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (resolution, resolution)).save(image_dir / name)
    (sparse_dir / "cameras.txt").write_text(
        f"1 SIMPLE_PINHOLE {resolution} {resolution} "
        f"{float(resolution)} {resolution / 2} {resolution / 2}\n",
        encoding="utf-8",
    )
    lines = []
    for image_id, name in enumerate(reversed(names), start=1):
        lines.extend(
            [
                f"{image_id} 1 0 0 0 0 0 3 1 {name}",
                "",
            ]
        )
    (sparse_dir / "images.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_scene(
    root,
    split,
    scene,
    source_splats=None,
    target_splats=None,
    fitted_splats=None,
    source_names=("001.png", "000.png"),
    target_names=("001.png", "000.png"),
):
    source_splats = source_splats or _splats(3)
    target_splats = target_splats or _splats(4, offset=10.0)
    fitted_splats = fitted_splats or _splats(3, offset=1.0)
    for resolution, splats, names in (
        (128, source_splats, source_names),
        (512, target_splats, target_names),
    ):
        _write_colmap_scene(root, split, scene, resolution, names)
        gsplat_dir = (
            root
            / f"{split}-set"
            / "objaverse"
            / str(resolution)
            / "gsplat"
            / scene
        )
        _write_checkpoint(gsplat_dir, splats, 1)
    fitted_dir = (
        root
        / f"{split}-set-4x-up"
        / "objaverse"
        / "128"
        / "gsplat"
        / scene
    )
    _write_checkpoint(fitted_dir, fitted_splats, 1)


def _make_dataset(tmp_path, scenes=("scene_a",), split="test", **overrides):
    manifest = tmp_path / "scenes.csv"
    _write_manifest(manifest, scenes)
    kwargs = {
        "train_or_test": split,
        "dataset_root": str(tmp_path),
        "scene_list": str(manifest),
        "image_per_scene": None,
        "remove_outlier_ndevs": -1,
        "max_gs_num": 1000,
        "split_across_gpus": False,
        "resolutions": [128, 512],
        "background_color": [0, 0, 0],
    }
    kwargs.update(overrides)
    return SplatFactoSRDataset(**kwargs)


def test_csv_manifest_preserves_order_and_filters_incomplete_scenes(tmp_path):
    _write_scene(tmp_path, "test", "scene_b")
    _write_scene(tmp_path, "test", "scene_a")
    dataset = _make_dataset(
        tmp_path, scenes=("scene_b", "missing_scene", "scene_a")
    )

    assert [entry["scene_name"] for entry in dataset.folders] == [
        "scene_b",
        "scene_a",
    ]
    assert dataset.filtered_scenes[0]["scene_name"] == "missing_scene"
    paths = dataset.folders[0]["resolution_paths"][128]
    assert Path(paths["image_dir"]) == (
        tmp_path / "test-set" / "objaverse" / "128" / "colmap"
        / "scene_b" / "images"
    )
    assert Path(dataset.folders[0]["fit_lr_to_hr_paths"]["gsplat_dir"]) == (
        tmp_path / "test-set-4x-up" / "objaverse" / "128" / "gsplat" / "scene_b"
    )


def test_manifest_requires_scene_id_and_at_least_one_complete_scene(tmp_path):
    bad_manifest = tmp_path / "bad.csv"
    bad_manifest.write_text("name\nscene_a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scene_id"):
        SplatFactoSRDataset(
            train_or_test="test",
            dataset_root=str(tmp_path),
            scene_list=str(bad_manifest),
            image_per_scene=None,
            remove_outlier_ndevs=-1,
            max_gs_num=100,
            split_across_gpus=False,
        )

    manifest = tmp_path / "missing.csv"
    _write_manifest(manifest, ["missing"])
    with pytest.raises(ValueError, match="No complete scenes"):
        SplatFactoSRDataset(
            train_or_test="test",
            dataset_root=str(tmp_path),
            scene_list=str(manifest),
            image_per_scene=None,
            remove_outlier_ndevs=-1,
            max_gs_num=100,
            split_across_gpus=False,
        )


def test_latest_gsplat_checkpoint_maps_feature_shapes(tmp_path):
    _write_scene(tmp_path, "test", "scene_a")
    gsplat_dir = (
        tmp_path / "test-set" / "objaverse" / "128" / "gsplat" / "scene_a"
    )
    latest = _write_checkpoint(gsplat_dir, _splats(5, offset=2.0), 20)
    fitted_dir = (
        tmp_path
        / "test-set-4x-up"
        / "objaverse"
        / "128"
        / "gsplat"
        / "scene_a"
    )
    _write_checkpoint(fitted_dir, _splats(5, offset=3.0), 20)
    dataset = _make_dataset(tmp_path)
    scene = dataset.load_scene(0)
    source = scene["resolution_data"][128]

    assert source["checkpoint_path"] == str(latest)
    assert source["gs_params"]["features_dc"].shape == (5, 3)
    assert source["gs_params"]["features_rest"].shape == (5, 3, 3)
    assert source["gs_params"]["opacities"].shape == (5, 1)


def test_fitted_pair_uses_shared_mask_and_target_frame(tmp_path):
    source = _splats(4)
    fitted = _splats(4, offset=1.0)
    fitted["opacities"][1] = float("nan")
    target = _splats(5, offset=10.0)
    _write_scene(
        tmp_path,
        "test",
        "scene_a",
        source_splats=source,
        target_splats=target,
        fitted_splats=fitted,
    )
    dataset = _make_dataset(tmp_path)
    scene = dataset.load_scene(0)

    source_entry = scene["resolution_data"][128]
    fitted_entry = scene["fit_lr_to_hr"]
    target_scaler = scene["resolution_data"][512]["scaler"]
    assert source_entry["gs_params"]["means"].shape[0] == 3
    assert fitted_entry["gs_params"]["means"].shape[0] == 3
    expected_fitted_means = target_scaler.transform(fitted["means"][[0, 2, 3]])
    torch.testing.assert_close(
        fitted_entry["gs_params"]["means"], expected_fitted_means
    )
    assert fitted_entry["coordinate_frame"] == "target_resolution"


def test_fitted_pair_rejects_shape_mismatch(tmp_path):
    _write_scene(
        tmp_path,
        "test",
        "scene_a",
        source_splats=_splats(3),
        fitted_splats=_splats(4),
    )
    dataset = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        dataset.load_scene(0)


def test_colmap_views_are_name_sorted_and_cross_resolution_aligned(tmp_path):
    _write_scene(tmp_path, "test", "scene_a")
    dataset = _make_dataset(tmp_path)
    scene = dataset.load_scene(0)
    assert scene["resolution_data"][128]["imgs_name"] == ["000.png", "001.png"]
    assert scene["resolution_data"][512]["imgs_name"] == ["000.png", "001.png"]

    _write_scene(
        tmp_path,
        "test",
        "scene_a",
        target_names=("002.png", "000.png"),
    )
    dataset = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="image names do not match"):
        dataset.load_scene(0)


def test_iterator_returns_resolution_and_fitted_payloads(tmp_path, monkeypatch):
    _write_scene(tmp_path, "train", "scene_a")
    dataset = _make_dataset(
        tmp_path,
        split="train",
        image_per_scene=1,
        background_color="random",
    )
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))
    monkeypatch.setattr(torch, "rand", lambda _: torch.tensor([0.2, 0.3, 0.4]))
    sample = next(iter(dataset))

    assert set(sample) == {
        "scene_idx",
        "scene_name",
        "resolutions",
        "multiresolution",
        "fit_lr_to_hr",
    }
    assert sample["multiresolution"][128]["images_name"] == ["001.png"]
    assert sample["multiresolution"][512]["images_name"] == ["001.png"]
    assert sample["fit_lr_to_hr"]["source_resolution"] == 128
    assert sample["fit_lr_to_hr"]["target_resolution"] == 512


def test_test_scenes_are_partitioned_deterministically(tmp_path, monkeypatch):
    scenes = [f"scene_{index}" for index in range(5)]
    for scene in scenes:
        _write_scene(tmp_path, "test", scene)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    dataset = _make_dataset(tmp_path, scenes=scenes)
    assert dataset.remaining_scenes == [2, 3, 4]

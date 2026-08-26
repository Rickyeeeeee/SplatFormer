import math

import pytest
import torch

from dataset.GS_SR import SplatFactoSRDataset
from sr import alignment
from sr.alignment import build_precomputed_fit_pair
from utils.gs_utils import convert_gaussian_frame


class IdentityScaler:
    scale_ = torch.tensor(1.0)
    trans_ = torch.zeros(3)


class Scaler:
    def __init__(self, scale, translation):
        self.scale_ = torch.tensor(scale, dtype=torch.float32)
        self.trans_ = torch.tensor(translation, dtype=torch.float32)


def resolution_entry(count):
    return {
        "gs_params": {
            "means": torch.arange(
                count * 3, dtype=torch.float32
            ).reshape(count, 3),
            "scales": torch.zeros((count, 3)),
        },
        "scaler": IdentityScaler(),
    }


def test_precomputed_fit_pair_uses_dataset_target_and_provenance():
    source_entry = resolution_entry(3)
    target_entry = resolution_entry(4)
    fitted = {
        key: value + 1.0
        for key, value in source_entry["gs_params"].items()
    }
    scene = {
        "fit_lr_to_hr": {
            "source_resolution": 128,
            "target_resolution": 512,
            "gs_params": fitted,
            "checkpoint_path": "/fit/ckpt_2999_rank0.pt",
        }
    }

    source, target, provenance = build_precomputed_fit_pair(
        scene,
        source_entry,
        target_entry,
        input_resolution=128,
        target_resolution=512,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        source["means"], source_entry["gs_params"]["means"]
    )
    torch.testing.assert_close(target["means"], fitted["means"])
    assert provenance == {
        "status": "dataset_precomputed",
        "checkpoint_path": "/fit/ckpt_2999_rank0.pt",
    }


def test_precomputed_fit_pair_validates_resolution_and_identity_shape():
    source_entry = resolution_entry(3)
    target_entry = resolution_entry(4)
    scene = {
        "fit_lr_to_hr": {
            "source_resolution": 128,
            "target_resolution": 512,
            "gs_params": resolution_entry(2)["gs_params"],
            "checkpoint_path": "/fit/checkpoint.pt",
        }
    }
    with pytest.raises(ValueError, match="supports 128->512"):
        build_precomputed_fit_pair(
            scene,
            source_entry,
            target_entry,
            256,
            512,
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="identity-paired"):
        build_precomputed_fit_pair(
            scene,
            source_entry,
            target_entry,
            128,
            512,
            torch.device("cpu"),
        )


def test_dataset_scene_index_rejects_filtered_or_unknown_scene():
    dataset = type(
        "Dataset",
        (),
        {"folders": [{"scene_name": "available"}]},
    )()
    assert SplatFactoSRDataset.scene_index(dataset, "") == 0
    assert SplatFactoSRDataset.scene_index(dataset, "available") == 0
    with pytest.raises(ValueError, match="absent or was filtered"):
        SplatFactoSRDataset.scene_index(dataset, "filtered")


def test_convert_gaussian_frame_updates_means_and_log_scales():
    gaussians = {
        "means": torch.tensor([[3.0, 4.0, 5.0]]),
        "scales": torch.zeros((1, 3)),
        "opacities": torch.ones((1, 1)),
    }
    source_scaler = Scaler(2.0, [1.0, 2.0, 3.0])
    target_scaler = Scaler(4.0, [-1.0, -2.0, -3.0])

    converted = convert_gaussian_frame(
        gaussians, source_scaler, target_scaler
    )

    torch.testing.assert_close(
        converted["means"], torch.tensor([[3.0, 2.0, 1.0]])
    )
    torch.testing.assert_close(
        converted["scales"],
        torch.full((1, 3), math.log(2.0)),
    )
    torch.testing.assert_close(
        converted["opacities"], gaussians["opacities"]
    )
    assert converted["opacities"] is not gaussians["opacities"]


class DummyLogger:
    def info(self, *args, **kwargs):
        del args, kwargs


class DummyDataset:
    image_per_scene = 1

    def load_resolution_views(self, resolution_entry):
        del resolution_entry
        return [torch.zeros((2, 2, 3))], ["000.png"], {}


def prepare_kwargs():
    source_entry = resolution_entry(2)
    target_entry = resolution_entry(2)
    return {
        "dataset": DummyDataset(),
        "scene": {
            "scene_name": "scene_a",
            "fit_lr_to_hr": {
                "source_resolution": 128,
                "target_resolution": 512,
                "gs_params": {
                    key: value + 1.0
                    for key, value in source_entry["gs_params"].items()
                },
                "checkpoint_path": "/fit/checkpoint.pt",
            },
        },
        "input_resolution_entry": source_entry,
        "target_resolution_entry": target_entry,
        "target_images": [torch.zeros((2, 2, 3))],
        "target_cameras": {},
        "target_gs": target_entry["gs_params"],
        "output_dir": "/unused",
        "logger": DummyLogger(),
        "device": torch.device("cpu"),
        "eval_chunk_size": 1,
        "attribute_init": "aligned",
        "emd_eps": 0.01,
        "emd_iters": 100,
        "input_resolution": 128,
        "target_resolution": 512,
        "matching_cache_root": "/cache",
        "force_matching_fit": False,
        "matching_config": {"total_steps": 10, "image_per_step": 1},
        "matching_optimizer_factory": lambda gaussians: gaussians,
    }


@pytest.mark.parametrize("mode", ["emd", "random"])
def test_prepare_alignment_densification_modes(mode, monkeypatch):
    kwargs = prepare_kwargs()
    base = {
        key: value.clone()
        for key, value in kwargs["input_resolution_entry"]["gs_params"].items()
    }
    stages = {
        "00_low_res_gs.ply": base,
        "01_interpolated_high_res_gs.ply": base,
        "02_gt_high_res_gs.ply": kwargs["target_gs"],
        "03_input_high_res_gs.ply": base,
    }
    calls = []

    def build_densified_input(**arguments):
        calls.append(arguments)
        return (
            {key: value.clone() for key, value in base.items()},
            dict(stages),
        )

    monkeypatch.setattr(
        alignment.densification,
        "build_densified_input",
        build_densified_input,
    )
    monkeypatch.setattr(
        alignment.densification,
        "save_densification_stages",
        lambda **arguments: None,
    )
    monkeypatch.setattr(
        alignment,
        "write_densify_stage_render_metrics",
        lambda **arguments: {},
    )
    if mode == "random":
        monkeypatch.setattr(
            alignment.torch,
            "randperm",
            lambda count, device: torch.tensor([1, 0], device=device),
        )

    aligned_input, aligned_target, info = alignment.prepare_alignment(
        alignment=mode, **kwargs
    )

    expected_alignment = "emd" if mode == "emd" else "none"
    assert calls[0]["alignment"] == expected_alignment
    assert info == {"cache_status": "not_applicable"}
    assert aligned_target is kwargs["target_gs"]
    expected_means = base["means"]
    if mode == "random":
        expected_means = expected_means[[1, 0]]
    torch.testing.assert_close(aligned_input["means"], expected_means)


def test_prepare_alignment_fit_lr_to_hr_uses_precomputed_pair(monkeypatch):
    kwargs = prepare_kwargs()
    monkeypatch.setattr(
        alignment.matching,
        "save_matching_artifacts",
        lambda *args, **arguments: {},
    )

    source, target, info = alignment.prepare_alignment(
        alignment="fit_lr_to_hr", **kwargs
    )

    torch.testing.assert_close(
        source["means"],
        kwargs["input_resolution_entry"]["gs_params"]["means"],
    )
    torch.testing.assert_close(
        target["means"],
        kwargs["scene"]["fit_lr_to_hr"]["gs_params"]["means"],
    )
    assert info == {
        "cache_status": "dataset_precomputed",
        "cache_path": "/fit/checkpoint.pt",
        "matching_steps": 0,
        "matching_images_per_step": 0,
    }


def test_prepare_alignment_fit_hr_to_lr_uses_reverse_fit(monkeypatch):
    kwargs = prepare_kwargs()
    high_res_in_low_frame = {
        key: value + 2.0
        for key, value in kwargs["target_gs"].items()
    }
    fitted = {
        key: value + 1.0
        for key, value in high_res_in_low_frame.items()
    }
    matching_calls = []

    monkeypatch.setattr(
        alignment.matching,
        "build_matching_source",
        lambda *args: high_res_in_low_frame,
    )

    def get_or_fit_matching_target(**arguments):
        matching_calls.append(arguments)
        return fitted, {
            "status": "hit",
            "checkpoint_path": "/cache/matching_target.pt",
        }

    monkeypatch.setattr(
        alignment.matching,
        "get_or_fit_matching_target",
        get_or_fit_matching_target,
    )
    monkeypatch.setattr(
        alignment.matching,
        "save_matching_artifacts",
        lambda *args, **arguments: {},
    )

    source, target, info = alignment.prepare_alignment(
        alignment="fit_hr_to_lr", **kwargs
    )

    torch.testing.assert_close(source["means"], fitted["means"])
    assert target is kwargs["target_gs"]
    assert matching_calls[0]["input_resolution"] == 512
    assert matching_calls[0]["target_resolution"] == 128
    assert info == {
        "cache_status": "hit",
        "cache_path": "/cache/matching_target.pt",
        "matching_steps": 10,
        "matching_images_per_step": 1,
    }


def test_prepare_alignment_can_skip_densification_artifacts(monkeypatch):
    kwargs = prepare_kwargs()
    base = {
        key: value.clone()
        for key, value in kwargs["input_resolution_entry"]["gs_params"].items()
    }
    calls = []

    def build_densified_input(**arguments):
        calls.append(arguments)
        return {key: value.clone() for key, value in base.items()}

    monkeypatch.setattr(
        alignment.densification,
        "build_densified_input",
        build_densified_input,
    )
    monkeypatch.setattr(
        alignment.densification,
        "save_densification_stages",
        lambda **arguments: pytest.fail("artifacts should be disabled"),
    )
    monkeypatch.setattr(
        alignment,
        "write_densify_stage_render_metrics",
        lambda **arguments: pytest.fail("metrics should be disabled"),
    )

    source, target, info = alignment.prepare_alignment(
        alignment="emd", write_artifacts=False, **kwargs
    )

    assert calls[0]["return_stages"] is False
    assert target is kwargs["target_gs"]
    assert info == {"cache_status": "not_applicable"}
    torch.testing.assert_close(source["means"], base["means"])


def test_prepare_alignment_uses_supplied_reverse_fit_views(monkeypatch):
    kwargs = prepare_kwargs()
    supplied_images = [torch.ones((2, 2, 3))]
    supplied_cameras = {"camera_to_worlds": torch.ones((1, 3, 4))}
    high_res_in_low_frame = {
        key: value + 2.0 for key, value in kwargs["target_gs"].items()
    }
    fitted = {
        key: value + 1.0 for key, value in high_res_in_low_frame.items()
    }
    matching_calls = []

    monkeypatch.setattr(
        kwargs["dataset"],
        "load_resolution_views",
        lambda entry: pytest.fail("dataset views should not be reloaded"),
    )
    monkeypatch.setattr(
        alignment.matching,
        "build_matching_source",
        lambda *args: high_res_in_low_frame,
    )

    def get_or_fit_matching_target(**arguments):
        matching_calls.append(arguments)
        return fitted, {
            "status": "hit",
            "checkpoint_path": "/cache/matching_target.pt",
        }

    monkeypatch.setattr(
        alignment.matching,
        "get_or_fit_matching_target",
        get_or_fit_matching_target,
    )

    _, target, info = alignment.prepare_alignment(
        alignment="fit_hr_to_lr",
        input_images=supplied_images,
        input_cameras=supplied_cameras,
        write_artifacts=False,
        **kwargs,
    )

    assert matching_calls[0]["target_images"] is supplied_images
    assert matching_calls[0]["target_cameras"] is supplied_cameras
    assert target is kwargs["target_gs"]
    assert info["cache_status"] == "hit"

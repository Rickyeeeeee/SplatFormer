import pytest
import torch

from utils.sr_dataset_utils import build_precomputed_fit_pair, find_scene_index


class _IdentityScaler:
    scale_ = torch.tensor(1.0)
    trans_ = torch.zeros(3)


def _entry(count):
    return {
        "gs_params": {
            "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
            "scales": torch.zeros((count, 3)),
        },
        "scaler": _IdentityScaler(),
    }


def test_build_precomputed_fit_pair_uses_dataset_target_and_provenance():
    source_entry = _entry(3)
    target_entry = _entry(4)
    fitted = {
        key: value + 1.0 for key, value in source_entry["gs_params"].items()
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

    torch.testing.assert_close(source["means"], source_entry["gs_params"]["means"])
    torch.testing.assert_close(target["means"], fitted["means"])
    assert provenance == {
        "status": "dataset_precomputed",
        "checkpoint_path": "/fit/ckpt_2999_rank0.pt",
    }


def test_build_precomputed_fit_pair_validates_resolution_and_identity_shape():
    source_entry = _entry(3)
    target_entry = _entry(4)
    scene = {
        "fit_lr_to_hr": {
            "source_resolution": 128,
            "target_resolution": 512,
            "gs_params": _entry(2)["gs_params"],
            "checkpoint_path": "/fit/checkpoint.pt",
        }
    }
    with pytest.raises(ValueError, match="supports 128->512"):
        build_precomputed_fit_pair(
            scene, source_entry, target_entry, 256, 512, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="identity-paired"):
        build_precomputed_fit_pair(
            scene, source_entry, target_entry, 128, 512, torch.device("cpu")
        )


def test_find_scene_index_rejects_filtered_or_unknown_scene():
    dataset = type(
        "Dataset",
        (),
        {"folders": [{"scene_name": "available"}]},
    )()
    assert find_scene_index(dataset, "available") == 0
    with pytest.raises(ValueError, match="absent or was filtered"):
        find_scene_index(dataset, "filtered")

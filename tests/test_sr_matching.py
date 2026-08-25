import os

import torch

from sr.matching import MatchingCache


def source_gaussians(count=2):
    return {
        "means": torch.arange(
            count * 3, dtype=torch.float32
        ).reshape(count, 3),
        "opacities": torch.zeros((count, 1), dtype=torch.float32),
    }


def test_matching_cache_round_trip_and_schema_validation(tmp_path):
    source = source_gaussians()
    config = {"total_steps": 10, "image_per_step": 2}
    cache = MatchingCache(
        root=str(tmp_path),
        scene_name="nested/scene_a",
        input_resolution=128,
        target_resolution=512,
        source_gs=source,
        matching_config=config,
    )
    target = {key: value + 1.0 for key, value in source.items()}

    with cache.lock():
        cache.save(target)

    loaded, reason = cache.load(source)
    assert reason == "compatible checkpoint"
    assert os.path.isfile(cache.checkpoint_path)
    assert cache.cache_dir.endswith("scene_a/ir128_tr512")
    for key in source:
        torch.testing.assert_close(loaded[key], target[key])
        assert loaded[key] is not target[key]

    payload = torch.load(cache.checkpoint_path, map_location="cpu")
    payload["metadata"]["target_resolution"] = 256
    invalid, reason = cache.validate(payload, source)
    assert invalid is None
    assert reason == "metadata mismatch for target_resolution"


def test_matching_cache_rejects_attribute_shape_changes(tmp_path):
    source = source_gaussians()
    cache = MatchingCache(
        root=str(tmp_path),
        scene_name="scene_a",
        input_resolution=128,
        target_resolution=512,
        source_gs=source,
        matching_config={},
    )
    payload = {
        "metadata": cache.metadata,
        "target_gs": {
            "means": torch.zeros((3, 3)),
            "opacities": torch.zeros((2, 1)),
        },
    }

    target, reason = cache.validate(payload, source)

    assert target is None
    assert reason == "cached target means shape or dtype mismatch"

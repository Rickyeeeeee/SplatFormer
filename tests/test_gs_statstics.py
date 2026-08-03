import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "gs_statstics.py"
SPEC = importlib.util.spec_from_file_location("gs_statstics", MODULE_PATH)
gs_statstics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gs_statstics)


def _gs(offset: float, count: int = 2):
    means = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) + offset
    return {
        "means": means,
        "scales": means + 1,
        "opacities": means[:, :1] + 2,
        "quats": torch.cat([means, means[:, :1]], dim=1) + 3,
        "features_dc": means + 4,
        "features_rest": torch.arange(count * 2 * 3, dtype=torch.float32).reshape(count, 2, 3) + offset,
    }


def test_streaming_stats_match_direct_channelwise_reduction():
    first = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    second = torch.arange(12, 30, dtype=torch.float32).reshape(3, 2, 3)
    values = torch.cat([first, second], dim=0).double()

    accumulator = gs_statstics.ChannelwiseStreamingStats()
    accumulator.update(first)
    accumulator.update(second)
    result = accumulator.as_dict()

    assert result["channel_shape"] == [2, 3]
    assert result["gaussian_count"] == len(values)
    torch.testing.assert_close(torch.tensor(result["mean"], dtype=torch.float64), values.mean(dim=0))
    torch.testing.assert_close(torch.tensor(result["min"], dtype=torch.float64), values.min(dim=0).values)
    torch.testing.assert_close(torch.tensor(result["max"], dtype=torch.float64), values.max(dim=0).values)
    torch.testing.assert_close(torch.tensor(result["variance"], dtype=torch.float64), values.var(dim=0, unbiased=False))
    torch.testing.assert_close(torch.tensor(result["std"], dtype=torch.float64), values.std(dim=0, unbiased=False))


class _FakeDataset:
    factors = [1, 2]
    skipped_scenes = []
    folders = [{"scene_name": "good"}, {"scene_name": "broken"}]

    def load_scene(self, scene_idx):
        if scene_idx == 1:
            raise RuntimeError("checkpoint is corrupt")
        return {
            "factor_data": {
                1: {"gs_params": _gs(0)},
                2: {"gs_params": _gs(100)},
            }
        }


def test_scene_failure_does_not_partially_contribute_to_any_factor():
    report = gs_statstics.analyze_dataset(_FakeDataset(), show_progress=False)

    assert report["candidate_scene_count"] == 2
    assert report["successful_scene_count"] == 1
    assert report["skipped_scene_count"] == 1
    assert report["skipped_scenes"][0]["scene_name"] == "broken"
    assert report["factors"]["df-1"]["scene_count"] == 1
    assert report["factors"]["df-2"]["scene_count"] == 1
    assert report["factors"]["df-1"]["gaussian_count"] == 2
    assert report["factors"]["df-2"]["gaussian_count"] == 2
    assert report["factors"]["df-1"]["parameters"]["means"]["mean"] == [1.5, 2.5, 3.5]

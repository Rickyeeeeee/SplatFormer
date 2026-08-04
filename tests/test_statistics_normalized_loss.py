import json
import tempfile
from pathlib import Path

import torch

from utils.loss_utils import feature_mse_loss, load_gs_statistics_normalizers


def test_component_normalizers_scale_means_and_mse_losses():
    output = {
        "means": torch.tensor([[3.0, 5.0, 12.0]], requires_grad=True),
        "features_rest": torch.tensor([[[3.0, 4.0], [6.0, 8.0]]], requires_grad=True),
    }
    target = {
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "features_rest": torch.zeros((1, 2, 2)),
    }
    normalizers = {
        "means": torch.tensor([2.0, 3.0, 9.0]),
        "features_rest": torch.tensor([[3.0, 4.0], [6.0, 8.0]]),
    }

    total, losses, weighted = feature_mse_loss(
        output,
        target,
        ["means", "features_rest"],
        loss_weights={"means": 1.0, "features_rest": 1.0},
        component_normalizers=normalizers,
    )

    torch.testing.assert_close(losses["means"], torch.tensor(1.0))
    torch.testing.assert_close(losses["features_rest"], torch.tensor(1.0))
    torch.testing.assert_close(weighted["means"], torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(2.0))
    total.backward()


def test_statistics_normalizers_select_factor_and_clamp_small_std():
    report = {
        "factors": {
            "df-2": {
                "parameters": {
                    "means": {"channel_shape": [3], "std": [0.0, 1e-8, 2.0]},
                }
            }
        }
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_path = Path(temp_dir) / "gs_statistics.json"
        stats_path.write_text(json.dumps(report))
        target = {"means": torch.zeros((4, 3))}
        normalizers, selected_statistics = load_gs_statistics_normalizers(stats_path, 2, ["means"], target)

    torch.testing.assert_close(normalizers["means"], torch.tensor([1e-6, 1e-6, 2.0]))
    assert selected_statistics["means"]["std"] == [0.0, 1e-8, 2.0]


def test_component_normalizers_reject_post_activated_loss():
    output = {"means": torch.ones((1, 3), requires_grad=True)}
    target = {"means": torch.zeros((1, 3))}
    try:
        feature_mse_loss(
            output,
            target,
            ["means"],
            post_activate_loss=True,
            component_normalizers={"means": torch.ones(3)},
        )
    except ValueError as error:
        assert "does not support post_activate_loss" in str(error)
        return
    raise AssertionError("Expected statistics-normalized post-activation loss to fail")

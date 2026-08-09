import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def _load_module():
    from utils import sr_matching_utils

    return sr_matching_utils


MODULE = _load_module()


def _gs(count=2):
    return {
        "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        "features_dc": torch.zeros((count, 3)),
        "features_rest": torch.zeros((count, 0, 3)),
        "opacities": torch.zeros((count, 1)),
        "scales": torch.zeros((count, 3)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
    }


def _scaler(scale, trans):
    return SimpleNamespace(
        scale_=torch.tensor(scale, dtype=torch.float32),
        trans_=torch.tensor(trans, dtype=torch.float32),
    )


def test_matching_source_preserves_count_order_and_isolated_tensors():
    input_gs = _gs(3)
    source = MODULE.build_matching_source(
        {"gs_params": input_gs, "scaler": _scaler([2, 2, 2], [1, 1, 1])},
        {"scaler": _scaler([4, 4, 4], [3, 3, 3])},
        torch.device("cpu"),
    )

    assert source["means"].shape == input_gs["means"].shape
    expected_means = ((input_gs["means"] - 1) / 2) * 4 + 3
    torch.testing.assert_close(source["means"], expected_means)
    assert source["features_dc"].data_ptr() != input_gs["features_dc"].data_ptr()

    trainable = MODULE.make_trainable_gs(source)
    assert set(trainable) == set(source)
    assert all(isinstance(value, torch.nn.Parameter) for value in trainable.values())
    with torch.no_grad():
        trainable["means"].add_(1)
    target = MODULE.detach_matching_target(trainable, source)
    assert target["means"].shape[0] == source["means"].shape[0]
    assert not torch.equal(target["means"], source["means"])


def test_mocked_matching_step_keeps_identity_correspondence(tmp_path):
    source = _gs(2)
    target_images = [torch.zeros((1, 1, 3), dtype=torch.float32) for _ in range(2)]
    target_cameras = {"camera_to_worlds": torch.eye(4).unsqueeze(0).repeat(2, 1, 1)}
    config = {
        "total_steps": 1,
        "image_per_step": 1,
        "log_interval": 1,
        "preview_interval": 1,
        "grad_clip_norm": 0.0,
        "image_l1_loss_weight": 1.0,
        "lpips_loss_weight": 0.0,
        "enable_amp": False,
        "empty_cache_fre": -1,
    }

    def fake_render(params, _cameras):
        assert _cameras["camera_to_worlds"].shape[0] == 1
        image = params["means"][:1].reshape(1, 1, 3).sigmoid()
        return [image], [torch.ones((1, 1, 1), dtype=image.dtype)]

    with mock.patch.object(MODULE.gs_utils, "rasterize_gaussians_to_multiimgs", side_effect=fake_render), mock.patch.object(
        MODULE, "build_3DGSoptimizer", side_effect=lambda params: torch.optim.SGD(params.parameters(), lr=0.1)
    ), mock.patch.object(MODULE, "build_scheduler", side_effect=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)):
        fitted = MODULE.fit_matching_target(
            source,
            target_images,
            target_cameras,
            str(tmp_path),
            logging.getLogger("test-noemd"),
            config,
        )

    assert set(fitted) == set(source)
    assert fitted["means"].shape == source["means"].shape
    assert fitted["means"].data_ptr() != source["means"].data_ptr()
    assert (tmp_path / "matching_train" / "loss.csv").is_file()
    assert (tmp_path / "matching_train" / "00000000_pred.png").is_file()

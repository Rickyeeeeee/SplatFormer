import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "overfit-sr-ar.py"
SPEC = importlib.util.spec_from_file_location("overfit_sr_ar", MODULE_PATH)
AR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AR)


def test_default_resolution_schedule():
    schedule = AR.build_resolution_schedule((128, 128), (256, 256), 10)
    assert [entry["t"] for entry in schedule] == [index / 10 for index in range(1, 11)]
    assert [entry["size"][0] for entry in schedule] == [
        141,
        154,
        166,
        179,
        192,
        205,
        218,
        230,
        243,
        256,
    ]


def test_resize_scales_images_and_camera_intrinsics():
    images = [torch.ones(256, 256, 3)]
    cameras = {
        "camera_to_worlds": torch.eye(4).unsqueeze(0)[:, :3],
        "fx": torch.tensor(200.0),
        "fy": torch.tensor(220.0),
        "cx": torch.tensor(128.0),
        "cy": torch.tensor(120.0),
        "width": torch.tensor(256.0),
        "height": torch.tensor(256.0),
        "background_color": torch.zeros(3),
    }
    resized_images, resized_cameras = AR.resize_images_and_cameras(
        images, cameras, (128, 192)
    )
    assert resized_images[0].shape == (128, 192, 3)
    assert torch.allclose(resized_cameras["fx"], torch.tensor(150.0))
    assert torch.allclose(resized_cameras["fy"], torch.tensor(110.0))
    assert torch.allclose(resized_cameras["cx"], torch.tensor(96.0))
    assert torch.allclose(resized_cameras["cy"], torch.tensor(60.0))
    assert resized_cameras["width"].item() == 192
    assert resized_cameras["height"].item() == 128


def test_apply_delta_preserves_unpredicted_keys():
    gs = {
        "means": torch.zeros(2, 3),
        "opacities": torch.ones(2, 1),
    }
    updated = AR.apply_gs_delta(gs, {"means": torch.full((2, 3), 0.25)})
    assert torch.allclose(updated["means"], torch.full((2, 3), 0.25))
    assert updated["opacities"] is gs["opacities"]


def test_prefix_rollout_is_detached_and_active_pass_gets_gradient():
    class MockDeltaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.timesteps = []

        def forward(self, batch_normalized_gs, timestep, **kwargs):
            del kwargs
            self.timesteps.append(float(timestep.item()))
            gs = batch_normalized_gs[0]
            return [{"means": torch.ones_like(gs["means"]) * self.weight * timestep}]

    model = MockDeltaModel()
    input_means = torch.zeros(2, 3, requires_grad=True)
    input_gs = {"means": input_means, "opacities": torch.ones(2, 1)}
    current_gs = AR.rollout_detached_prefix(model, input_gs, stage_index=2, num_stages=10)
    assert not current_gs["means"].requires_grad
    assert torch.allclose(torch.tensor(model.timesteps), torch.tensor([0.1, 0.2]))

    active_t = torch.tensor([0.3])
    active_delta = model([current_gs], timestep=active_t)[0]
    predicted_gs = AR.apply_gs_delta(current_gs, active_delta)
    predicted_gs["means"].sum().backward()
    assert model.weight.grad is not None
    assert input_means.grad is None


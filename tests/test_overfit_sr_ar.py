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


def test_training_schedule_nests_updates_stages_and_rollouts():
    expected = {
        0: (0, 0, 0),
        499: (0, 0, 499),
        500: (0, 1, 0),
        999: (0, 1, 499),
        1000: (1, 0, 0),
        1999: (1, 1, 499),
    }
    positions = {
        step: AR.training_position(step, num_stages=2, steps_per_stage=500)
        for step in expected
    }
    assert positions == expected


def test_training_schedule_config_and_validation():
    kwargs = {
        "total_steps": 17,
        "pretrain_steps": 0,
        "eval_interval": 1,
        "log_interval": 1,
        "save_interval": 1,
        "log_image_interval": 1,
        "grad_clip_norm": 0.0,
    }
    config = AR.training(
        **kwargs,
        ar_num_rollouts=2,
        ar_num_stages=2,
        ar_steps_per_stage=500,
    )
    assert config["total_steps"] == 2000
    assert config["legacy_total_steps"] == 17
    assert config["ema_enabled"]
    assert config["ema_decay"] == 0.999
    assert config["ema_warmup_steps"] == 0
    assert config["ema_resume_ckpt"] is None

    for invalid_field in ("ar_num_rollouts", "ar_num_stages", "ar_steps_per_stage"):
        invalid_schedule = {
            "ar_num_rollouts": 2,
            "ar_num_stages": 2,
            "ar_steps_per_stage": 500,
            invalid_field: 0,
        }
        try:
            AR.training(**kwargs, **invalid_schedule)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {invalid_field}=0 to be rejected")

    invalid_ema_configs = [
        {"ema_decay": -0.1},
        {"ema_decay": 1.0},
        {"ema_warmup_steps": -1},
        {"ema_enabled": False, "ema_resume_ckpt": "ema.pth"},
    ]
    for invalid_config in invalid_ema_configs:
        try:
            AR.training(**kwargs, **invalid_config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected EMA config to be rejected: {invalid_config}")


def test_ema_decay_fixed_and_warmup_modes():
    assert AR.effective_ema_decay(0.999, warmup_steps=0, num_updates=0) == 0.999
    assert AR.effective_ema_decay(0.999, warmup_steps=0, num_updates=2000) == 0.999
    assert AR.effective_ema_decay(0.999, warmup_steps=100, num_updates=0) == 0.0
    assert torch.isclose(
        torch.tensor(AR.effective_ema_decay(0.999, 100, 1)),
        torch.tensor(0.00999),
    )
    assert AR.effective_ema_decay(0.999, warmup_steps=100, num_updates=100) == 0.999
    assert AR.effective_ema_decay(0.999, warmup_steps=100, num_updates=200) == 0.999


def test_checkpoint_paths_keep_online_and_ema_separate():
    periodic = AR.checkpoint_paths("/tmp/checkpoints", completed_step=200)
    assert periodic["online"].endswith("/model_00000200.pth")
    assert periodic["ema"].endswith("/model_ema_00000200.pth")
    final = AR.checkpoint_paths("/tmp/checkpoints")
    assert final["online"].endswith("/model_last.pth")
    assert final["ema"].endswith("/model_ema_last.pth")


def test_model_ema_updates_parameters_and_buffers_only_on_applied_steps():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("float_buffer", torch.tensor(2.0))
            self.register_buffer("int_buffer", torch.tensor(3, dtype=torch.int64))

    model = ToyModel()
    model_ema = AR.ModelEMA(model, decay=0.5)
    assert not model_ema.module.training
    assert all(not parameter.requires_grad for parameter in model_ema.module.parameters())
    assert torch.equal(model_ema.module.weight, model.weight)
    assert torch.equal(model_ema.module.float_buffer, model.float_buffer)
    assert torch.equal(model_ema.module.int_buffer, model.int_buffer)

    with torch.no_grad():
        model.weight.fill_(3.0)
        model.float_buffer.fill_(4.0)
        model.int_buffer.fill_(7)

    assert not model_ema.update(model, update_applied=False)
    assert model_ema.num_updates == 0
    assert torch.equal(model_ema.module.weight, torch.tensor(1.0))
    assert torch.equal(model_ema.module.float_buffer, torch.tensor(2.0))
    assert torch.equal(model_ema.module.int_buffer, torch.tensor(3))

    assert model_ema.update(model, update_applied=True)
    assert model_ema.num_updates == 1
    assert torch.equal(model_ema.module.weight, torch.tensor(2.0))
    assert torch.equal(model_ema.module.float_buffer, torch.tensor(3.0))
    assert torch.equal(model_ema.module.int_buffer, torch.tensor(7))
    assert not model_ema.module.training


def test_preserve_model_mode_restores_mode_after_success_and_failure():
    model = torch.nn.Linear(1, 1)

    @AR.preserve_model_mode
    def switch_to_eval(inner_model, fail=False):
        inner_model.eval()
        if fail:
            raise RuntimeError("expected")

    model.train()
    switch_to_eval(model)
    assert model.training
    try:
        switch_to_eval(model, fail=True)
    except RuntimeError:
        pass
    assert model.training


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
            self.training_modes = []
            self.grad_modes = []
            self.timestep_dtypes = []

        def forward(self, batch_normalized_gs, timestep, **kwargs):
            del kwargs
            self.timesteps.append(float(timestep.item()))
            self.training_modes.append(self.training)
            self.grad_modes.append(torch.is_grad_enabled())
            self.timestep_dtypes.append(timestep.dtype)
            gs = batch_normalized_gs[0]
            return [{"means": torch.ones_like(gs["means"]) * self.weight * timestep}]

    model = MockDeltaModel()
    model_ema = AR.ModelEMA(model)
    prefix_model = model_ema.module
    with torch.no_grad():
        model.weight.fill_(2.0)
    model.train()
    input_means = torch.zeros(2, 3, requires_grad=True)
    input_gs = {"means": input_means, "opacities": torch.ones(2, 1)}
    current_gs = AR.rollout_detached_prefix(
        prefix_model, input_gs, stage_index=2, num_stages=10
    )
    assert not current_gs["means"].requires_grad
    assert torch.allclose(current_gs["means"], torch.full((2, 3), 0.3))
    assert torch.allclose(torch.tensor(prefix_model.timesteps), torch.tensor([0.1, 0.2]))
    assert prefix_model.training_modes == [False, False]
    assert prefix_model.grad_modes == [False, False]
    assert prefix_model.timestep_dtypes == [torch.float32, torch.float32]
    assert not prefix_model.training

    active_t = torch.tensor([0.3])
    active_delta = model([current_gs], timestep=active_t)[0]
    predicted_gs = AR.apply_gs_delta(current_gs, active_delta)
    predicted_gs["means"].sum().backward()
    assert model.weight.grad is not None
    assert prefix_model.weight.grad is None
    assert input_means.grad is None
    assert model.training_modes == [True]
    assert model.grad_modes == [True]

"""Focused checks for checkpoint sampling and both viewer color modes."""

import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sr import noise_to_gs
from utils.gaussian_viewer import _render_gsplat_for_camera
from utils.gs_normalization import STATISTIC_KEYS


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("visualize_noise_to_gs_checkpoint",
                                              ROOT / "scripts/visualize_noise_to_gs_checkpoint.py")
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


class UnitStandardizer:
    def __init__(self):
        shapes = {"means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
                  "features_dc": (3,), "features_rest": (0, 3)}
        self.means = {key: torch.zeros(shape) for key, shape in shapes.items()}
        self.stds = {key: torch.ones(shape) for key, shape in shapes.items()}

    def decode(self, state):
        return {key: state[key] * self.stds[key] + self.means[key] for key in STATISTIC_KEYS}


class ConstantVelocity(torch.nn.Module):
    backbone_type = "DIT"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
        return [{key: torch.ones_like(value) * self.weight for key, value in state.items()}
                for state in batch_flow_gs]


class CheckpointViewerTests(unittest.TestCase):
    def test_noise_only_uses_model_count_without_loading_dataset(self):
        args = viewer.parse_args(["--checkpoint", "run/checkpoints/model.pth"])
        model = types.SimpleNamespace(backbone_type="DIT", coordinate_mode="current",
                                      max_gaussians=5, sh_degree=0)
        model.to = lambda _: model
        model.load_state_dict = lambda *args, **kwargs: None
        model.eval = lambda: model
        flow_cfg = {"gs_statistics_path": "stats.json", "flow_steps": 3,
                    "flow_t_eps": 1e-4, "eval_noise_seed": 9,
                    "normalization_variance_floor": 1e-8}
        with patch.dict(viewer.PREDICTORS, {"dit": lambda: model}), \
                patch.object(viewer.gin, "parse_config_files_and_bindings"), \
                patch.object(viewer, "flow_matching", return_value=flow_cfg), \
                patch.object(viewer, "GaussianStandardizer"), \
                patch.object(viewer.torch, "load", return_value={}), \
                patch.object(viewer.SplatFactoSRDataset, "from_gin_scope") as load_dataset:
            _, _, count, scene_idx, reference_means, _, seed = viewer.load_experiment(args)
        load_dataset.assert_not_called()
        self.assertEqual((count, scene_idx, reference_means, seed), (5, 0, None, 9))

    def test_recorded_endpoint_matches_training_sampler(self):
        model, standardizer = ConstantVelocity(), UnitStandardizer()
        model.eval()
        states = viewer.record_sample(model, 5, 2, standardizer, 4, 1e-4, 17)
        expected = noise_to_gs.sample_flow_model(model, 5, 2, 4, standardizer, noise_seed=17)
        self.assertEqual(len(states), 5)
        for key in STATISTIC_KEYS:
            torch.testing.assert_close(states[-1][key], expected[key])
        self.assertFalse(model.training)

    def test_sequence_colors_and_bar_follow_slot_indices(self):
        colors = viewer.sequence_index_colors(3)
        bar, labels = viewer.sequence_colorbar(colors)
        self.assertEqual(colors.shape, (3, 3))
        self.assertEqual(colors.dtype, np.uint8)
        self.assertEqual(len(np.unique(colors, axis=0)), 3)
        np.testing.assert_array_equal(colors, viewer.sequence_index_colors(3))
        np.testing.assert_array_equal(bar[0, 0], colors[0])
        np.testing.assert_array_equal(bar[0, -1], colors[-1])
        self.assertIn("**0**", labels)
        self.assertIn("**2**", labels)
        self.assertEqual(viewer.sequence_colorbar(viewer.sequence_index_colors(9))[1],
                         "Index (left → right): **0** · **2** · **4** · **6** · **8**")
        single_bar, single_labels = viewer.sequence_colorbar(viewer.sequence_index_colors(1))
        np.testing.assert_array_equal(single_bar[0, -1],
                                      viewer.sequence_index_colors(1)[0])
        self.assertIn("**0**", single_labels)
        self.assertEqual(viewer.parse_args(["--checkpoint", "model.pth", "--initial_color_mode", "morton"]).initial_color_mode,
                         "morton")

    def test_viewer_keeps_sequence_colors_across_steps(self):
        initial = torch.tensor([[1., 1., 1.], [0., 0., 0.]])
        final = initial.flip(0)
        states = [{"means": means, "features_dc": torch.zeros((2, 3)),
                   "features_rest": torch.zeros((2, 0, 3))} for means in (initial, final)]
        instance = object.__new__(viewer.CheckpointViewer)
        instance.lock = threading.RLock()
        instance.states = states
        instance.count = 2
        instance.sequence_colors = viewer.sequence_index_colors(2)
        instance.step = types.SimpleNamespace(value=0)
        instance.color_mode = types.SimpleNamespace(value="Sequence index")
        instance.colorbar = types.SimpleNamespace(visible=False)
        instance.colorbar_labels = types.SimpleNamespace(visible=False, content="Index **0** (left) → **1** (right)")
        instance.point_size = types.SimpleNamespace(value=0.01)
        instance.state = {"view_mode": "pointcloud"}
        instance.point_handle = None
        instance.status = types.SimpleNamespace(content="")
        instance.args = types.SimpleNamespace(scene_name="scene", config="config.gin")
        instance.active_checkpoint = Path("model.pth")
        instance.active_seed = 0
        instance.flow_cfg = {"flow_steps": 1}
        instance.message = "Ready"
        point_colors, render_colors = [], []
        scene = types.SimpleNamespace(add_point_cloud=lambda *args, **kwargs: (
            point_colors.append(kwargs["colors"].copy()), types.SimpleNamespace(remove=lambda: None))[1])
        instance.server = types.SimpleNamespace(scene=scene)
        instance.renderer = types.SimpleNamespace(update_scene=lambda _, colors: render_colors.append(colors),
                                                  render_all=lambda: None)
        instance.refresh(None)
        instance.step.value = 1
        instance.refresh(None)
        np.testing.assert_array_equal(point_colors[0], point_colors[1])
        np.testing.assert_array_equal(render_colors[0], render_colors[1])
        self.assertTrue(instance.colorbar.visible)
        self.assertIn("**1**", instance.colorbar_labels.content)
        instance.color_mode.value = "GS color"
        instance.refresh(None)
        np.testing.assert_array_equal(point_colors[-1], np.full((2, 3), 128, dtype=np.uint8))
        self.assertIsNone(render_colors[-1])
        self.assertFalse(instance.colorbar.visible)

    def test_step_buttons_move_one_step_and_clamp(self):
        instance = object.__new__(viewer.CheckpointViewer)
        instance.step = types.SimpleNamespace(value=0)
        instance.flow_cfg = {"flow_steps": 2}
        instance.decrease_step(None)
        self.assertEqual(instance.step.value, 0)
        instance.increase_step(None)
        self.assertEqual(instance.step.value, 1)
        instance.increase_step(None)
        instance.increase_step(None)
        self.assertEqual(instance.step.value, 2)
        instance.decrease_step(None)
        self.assertEqual(instance.step.value, 1)

    def test_resampling_keeps_slot_colors_with_new_positions(self):
        instance = object.__new__(viewer.CheckpointViewer)
        instance.lock = threading.RLock()
        instance.sampling = False
        instance.seed = types.SimpleNamespace(value=8)
        instance.resample_button = types.SimpleNamespace(disabled=False)
        instance.reload_button = types.SimpleNamespace(disabled=False)
        instance.active_checkpoint = Path("valid.pth")
        instance.model = ConstantVelocity()
        instance.standardizer = UnitStandardizer()
        instance.count, instance.scene_idx = 2, 0
        instance.reference_means = None
        instance.flow_cfg = {"flow_steps": 1, "flow_t_eps": 1e-4}
        instance.step = types.SimpleNamespace(value=1)
        instance.states = [{"means": torch.tensor([[0., 0., 0.], [1., 1., 1.]])}]
        instance.sequence_colors = viewer.sequence_index_colors(2)
        original_colors = instance.sequence_colors
        instance.refresh = lambda _: None
        new_states = [{"means": torch.tensor([[1., 1., 1.], [0., 0., 0.]])}]
        with patch.object(viewer.threading, "Thread",
                          side_effect=lambda target, daemon: types.SimpleNamespace(start=target)), \
                patch.object(viewer, "record_sample", return_value=new_states):
            instance.start_sampling(reload_checkpoint=False)
        self.assertIs(instance.sequence_colors, original_colors)
        self.assertIs(instance.states, new_states)
        self.assertEqual(instance.step.value, 0)

    def test_gs_color_override_preserves_geometry_and_opacity(self):
        captured = {}

        def rasterization(**kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2, 1)), None

        rendering = types.ModuleType("gsplat.rendering")
        rendering.rasterization = rasterization
        gsplat = types.ModuleType("gsplat")
        gsplat.rendering = rendering
        camera = types.SimpleNamespace(wxyz=np.array([1., 0., 0., 0.]), position=np.zeros(3),
                                       aspect=1., fov=1.)
        means = torch.tensor([[0., 0., 1.], [1., 0., 1.]])
        quats = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        scales = torch.ones((2, 3))
        opacities = torch.tensor([0.2, 0.8])
        gs_colors = torch.full((2, 3), 0.4)
        gaussians = {"means": means}
        cfg = {"device": "cpu", "height": 2, "near_plane": 0.01, "far_plane": 10.,
               "radius_clip": 0., "eps2d": 0.3, "rasterize_mode": "classic",
               "background": np.zeros(3)}
        sequence_colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        with patch.dict(sys.modules, {"gsplat": gsplat, "gsplat.rendering": rendering}), \
                patch("utils.gs_utils._prepare_render_inputs",
                      return_value=(means, quats, scales, opacities, gs_colors, None)):
            _render_gsplat_for_camera(camera, gaussians, cfg)
            torch.testing.assert_close(captured["colors"], gs_colors)
            _render_gsplat_for_camera(camera, gaussians, cfg, sequence_colors)
        torch.testing.assert_close(captured["colors"], torch.tensor(sequence_colors, dtype=torch.float32) / 255)
        torch.testing.assert_close(captured["means"], means)
        torch.testing.assert_close(captured["quats"], quats)
        torch.testing.assert_close(captured["scales"], scales)
        torch.testing.assert_close(captured["opacities"], opacities)
        self.assertIsNone(captured["sh_degree"])

    def test_failed_checkpoint_reload_keeps_active_sample(self):
        instance = object.__new__(viewer.CheckpointViewer)
        instance.lock = threading.RLock()
        instance.sampling = False
        instance.seed = types.SimpleNamespace(value=7)
        instance.checkpoint_input = types.SimpleNamespace(value="missing.pth")
        instance.resample_button = types.SimpleNamespace(disabled=False)
        instance.reload_button = types.SimpleNamespace(disabled=False)
        instance.active_checkpoint = Path("valid.pth")
        instance.model = ConstantVelocity()
        instance.states = [object()]
        instance.args = types.SimpleNamespace(device="cpu")
        instance.refresh = lambda _: None
        with patch.object(viewer.threading, "Thread",
                          side_effect=lambda target, daemon: types.SimpleNamespace(start=target)), \
                patch.object(viewer.torch, "load", side_effect=FileNotFoundError("missing.pth")):
            instance.start_sampling(reload_checkpoint=True)
        self.assertEqual(instance.active_checkpoint, Path("valid.pth"))
        self.assertEqual(len(instance.states), 1)
        self.assertTrue(instance.message.startswith("Checkpoint load failed:"))
        self.assertFalse(instance.sampling)
        self.assertFalse(instance.reload_button.disabled)


if __name__ == "__main__":
    unittest.main()

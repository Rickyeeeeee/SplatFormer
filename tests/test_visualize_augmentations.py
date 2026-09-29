"""CPU checks against real augmentation, interpolant, and attention implementations."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gin
import numpy as np
import torch

from scripts import visualize_augmentations as script
from sr import edipt_interpolants, interpolants
from utils import augmentation_viewer as viewer
from utils.data_augmentation import (GAUSSIAN_PARAMETERS, jitter_gaussian_parameter,
                                     jitter_gaussian_parameters, rotate_gaussians,
                                     sample_uniform_rotation_quaternion, sample_uniform_z_rotation_quaternion)
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS
from utils.rotation_flow import rotation_exp


ROOT = Path(__file__).resolve().parents[1]
SHAPES = {"means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
          "features_dc": (3,), "features_rest": (3, 3)}


class AugmentationViewerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(41)
        self.source = {key: torch.randn((19,) + shape, generator=generator) for key, shape in SHAPES.items()}
        self.source["means"] = self.source["means"] * .2 + .5
        self.target = {key: torch.randn(value.shape, generator=generator) for key, value in self.source.items()}
        self.target["means"] = self.target["means"] * .2 + .5
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "statistics.json"
        statistics = {stored: {"mean": torch.full(SHAPES[key], .2).tolist(),
                               "variance": torch.full(SHAPES[key], .49).tolist()}
                      for key, stored in STATISTIC_KEYS.items()}
        self.path.write_text(json.dumps({"aggregate": {"normalized": {"output": statistics}}}))
        self.settings = {"grid_resolution": 1536, "shift_negative_grid_coords": True,
                         "depth": 2, "patch_sizes": (4, 8), "order": ("z", "z-trans"),
                         "shuffle_orders": True, "enable_flash": True}
        self.augmentation = {"random_rotate": True, "random_jitter": True, "rotation_mode": "full",
                             "rotation_pivot": (.5, .5, .5), "rotation_max_degrees": 45.,
                             "jitter_max_levels": dict.fromkeys(GAUSSIAN_PARAMETERS, .1),
                             "serialization_reference": "augmented"}

    def tearDown(self):
        self.directory.cleanup()
        gin.clear_config()

    def comparison(self, backend="dipt", mode="linear"):
        representation = "unit_unstandardized" if backend == "edipt" else "raw_standardized"
        norm = GaussianStandardizer(self.path, quaternion_representation=representation)
        return viewer.AugmentationComparison(self.source, self.target, norm, backend, self.settings, 23, 17, mode)

    def assert_gaussians_close(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])

    def test_sampled_combinations_match_training_order_and_keep_inputs(self):
        originals = {key: value.clone() for key, value in self.source.items()}
        for backend in ("dipt", "edipt"):
            comparison = self.comparison(backend)
            for mode in ("full", "gravity_consistent"):
                for rotate, jitter in ((False, False), (True, False), (False, True), (True, True)):
                    settings = {**self.augmentation, "rotation_mode": mode, "random_rotate": rotate, "random_jitter": jitter}
                    rng_state = torch.random.get_rng_state().clone()
                    rotation, levels = comparison.augment(settings, 53)
                    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
                    generator = torch.Generator().manual_seed(53)
                    expected_source, expected_target = comparison.source, comparison.target
                    if rotate:
                        sampler = sample_uniform_z_rotation_quaternion if mode == "gravity_consistent" else sample_uniform_rotation_quaternion
                        expected_rotation = sampler(generator=generator, max_degrees=45.)
                        torch.testing.assert_close(rotation, expected_rotation)
                        expected_source = rotate_gaussians(expected_source, expected_rotation, (.5, .5, .5))
                        expected_target = rotate_gaussians(expected_target, expected_rotation, (.5, .5, .5))
                    if jitter:
                        expected_source, expected_levels = jitter_gaussian_parameters(expected_source, settings["jitter_max_levels"], generator)
                        self.assertEqual(levels, expected_levels)
                    if backend == "edipt" and (rotate or jitter):
                        expected_source, expected_target = comparison.standardizer.prepare_endpoints(expected_source, expected_target, align_target_sign=False)
                    self.assert_gaussians_close(comparison.current_source, expected_source)
                    self.assert_gaussians_close(comparison.current_target, expected_target)
                    first = {key: value.clone() for key, value in comparison.current_source.items()}
                    comparison.augment(settings, 53)
                    self.assert_gaussians_close(comparison.current_source, first)
        self.assert_gaussians_close(self.source, originals)

    def test_manual_levels_rotation_and_zero_axis(self):
        comparison = self.comparison("edipt")
        for mode in ("full", "gravity_consistent"):
            settings = {**self.augmentation, "control_mode": "Manual", "rotation_axis": (1., 0., 0.),
                        "rotation_degrees": 30., "rotation_mode": mode}
            q = rotation_exp(torch.tensor([1., 0., 0.] if mode == "full" else [0., 0., 1.]) * np.pi / 6)
            expected = rotate_gaussians(comparison.source, q, (.5, .5, .5))
            generator = torch.Generator().manual_seed(19)
            for key in GAUSSIAN_PARAMETERS:
                expected = jitter_gaussian_parameter(expected, key, .1, generator)
            comparison.augment(settings, 19)
            self.assert_gaussians_close(comparison.current_source, expected)
        with self.assertRaises(ValueError):
            comparison.augment({**settings, "rotation_mode": "full", "rotation_axis": (0., 0., 0.)}, 19)

    def test_states_match_both_training_interpolants_and_exact_endpoints(self):
        for backend in ("dipt", "edipt"):
            comparison = self.comparison(backend)
            comparison.augment(self.augmentation, 53)
            module = edipt_interpolants if backend == "edipt" else interpolants
            for mode in interpolants.MODES:
                kwargs = {"rotation_noise_std": .4} if backend == "edipt" else {}
                for t in (.25, .5, .75):
                    actual = comparison.states("Interpolant", mode, t, .7, .4)
                    for state, flows, endpoints in zip(actual, (comparison.base_flow, comparison.current_flow),
                                                       ((comparison.source, comparison.target), (comparison.current_source, comparison.current_target))):
                        expected = module.construct_path(None if mode == "one_sided" else flows[0], flows[1], t, mode,
                                                          endpoints[0]["means"], endpoints[1]["means"], .7,
                                                          noise=comparison.noise, **kwargs)["query"]
                        self.assert_gaussians_close(state, comparison.standardizer.decode(expected))
                start = comparison.states("Interpolant", mode, 0., .7, .4)[0]
                end = comparison.states("Interpolant", mode, 1., .7, .4)[0]
                expected_start = comparison.source
                if mode == "one_sided":
                    noise = comparison.noise if backend == "dipt" else {
                        **comparison.noise["euclidean"], "quats": rotation_exp(.4 * comparison.noise["rotation"])}
                    expected_start = comparison.standardizer.decode(noise)
                self.assert_gaussians_close(start, expected_start)
                self.assert_gaussians_close(end, comparison.target)

    def test_mode_changes_prepare_quaternion_signs_from_raw_endpoints(self):
        self.target["quats"] = -2 * self.source["quats"]
        comparison = self.comparison("edipt")
        torch.testing.assert_close(comparison.source["quats"], comparison.target["quats"])
        comparison.set_mode("one_sided")
        torch.testing.assert_close(comparison.source["quats"], -comparison.target["quats"])
        comparison.set_mode("linear")
        torch.testing.assert_close(comparison.source["quats"], comparison.target["quats"])

    def test_serialization_matches_pointcept_with_negative_and_duplicate_cells(self):
        from pointcept.models.utils.structure import Point

        means = torch.tensor([[-.2, .6, 0.], [.1, .2, .3], [.1, .2, .3], [.3, -.1, .8]])
        for shifted in (False, True):
            settings = {**self.settings, "shift_negative_grid_coords": shifted}
            grid = torch.floor(means * 1536).int()
            if shifted:
                grid -= grid.amin(0).clamp(max=0)
            point = Point({"grid_coord": grid, "offset": torch.tensor([4])})
            point.serialization(order=("z", "z-trans"), shuffle_orders=False)
            torch.testing.assert_close(viewer.serialize_reference(means, settings), point.serialized_order)

    def test_dipt_patch_members_match_actual_padding_flash_and_nonflash(self):
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedAttention
        from pointcept.models.utils.structure import Point

        for count in (1, 3, 4, 5, 9, 10):
            for size in (1, 2, 4, 8):
                order = torch.arange(count - 1, -1, -1)
                for flash in (False, True):
                    actual = viewer.patch_layout(order, size, "dipt", flash)
                    point = Point({"offset": torch.tensor([count])})
                    pad, unpad, offsets = SerializedAttention.get_padding_and_inverse(SimpleNamespace(patch_size=actual["effective_size"]), point)
                    torch.testing.assert_close(actual["members"], order[pad])
                    torch.testing.assert_close(actual["offsets"], offsets.long())
                    expected_ids = unpad[order.argsort()] // actual["effective_size"]
                    torch.testing.assert_close(actual["ids"], expected_ids)

    def test_edipt_patch_members_and_masks_match_attention(self):
        from models.equivariant_gaussian_dipt import GeometricAttention

        for count in (1, 3, 4, 5, 9):
            for size in (1, 2, 4, 8):
                order = torch.arange(count - 1, -1, -1)
                layout = viewer.patch_layout(order, size, "edipt")
                attention = SimpleNamespace(attend=mock.Mock(side_effect=lambda features, *args: features))
                features = torch.arange(count).float().reshape(-1, 1)
                result = GeometricAttention.forward(attention, features, features, features, order, order.argsort(), [count], size)
                called_features, _, _, valid = attention.attend.call_args.args
                torch.testing.assert_close(called_features.flatten().long(), layout["members"])
                torch.testing.assert_close(valid.flatten(), layout["valid"])
                torch.testing.assert_close(result, features)

    def test_reference_switch_and_slider_regrouping_reuse_serialized_orders(self):
        comparison = self.comparison("edipt")
        comparison.augment(self.augmentation, 53)
        with mock.patch.object(viewer, "serialize_reference", wraps=viewer.serialize_reference) as serialize:
            comparison.layouts("Interpolant", "encoding_decoding", .49, "augmented", 0, "Block default", 4)
            self.assertEqual(serialize.call_count, 2)
            comparison.layouts("Interpolant", "encoding_decoding", .49, "augmented", 1, "z-trans", 7)
            self.assertEqual(serialize.call_count, 2)
            comparison.layouts("Interpolant", "encoding_decoding", .5, "augmented", 0, "Block default", 4)
            self.assertEqual(serialize.call_count, 4)
            fixed = comparison.layouts("Interpolant", "encoding_decoding", .5, "unaugmented", 0, "Block default", 4)
            self.assertEqual(serialize.call_count, 4)
            torch.testing.assert_close(fixed[0]["ids"], fixed[1]["ids"])
            comparison.layouts("Interpolant", "one_sided", 0., "augmented", 0, "z", 8)
            self.assertEqual(serialize.call_count, 4)

    def test_display_offsets_colors_and_subsampling_do_not_change_comparison(self):
        comparison = self.comparison()
        comparison.augment(self.augmentation, 53)
        originals = [{key: value.clone() for key, value in gs.items()} for gs in (comparison.source, comparison.current_source)]
        layouts = comparison.layouts("Source", "linear", 0., "augmented", 0, "z", 4)
        for colors in ("Gaussian RGB", "Patches"):
            displayed, combined = viewer.compose_comparison(*originals, layouts, colors, (.5, .5, .5))
            self.assertEqual(len(combined["means"]), 38)
            for original, gs, layout in zip(originals, displayed, layouts):
                differences = gs["means"] - original["means"]
                torch.testing.assert_close(differences, differences[:1].expand_as(differences))
                if colors == "Patches":
                    rgb = (gs["features_dc"].sigmoid() * 255).round().byte().numpy()
                    np.testing.assert_array_equal(rgb, viewer.patch_colors(layout["ids"]))
            self.assert_gaussians_close(comparison.source, originals[0])
            self.assert_gaussians_close(comparison.current_source, originals[1])
        np.testing.assert_array_equal(viewer.patch_colors(torch.tensor([0, 1])), viewer.patch_colors(torch.tensor([0, 1, 2]))[:2])

    def test_real_gin_model_configuration_and_loader_overrides(self):
        from dataset.GS_SR import SplatFactoSRDataset

        dataset = SimpleNamespace(src_resolution=32, tgt_resolution=128, scene_index=lambda name: 3,
                                  load_scene=lambda *args, **kwargs: {"scene_name": "fixture", "data": {32: {"gs_params": self.source}},
                                                                   "fit_lr_to_hr": {"tgt_gs": self.target}})
        for backend in ("dipt", "edipt"):
            args = script.parse_args(["--backend", backend, "--config", str(ROOT / "configs/visualize/interpolants.gin"),
                                      "--scene_name", "fixture", "--gin_param", f"flow_matching.gs_statistics_path='{self.path}'",
                                      "--gin_param", "training_augmentation.rotation_max_degrees=12",
                                      "--gin_param", "training_augmentation.random_rotate=True"])
            with mock.patch.object(SplatFactoSRDataset, "from_gin_scope", return_value=dataset):
                comparison, metadata, flow_cfg, augmentation_cfg = script.load_scene(args)
            self.assertEqual(comparison.settings["grid_resolution"], 1536)
            self.assertEqual(comparison.settings["patch_sizes"][:4], (256,) * 4 if backend == "edipt" else (256, 512, 1024, 1024))
            self.assertEqual(comparison.settings["shift_negative_grid_coords"], backend == "edipt")
            self.assertEqual(metadata["noise_seed"], 3)
            self.assertEqual(augmentation_cfg["rotation_max_degrees"], 12)
            self.assertTrue(augmentation_cfg["random_rotate"])

    def test_camera_framing_keeps_the_display_center_as_look_at(self):
        from viser import CameraHandle

        client = SimpleNamespace(_websock_connection=mock.Mock())
        client.camera = CameraHandle(client)
        client.camera._state.position = np.array([3., 3., 3.])
        client.camera._state.look_at = np.zeros(3)
        client.camera._state.up_direction = np.array([0., 0., 1.])
        client.camera._state.update_timestamp = 1.
        bounds = (torch.tensor([-1., 0., 0.]), torch.tensor([2., 1., 1.]))
        script.AugmentationViewer.frame_client(SimpleNamespace(display_bounds=bounds), client)
        np.testing.assert_allclose(client.camera.look_at, (.5, .5, .5))
        np.testing.assert_allclose(client.camera.up_direction, (0., 0., 1.))

    def test_slider_callbacks_preserve_augmentation_and_restore_block_size(self):
        # Exercise real GUI handles, without a browser or GPU renderer.
        comparison = self.comparison()
        args = script.parse_args(["--config", "unused.gin", "--scene_name", "fixture", "--host", "127.0.0.1", "--port", "0"])
        metadata = {"scene_name": "fixture", "source_resolution": 32, "target_resolution": 128, "split": "test", "noise_seed": 23}
        flow_cfg = {"interpolant_type": "linear", "flow_noise_std": 1., "rotation_noise_std": .3}
        ui = script.AugmentationViewer(comparison, metadata, flow_cfg, self.augmentation, args)
        try:
            self.assertEqual((ui.patch_size.min, ui.patch_size.max, ui.patch_size.step), (1, 4096, 1))
            with ui.lock, mock.patch.object(comparison, "augment", wraps=comparison.augment) as augment:
                original = {key: value.clone() for key, value in comparison.current_source.items()}
                ui.patch_size.value = 7
                ui.on_display(None)
                self.assertEqual(augment.call_count, 0)
                self.assert_gaussians_close(comparison.current_source, original)
                ui.block.value = "2"
                ui.on_block(None)
                self.assertEqual(ui.patch_size.value, 8)
                self.assertEqual(augment.call_count, 0)
                ui.mode.value = "one_sided"
                ui.on_mode(None)
                self.assertTrue(ui.jitter.disabled)
                self.assertFalse(ui.augmentation_settings()["random_jitter"])
        finally:
            ui.server.stop()


if __name__ == "__main__":
    unittest.main()

"""Focused checks for the analytic noise viewer and portable presets."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from sr import noise_to_gs
from utils.gs_normalization import STATISTIC_KEYS


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("visualize_noise_to_gs", ROOT / "scripts/visualize_noise_to_gs.py")
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


class UnitStandardizer:
    def __init__(self):
        shapes = {"means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
                  "features_dc": (3,), "features_rest": (3, 3)}
        self.means = {key: torch.full(shape, 0.5) for key, shape in shapes.items()}
        self.stds = {key: torch.full(shape, 2.0) for key, shape in shapes.items()}

    def decode(self, state):
        return {key: state[key] * self.stds[key] + self.means[key] for key in STATISTIC_KEYS}


class NoiseViewerTests(unittest.TestCase):
    def test_default_path_matches_training_seed_and_endpoints(self):
        standardizer = UnitStandardizer()
        target_flow = {key: torch.full((7,) + tuple(mean.shape), 0.25) for key, mean in standardizer.means.items()}
        target = standardizer.decode(target_flow)
        scene_idx, seed = 3, 11
        generator = torch.Generator(device="cpu").manual_seed(seed + scene_idx)
        noise = noise_to_gs.sample_noise(7, standardizer, "cpu", generator)
        settings = viewer.default_settings()
        start = viewer.analytic_state(target, target_flow, noise, standardizer, 0.0, settings)
        middle = viewer.analytic_state(target, target_flow, noise, standardizer, 0.5, settings)
        end = viewer.analytic_state(target, target_flow, noise, standardizer, 1.0, settings)
        for key in STATISTIC_KEYS:
            torch.testing.assert_close(start[key], standardizer.decode(noise)[key])
            torch.testing.assert_close(middle[key], (start[key] + target[key]) / 2)
            torch.testing.assert_close(end[key], target[key])
            self.assertEqual(len(middle[key]), 7)
        generator = torch.Generator(device="cpu").manual_seed(seed + scene_idx)
        repeated = noise_to_gs.sample_noise(7, standardizer, "cpu", generator)
        for key in STATISTIC_KEYS:
            torch.testing.assert_close(noise[key], repeated[key])

    def test_adjustment_clips_before_strength_and_preserves_target(self):
        standardizer = UnitStandardizer()
        target_flow = {key: torch.zeros((2,) + tuple(mean.shape)) for key, mean in standardizer.means.items()}
        target = standardizer.decode(target_flow)
        noise = {key: torch.full_like(value, 3.0) for key, value in target_flow.items()}
        settings = viewer.default_settings()
        settings["means"] = {"strength": 2.0, "clip": 1.0}
        start = viewer.analytic_state(target, target_flow, noise, standardizer, 0.0, settings)
        torch.testing.assert_close(start["means"], torch.full_like(start["means"], 4.5))
        torch.testing.assert_close(start["scales"], torch.full_like(start["scales"], 6.5))
        torch.testing.assert_close(viewer.analytic_state(target, target_flow, noise, standardizer, 1.0, settings)["means"], target["means"])
        self.assertEqual(noise["means"][0, 0].item(), 3.0)

    def test_preset_round_trip_and_validation(self):
        settings = viewer.default_settings()
        settings["scales"] = {"strength": 0.7, "clip": 2.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "presets" / "noise.json"
            viewer.save_preset(path, settings)
            self.assertEqual(viewer.load_preset(path), settings)
        settings["means"]["clip"] = -1.0
        with self.assertRaisesRegex(ValueError, "means.clip"):
            viewer.validate_settings(settings)


if __name__ == "__main__":
    unittest.main()

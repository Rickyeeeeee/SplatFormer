"""CPU tests for Gaussian parameter augmentation utilities."""

import math
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

from utils.data_augmentation import (
    GAUSSIAN_PARAMETERS,
    _quaternion_to_rotation_matrix,
    jitter_gaussian_parameter,
    jitter_gaussian_parameters,
    quaternion_inverse,
    quaternion_multiply,
    rotate_camera_to_worlds,
    rotate_gaussians,
    sample_uniform_rotation_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]


class DataAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.gs = {
            "means": torch.tensor(((0.2, 0.3, 0.4), (0.8, 0.5, 0.7), (0.4, 0.9, 0.1)), dtype=torch.float64),
            "scales": torch.tensor(((-2.0, -1.0, -0.5), (-1.5, -0.7, -0.1), (-1.0, -0.2, 0.3)), dtype=torch.float64),
            "opacities": torch.tensor(((-1.0,), (0.0,), (1.0,)), dtype=torch.float64),
            "quats": F.normalize(torch.tensor(((1.0, 0.2, 0.1, 0.0), (0.8, -0.1, 0.3, 0.1),
                                                (0.9, 0.0, -0.2, 0.2)), dtype=torch.float64), dim=-1),
            "features_dc": torch.tensor(((0.1, 0.2, 0.3), (0.3, 0.1, -0.2), (-0.1, 0.0, 0.4)), dtype=torch.float64),
            "features_rest": torch.arange(27, dtype=torch.float64).reshape(3, 3, 3) / 20,
        }

    def test_jitter_is_reproducible_relative_and_non_mutating(self):
        original = {key: value.clone() for key, value in self.gs.items()}
        first = jitter_gaussian_parameter(self.gs, "means", 0.25, torch.Generator().manual_seed(7))
        second = jitter_gaussian_parameter(self.gs, "means", 0.25, torch.Generator().manual_seed(7))
        expected_noise = torch.randn(self.gs["means"].shape, dtype=torch.float64,
                                     generator=torch.Generator().manual_seed(7))
        expected = self.gs["means"] + 0.25 * self.gs["means"].std(dim=0, correction=0) * expected_noise
        torch.testing.assert_close(first["means"], second["means"])
        torch.testing.assert_close(first["means"], expected)
        for key in GAUSSIAN_PARAMETERS:
            torch.testing.assert_close(self.gs[key], original[key])
            if key != "means":
                torch.testing.assert_close(first[key], self.gs[key])

    def test_zero_jitter_is_exact_and_quaternion_jitter_is_normalized(self):
        zero = jitter_gaussian_parameter(self.gs, "quats", 0.0, torch.Generator().manual_seed(3))
        for key in self.gs:
            self.assertIsNot(zero[key], self.gs[key])
            self.assertTrue(torch.equal(zero[key], self.gs[key]))
        jittered = jitter_gaussian_parameter(self.gs, "quats", 0.5, torch.Generator().manual_seed(3))
        torch.testing.assert_close(torch.linalg.vector_norm(jittered["quats"], dim=-1), torch.ones(3, dtype=torch.float64))

    def test_multi_parameter_jitter_samples_configured_levels(self):
        maxima = {"means": 0.2, "opacities": 0.1}
        first, first_levels = jitter_gaussian_parameters(self.gs, maxima, torch.Generator().manual_seed(5))
        second, second_levels = jitter_gaussian_parameters(self.gs, maxima, torch.Generator().manual_seed(5))
        self.assertEqual(first_levels, second_levels)
        self.assertTrue(0 < first_levels["means"] < maxima["means"])
        self.assertTrue(0 < first_levels["opacities"] < maxima["opacities"])
        for key in GAUSSIAN_PARAMETERS:
            torch.testing.assert_close(first[key], second[key])
            if key not in maxima:
                self.assertEqual(first_levels[key], 0.0)
                torch.testing.assert_close(first[key], self.gs[key])
        self.assertFalse(torch.equal(first["means"], self.gs["means"]))
        self.assertFalse(torch.equal(first["opacities"], self.gs["opacities"]))
        with self.assertRaises(ValueError):
            jitter_gaussian_parameters(self.gs, {"unknown": 0.1})

    def test_quaternion_composition_inverse_and_known_axis_rotation(self):
        half_angle = math.pi / 4
        rotation = torch.tensor((math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)), dtype=torch.float64)
        identity = quaternion_multiply(quaternion_inverse(rotation), rotation)
        torch.testing.assert_close(identity, torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float64), atol=1e-12, rtol=0)

        simple = {key: value[:1].clone() for key, value in self.gs.items()}
        simple["means"] = torch.tensor(((1.0, 0.0, 0.0),), dtype=torch.float64)
        simple["quats"] = torch.tensor(((1.0, 0.0, 0.0, 0.0),), dtype=torch.float64)
        rotated = rotate_gaussians(simple, rotation, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(rotated["means"], torch.tensor(((0.0, 1.0, 0.0),), dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(rotated["quats"][0], rotation, atol=1e-12, rtol=0)

    def test_rotation_about_pivot_and_covariance_transform(self):
        half_angle = math.pi / 4
        rotation = torch.tensor((math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)), dtype=torch.float64)
        pivot = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float64)
        rotated = rotate_gaussians(self.gs, rotation, pivot)
        expected_first = torch.tensor((0.7, 0.2, 0.4), dtype=torch.float64)
        torch.testing.assert_close(rotated["means"][0], expected_first, atol=1e-12, rtol=0)

        old_rotation = _quaternion_to_rotation_matrix(self.gs["quats"])
        new_rotation = _quaternion_to_rotation_matrix(rotated["quats"])
        linear_scales = self.gs["scales"].exp()
        old_covariance = old_rotation @ torch.diag_embed(linear_scales.square()) @ old_rotation.transpose(-1, -2)
        new_covariance = new_rotation @ torch.diag_embed(linear_scales.square()) @ new_rotation.transpose(-1, -2)
        global_rotation = _quaternion_to_rotation_matrix(rotation)
        expected_covariance = global_rotation @ old_covariance @ global_rotation.transpose(0, 1)
        torch.testing.assert_close(new_covariance, expected_covariance, atol=1e-12, rtol=1e-12)

    def test_degree_one_sh_follows_rotated_directions(self):
        rotation = F.normalize(torch.tensor((0.7, -0.2, 0.4, 0.5), dtype=torch.float64), dim=-1)
        rotated = rotate_gaussians(self.gs, rotation, torch.zeros(3, dtype=torch.float64))
        rotation_matrix = _quaternion_to_rotation_matrix(rotation)
        directions = F.normalize(torch.tensor(((0.2, 0.7, -0.1), (-0.4, 0.3, 0.8),
                                                (0.6, -0.2, 0.5)), dtype=torch.float64), dim=-1)
        rotated_directions = directions @ rotation_matrix.transpose(0, 1)
        basis = lambda value: torch.stack((-value[:, 1], value[:, 2], -value[:, 0]), dim=-1)
        original_colors = torch.einsum("ni,nic->nc", basis(directions), self.gs["features_rest"])
        rotated_colors = torch.einsum("ni,nic->nc", basis(rotated_directions), rotated["features_rest"])
        torch.testing.assert_close(rotated_colors, original_colors, atol=1e-12, rtol=1e-12)

    def test_camera_rotation_and_inverse_roundtrip(self):
        half_angle = math.pi / 4
        rotation = torch.tensor((math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)), dtype=torch.float64)
        pivot = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float64)
        cameras = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        cameras[0, :3, 3] = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
        cameras[1, :3, 3] = torch.tensor((0.5, 0.5, 2.0), dtype=torch.float64)
        rotated = rotate_camera_to_worlds(cameras, rotation, pivot)
        expected_rotation = _quaternion_to_rotation_matrix(rotation)
        torch.testing.assert_close(rotated[:, :3, :3], expected_rotation.expand(2, -1, -1), atol=1e-12, rtol=0)
        torch.testing.assert_close(rotated[0, :3, 3], torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64),
                                   atol=1e-12, rtol=0)
        restored = rotate_camera_to_worlds(rotated, quaternion_inverse(rotation), pivot)
        torch.testing.assert_close(restored, cameras, atol=2e-12, rtol=2e-12)

    def test_rotation_inverse_roundtrip(self):
        rotation = sample_uniform_rotation_quaternion(torch.float64, "cpu", torch.Generator().manual_seed(11))
        pivot = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float64)
        rotated = rotate_gaussians(self.gs, rotation, pivot)
        restored = rotate_gaussians(rotated, quaternion_inverse(rotation), pivot)
        for key in self.gs:
            expected = self.gs[key]
            actual = restored[key]
            if key == "quats":
                expected = F.normalize(expected, dim=-1)
                actual = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0, -actual, actual)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_uniform_rotation_sampler_is_seeded_and_unit_length(self):
        first = sample_uniform_rotation_quaternion(torch.float64, "cpu", torch.Generator().manual_seed(19))
        second = sample_uniform_rotation_quaternion(torch.float64, "cpu", torch.Generator().manual_seed(19))
        self.assertEqual(first.shape, (4,))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(torch.linalg.vector_norm(first), torch.tensor(1.0, dtype=torch.float64))

    def test_invalid_parameter_and_sh_degree_are_rejected(self):
        with self.assertRaises(ValueError):
            jitter_gaussian_parameter(self.gs, "unknown", 0.1)
        invalid = {key: value.clone() for key, value in self.gs.items()}
        invalid["features_rest"] = torch.zeros(3, 8, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "degree 1"):
            rotate_gaussians(invalid, torch.tensor((1.0, 0.0, 0.0, 0.0)), torch.zeros(3))

    def test_runner_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "sr.data_augmentation", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--jitter_levels", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""Degree-zero SH rendering checks without requiring gsplat CUDA kernels."""

import ast
import math
from pathlib import Path
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


class DegreeZeroRenderTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "utils/gs_utils.py"
        tree = ast.parse(path.read_text())
        selected = {"_prepare_render_inputs", "_rasterize_gaussians",
                    "rasterize_gaussians_to_multiimgs", "rasterize_gaussians_to_singleimg"}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.rasterization = mock.Mock(return_value=(torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2, 1)), {}))
        namespace = {"torch": torch, "math": math, "BLOCK_WIDTH": 16, "rasterization": self.rasterization,
                     "_camera_to_viewmats": lambda cameras, device, dtype: torch.eye(4).unsqueeze(0),
                     "_to_python_scalar": lambda value: value}
        exec(compile(tree, str(path), "exec"), namespace)
        self.prepare = namespace["_prepare_render_inputs"]
        self.render = namespace["_rasterize_gaussians"]
        self.render_multi = namespace["rasterize_gaussians_to_multiimgs"]
        self.gaussians = {"means": torch.zeros((2, 3)), "scales": torch.zeros((2, 3)),
                          "quats": torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
                          "opacities": torch.zeros((2, 1)), "features_dc": torch.ones((2, 3)),
                          "features_rest": torch.empty((2, 0, 3))}

    def test_explicit_degree_zero_uses_dc_sh_coefficient(self):
        colors, degree = self.prepare(self.gaussians, sh_degree_zero=True)[-2:]
        self.assertEqual(tuple(colors.shape), (2, 1, 3))
        torch.testing.assert_close(colors[:, 0], self.gaussians["features_dc"])
        self.assertEqual(degree, 0)
        self.render(self.gaussians, torch.eye(4), 1, 1, 1, 1, 2, 2, (0, 0, 0), sh_degree_zero=True)
        self.assertEqual(self.rasterization.call_args.kwargs["sh_degree"], 0)
        torch.testing.assert_close(self.rasterization.call_args.kwargs["colors"], colors)

    def test_public_multi_view_routes_degree_zero_in_both_modes(self):
        cameras = {"camera_to_worlds": torch.eye(4).unsqueeze(0), "cx": 1, "cy": 1,
                   "fx": 1, "fy": 1, "width": 2, "height": 2, "background_color": (0, 0, 0)}
        for batched in (False, True):
            with self.subTest(batched=batched):
                self.rasterization.reset_mock()
                images, _ = self.render_multi(self.gaussians, cameras, batched=batched, sh_degree_zero=True)
                self.assertEqual(len(images), 1)
                self.assertEqual(self.rasterization.call_args.kwargs["sh_degree"], 0)

    def test_default_empty_rest_keeps_color_logit_behavior(self):
        colors, degree = self.prepare(self.gaussians)[-2:]
        self.assertIsNone(degree)
        torch.testing.assert_close(colors, torch.sigmoid(self.gaussians["features_dc"]))


if __name__ == "__main__":
    unittest.main()

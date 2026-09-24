"""CPU-only tests for the analytic interpolant viewer."""

import argparse
import ast
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from sr import interpolants


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "scripts/visualize_interpolants.py"
SHAPES = {
    "means": (3,), "features_dc": (3,), "features_rest": (3, 3),
    "opacities": (1,), "scales": (3,), "quats": (4,),
}


def load_definitions(names, namespace):
    tree = ast.parse(VIEWER.read_text())
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(tree, str(VIEWER), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class IdentityStandardizer:
    def encode(self, gaussians):
        return {key: value.clone() for key, value in gaussians.items()}

    def decode(self, gaussians):
        return {key: value.clone() for key, value in gaussians.items()}


class InterpolantViewerTests(unittest.TestCase):
    def setUp(self):
        self.standardizer = IdentityStandardizer()
        self.source = {key: torch.rand((5,) + shape) for key, shape in SHAPES.items()}
        self.target = {key: value + .5 for key, value in self.source.items()}
        self.noise = interpolants.seeded_noise_like(self.target, 17)
        namespace = {"np": np, "torch": torch, "interpolants": interpolants}
        self.viewer = load_definitions({"analytic_state"}, namespace)

    def test_states_match_construct_path(self):
        for mode in interpolants.MODES:
            actual = self.viewer.analytic_state(
                mode, .25, .8, self.source, self.target, self.source,
                self.target, self.noise, self.standardizer,
            )
            expected = interpolants.construct_path(
                None if mode == "one_sided" else self.source,
                self.target, .25, mode,
                None if mode == "one_sided" else self.source["means"],
                self.target["means"], .8, noise=self.noise,
            )["query"]
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])

    def test_exact_endpoints_and_encoding_midpoint(self):
        for mode in interpolants.MODES:
            start = self.viewer.analytic_state(
                mode, 0, .8, self.source, self.target, self.source,
                self.target, self.noise, self.standardizer,
            )
            end = self.viewer.analytic_state(
                mode, 1, .8, self.source, self.target, self.source,
                self.target, self.noise, self.standardizer,
            )
            expected_start = self.noise if mode == "one_sided" else self.source
            for key in self.source:
                torch.testing.assert_close(start[key], expected_start[key])
                torch.testing.assert_close(end[key], self.target[key])
        midpoint = self.viewer.analytic_state(
            "encoding_decoding", .5, .8, self.source, self.target,
            self.source, self.target, self.noise, self.standardizer,
        )
        for key in midpoint:
            torch.testing.assert_close(midpoint[key], (.8 * np.sqrt(.5)) * self.noise[key])

    def test_cli_parses_gin_overrides_and_rejects_invalid_rendering(self):
        namespace = {"argparse": argparse, "Path": Path, "np": np}
        viewer = load_definitions({"parse_args"}, namespace)
        args = viewer.parse_args([
            "--config", "config.gin", "--scene_name", "scene",
            "--gin_param", "flow_matching.flow_noise_std=0.5",
            "--gin_param", "SplatFactoSRDataset.max_gs_num=50000",
        ])
        self.assertEqual(len(args.gin_param), 2)
        self.assertEqual(args.config, [Path("config.gin")])
        self.assertEqual(args.initial_view_mode, "gsplat")
        with self.assertRaises(SystemExit):
            viewer.parse_args(["--config", "config.gin", "--scene_name", "scene", "--near_plane", "2", "--far_plane", "1"])
        with self.assertRaises(SystemExit):
            viewer.parse_args(["--config", "config.gin", "--scene_name", "scene", "--background", "0", "2", "0"])

    def test_renderer_throttles_camera_updates_and_force_renders(self):
        render = mock.Mock(return_value=np.zeros((2, 2, 3), dtype=np.uint8))
        namespace = {
            "threading": threading, "time": time, "traceback": traceback,
            "np": np, "_render_gsplat_for_camera": render,
        }
        viewer = load_definitions({"GsplatBackgroundRenderer"}, namespace)
        camera = SimpleNamespace(_state=SimpleNamespace(update_timestamp=1.0), position=(1, 2, 3), wxyz=(1, 0, 0, 0))
        client = SimpleNamespace(camera=camera, set_background_image=mock.Mock())
        server = SimpleNamespace(get_clients=lambda: {"client": client})
        state = {"view_mode": "gsplat"}
        config = {"interval": 60, "background": np.zeros(3)}
        renderer = viewer.GsplatBackgroundRenderer(server, {"means": torch.zeros(1, 3)}, state, config)

        renderer.render_client(client)
        renderer.render_client(client)
        self.assertEqual(render.call_count, 1)
        renderer.render_all()
        self.assertEqual(render.call_count, 2)
        self.assertEqual(camera.position, (1, 2, 3))
        self.assertEqual(camera.wxyz, (1, 0, 0, 0))

        state["view_mode"] = "pointcloud"
        renderer.clear_all()
        image = client.set_background_image.call_args.args[0]
        self.assertEqual(image.shape, (2, 2, 3))


if __name__ == "__main__":
    unittest.main()

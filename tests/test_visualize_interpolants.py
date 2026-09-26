"""CPU-only tests for the analytic interpolant viewer."""

import argparse
import ast
import json
import tempfile
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
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS, QUATERNION_REPRESENTATIONS


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

    def test_flow_settings_preserve_legacy_default_and_accept_unit_representation(self):
        viewer = load_definitions({"flow_matching"}, {"QUATERNION_REPRESENTATIONS": QUATERNION_REPRESENTATIONS})
        self.assertEqual(viewer.flow_matching()["quaternion_representation"], "raw_standardized")
        settings = viewer.flow_matching(quaternion_representation="unit_unstandardized", interpolant_type="one_sided")
        self.assertEqual(settings["quaternion_representation"], "unit_unstandardized")
        self.assertEqual(settings["interpolant_type"], "one_sided")
        with self.assertRaises(ValueError):
            viewer.flow_matching(quaternion_representation="invalid")

    def test_loader_prepares_endpoints_and_records_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.json"
            attributes = {stored: {"mean": torch.full(SHAPES[key], .3).tolist(),
                                   "variance": torch.full(SHAPES[key], 4.).tolist()}
                          for key, stored in STATISTIC_KEYS.items()}
            path.write_text(json.dumps({"aggregate": {"normalized": {"output": attributes}}}))
            raw_target = {**self.source, "quats": -2 * self.source["quats"]}
            scene = {"scene_name": "test", "scene_idx": 0,
                     "data": {32: {"gs_params": self.source}}, "fit_lr_to_hr": {"tgt_gs": raw_target}}
            dataset = SimpleNamespace(src_resolution=32, tgt_resolution=128, scene_index=lambda name: 0,
                                      load_scene=lambda *args, **kwargs: scene)
            config = {"gs_statistics_path": str(path), "normalization_variance_floor": 1e-8,
                      "eval_noise_seed": 0, "flow_noise_std": 1., "interpolant_type": "linear",
                      "quaternion_representation": "unit_unstandardized"}
            namespace = {"torch": torch, "np": np, "Path": Path, "gin": mock.Mock(), "interpolants": interpolants,
                         "GaussianStandardizer": GaussianStandardizer, "flow_matching": lambda: config,
                         "SplatFactoSRDataset": SimpleNamespace(from_gin_scope=lambda *args, **kwargs: dataset)}
            viewer = load_definitions({"load_scene", "cpu_snapshot", "analytic_state"}, namespace)
            for representation in ("raw_standardized", "unit_unstandardized"):
                for mode in ("linear", "one_sided"):
                    config.update(quaternion_representation=representation, interpolant_type=mode)
                    source, target, source_flow, target_flow, noise, norm, metadata = viewer.load_scene("config.gin", "test")
                    self.assertEqual(metadata["quaternion_representation"], representation)
                    if representation == "unit_unstandardized":
                        expected = torch.nn.functional.normalize(self.source["quats"], dim=-1)
                        torch.testing.assert_close(source_flow["quats"], expected)
                        torch.testing.assert_close(target_flow["quats"], -expected if mode == "one_sided" else expected)
                    else:
                        torch.testing.assert_close(source["quats"], self.source["quats"])
                        torch.testing.assert_close(source_flow["quats"], (self.source["quats"] - .3) / 2)
                    end = viewer.analytic_state(mode, 1., 1., source, target, source_flow, target_flow, noise, norm)
                    torch.testing.assert_close(end["quats"], target["quats"])
            torch.testing.assert_close(raw_target["quats"], -2 * self.source["quats"])

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

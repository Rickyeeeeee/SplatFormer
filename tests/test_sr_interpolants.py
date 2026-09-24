"""CPU regression tests independent of rasterization and PTv3 dependencies."""
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from sr import interpolants
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS
from utils.spatial_coordinates import build_spatial_point_fields

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {"means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
          "features_dc": (3,), "features_rest": (3, 3)}


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in tree.body:
        node.decorator_list = []
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class InterpolantTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "statistics.json"
        attributes = {stored: {"mean": torch.full(SHAPES[key], 0.3).tolist(),
                               "variance": torch.full(SHAPES[key], 4.).tolist()}
                      for key, stored in STATISTIC_KEYS.items()}
        attributes["sh0"] = {"mean": [[0.3] * 3], "variance": [[4.] * 3]}
        attributes["opacities"] = {"mean": 0.3, "variance": 4.}
        self.document = {"aggregate": {"normalized": {"output": attributes}}}
        self.path.write_text(json.dumps(self.document))
        self.normalizer = GaussianStandardizer(self.path)
        self.source = {key: torch.rand((4,) + shape) for key, shape in SHAPES.items()}
        self.flow = load_functions(ROOT / "sr/flow.py", {"sample_stochastic_interpolant", "predict_x1_from_velocity", "apply_feature_update", "loss_mix_weights"},
                                   {"torch": torch, "SUPPORTED_GS_KEYS": list(STATISTIC_KEYS), "FLOW_EPSILON": 1e-6})

    def test_roundtrip_shapes_and_decode_gradients(self):
        encoded = self.normalizer.encode(self.source)
        decoded = self.normalizer.decode(encoded)
        for key in SHAPES:
            torch.testing.assert_close(decoded[key], self.source[key])
        encoded = {key: value.detach().requires_grad_() for key, value in encoded.items()}
        sum(value.sum() for value in self.normalizer.decode(encoded).values()).backward()
        for value in encoded.values():
            torch.testing.assert_close(value.grad, torch.full_like(value, 2.))
        self.assertEqual(self.normalizer.report()["statistics_group"], "aggregate.normalized.output")

    def test_floor_and_invalid_statistics(self):
        for bad in (-1., float("nan"), float("inf")):
            self.document["aggregate"]["normalized"]["output"]["means"]["variance"] = [bad] * 3
            self.path.write_text(json.dumps(self.document))
            with self.assertRaisesRegex(ValueError, "variance"):
                GaussianStandardizer(self.path)
        self.document["aggregate"]["normalized"]["output"]["means"]["variance"] = [0.] * 3
        self.path.write_text(json.dumps(self.document))
        norm = GaussianStandardizer(self.path, 1e-4)
        torch.testing.assert_close(norm.stds["means"], torch.full((3,), .01))
        with self.assertRaises(ValueError):
            GaussianStandardizer(self.path, 0)
        self.path.write_text('{"aggregate": {"delta": {}}}')
        with self.assertRaisesRegex(ValueError, "aggregate.normalized.output"):
            GaussianStandardizer(self.path)
        with self.assertRaises(FileNotFoundError):
            GaussianStandardizer(self.path.parent / "absent.json")

    def test_incompatible_channels(self):
        self.source["features_rest"] = torch.zeros(4, 8, 3)
        with self.assertRaisesRegex(ValueError, "channels"):
            self.normalizer.encode(self.source)

    def test_interpolant_targets_and_endpoint(self):
        source = self.normalizer.encode(self.source)
        target = {key: value + .4 for key, value in source.items()}
        time = torch.tensor([.25])
        for strength in (0., 1.):
            query, noise, gamma, gamma_dot = self.flow.sample_stochastic_interpolant(source, target, time, strength)
            velocity = {key: target[key] - source[key] + gamma_dot * noise[key] for key in source}
            endpoint = self.flow.predict_x1_from_velocity(object(), source, query, velocity, noise, gamma, gamma_dot, time)
            for key in source:
                torch.testing.assert_close(query[key], .75 * source[key] + .25 * target[key] + gamma * noise[key])
                torch.testing.assert_close(endpoint[key], target[key])
            loss, _, _ = interpolants.attribute_mse(velocity, velocity)
            self.assertEqual(loss.item(), 0.)

    def test_attribute_weighting(self):
        target = {key: torch.zeros_like(value) for key, value in self.source.items()}
        prediction = {key: value + 1 for key, value in target.items()}
        loss, parts, _ = interpolants.attribute_mse(prediction, target)
        self.assertEqual(loss.item(), 6.)
        self.assertTrue(all(part.item() == 1 for part in parts.values()))

    def test_seeded_training_noise_is_fixed_without_changing_global_rng(self):
        reference = self.normalizer.encode(self.source)
        rng = torch.get_rng_state().clone()
        first = interpolants.seeded_noise_like(reference, 7)
        second = interpolants.seeded_noise_like(reference, 7)
        third = interpolants.seeded_noise_like(reference, 8)
        for key in reference:
            torch.testing.assert_close(first[key], second[key])
        self.assertFalse(torch.equal(first["means"], third["means"]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_microbatch_reuses_precomputed_training_noise(self):
        namespace = {"torch": torch, "np": np, "flow": self.flow, "interpolants": interpolants,
                     "SUPPORTED_GS_KEYS": list(STATISTIC_KEYS),
                     "gpu_utils": SimpleNamespace(move_to_device=lambda value, device: value)}
        module = load_functions(ROOT / "overfit-sr-interpolants.py", {"compute_microbatch_loss"}, namespace)
        queries = []
        class Model(torch.nn.Module):
            def forward(inner, batch_flow_gs, **kwargs):
                queries.append({key: value.clone() for key, value in batch_flow_gs[0].items()})
                return [{key: torch.zeros_like(value) for key, value in batch_flow_gs[0].items()}]
        target = self.normalizer.encode(self.source)
        fixed_noise = {key: torch.full_like(value, .5) for key, value in target.items()}
        scene = {"target_gs": self.source, "target_flow_gs": target, "fixed_train_noise": fixed_noise,
                 "scene_idx": 0, "render_view_count": 0, "target_images": [], "target_cameras": {}}
        config = {"flow_t_eps": 1e-4, "flow_noise_std": 1., "loss_type": "velocity",
                  "interpolant_type": "one_sided", "loss_rollout_steps": 4, "fixed_train_noise": True}
        with mock.patch.object(torch.Tensor, "uniform_", lambda value, *args: value.fill_(.25)):
            for _ in range(2):
                module.compute_microbatch_loss(Model(), [scene], "cpu", config, {"schedule": "fm-only"},
                                               self.normalizer, 0., 0., None, False)
        for key in target:
            torch.testing.assert_close(queries[0][key], queries[1][key])
            torch.testing.assert_close(queries[0][key], .75 * fixed_noise[key] + .25 * target[key])

    def test_euler_uses_scene_references_and_decodes(self):
        calls = []
        class Model(torch.nn.Module):
            def forward(inner, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
                calls.append((batch_flow_gs[0], batch_reference_means[0], t))
                return [{key: torch.ones_like(value) * .5 for key, value in batch_flow_gs[0].items()}]
        for steps in (1, 4):
            calls.clear()
            result = interpolants.sample_flow_model(Model(), self.source, 0, steps, self.normalizer)
            for key in self.source:
                torch.testing.assert_close(result[key], self.source[key] + 1)
            torch.testing.assert_close(calls[0][0]["means"], self.normalizer.encode(self.source)["means"])
            for _, reference, _ in calls:
                self.assertIs(reference, self.source["means"])

    def test_microbatch_decodes_before_render_and_backpropagates(self):
        rendered = []
        def render(gs, cameras):
            rendered.append(gs)
            return [gs["means"].mean().expand(2, 2, 3)], None
        namespace = {"torch": torch, "np": np, "flow": self.flow, "interpolants": interpolants,
                     "SUPPORTED_GS_KEYS": list(STATISTIC_KEYS),
                     "gpu_utils": SimpleNamespace(move_to_device=lambda value, device: value),
                     "gs_utils": SimpleNamespace(rasterize_gaussians_to_multiimgs=render)}
        module = load_functions(ROOT / "overfit-sr-interpolants.py", {"compute_microbatch_loss"}, namespace)
        references = []
        class Model(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(inner, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
                references.extend(batch_reference_means)
                return [{key: torch.ones_like(value) * inner.weight for key, value in gs.items()} for gs in batch_flow_gs]
        encoded = self.normalizer.encode(self.source)
        scene = {"source_gs": self.source, "target_gs": self.source, "source_flow_gs": encoded, "target_flow_gs": encoded,
                 "scene_idx": 0, "render_view_count": 1, "target_images": [torch.zeros(2, 2, 3)],
                 "target_cameras": {"camera_to_worlds": torch.eye(4)[None]}}
        for kind in ("velocity", "x1"):
            model = Model()
            loss, _ = module.compute_microbatch_loss(model, [scene], "cpu",
                {"flow_t_eps": 1e-4, "flow_noise_std": 0., "loss_type": kind, "interpolant_type": "linear", "loss_rollout_steps": 10}, {"schedule": "linear"},
                self.normalizer, 1., 0., None, False)
            loss.backward()
            self.assertTrue(torch.isfinite(model.weight.grad))
            torch.testing.assert_close(rendered[-1]["means"], self.source["means"] + .4)
            torch.testing.assert_close(references[-1], self.source["means"])

    def test_all_path_derivatives_and_references(self):
        source = {key: value.double() for key, value in self.normalizer.encode(self.source).items()}
        target = {key: value + .7 for key, value in source.items()}
        noise = {key: torch.randn_like(value) for key, value in source.items()}
        lr, hr = self.source["means"], self.source["means"] + .1
        for mode in interpolants.MODES:
            for time in (.2, .7):
                with self.subTest(mode=mode, time=time):
                    path = interpolants.construct_path(source, target, time, mode, lr, hr, .8, noise)
                    before = interpolants.construct_path(source, target, time - 1e-5, mode, lr, hr, .8, noise)
                    after = interpolants.construct_path(source, target, time + 1e-5, mode, lr, hr, .8, noise)
                    for key in source:
                        torch.testing.assert_close(path["velocity"][key], (after["query"][key] - before["query"][key]) / 2e-5)
                    expected = hr if mode == "one_sided" or (mode == "encoding_decoding" and time >= .5) else lr
                    self.assertIs(path["reference_means"], expected)
        midpoint = interpolants.construct_path(source, target, .5, "encoding_decoding", lr, hr, .8, noise)
        for key in source:
            torch.testing.assert_close(midpoint["query"][key], noise[key] * (.8 / 2**.5))
            self.assertEqual(midpoint["velocity"][key].abs().sum().item(), 0.)
        changed = {key: value + 100 for key, value in target.items()}
        first = interpolants.construct_path(source, target, .2, "encoding_decoding", lr, hr, 1., noise)
        second = interpolants.construct_path(source, changed, .2, "encoding_decoding", lr, hr, 1., noise)
        for key in source:
            torch.testing.assert_close(first["query"][key], second["query"][key])
            torch.testing.assert_close(first["velocity"][key], second["velocity"][key])
        for time, expected in ((0., noise), (1., target)):
            path = interpolants.construct_path(None, target, time, "one_sided", None, hr, noise=noise)
            for key in source:
                torch.testing.assert_close(path["query"][key], expected[key])
        for mode in ("linear", "latent", "encoding_decoding"):
            for time, expected in ((1e-10, source), (1 - 1e-10, target)):
                path = interpolants.construct_path(source, target, time, mode, lr, hr, noise=noise)
                for key in source:
                    torch.testing.assert_close(path["query"][key], expected[key], atol=1e-4, rtol=1e-4)

    def test_rollout_switch_seed_and_validation(self):
        calls = []
        class Model(torch.nn.Module):
            def forward(inner, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
                calls.append((batch_reference_means[0], batch_flow_gs[0], t))
                return [{key: torch.zeros_like(value) for key, value in batch_flow_gs[0].items()}]
        model = Model()
        lr, hr = self.source["means"], self.source["means"] + .2
        state = self.normalizer.encode(self.source)
        interpolants.rollout(model, state, 0, 4, "encoding_decoding", lr, hr)
        self.assertEqual(len(calls), 4)
        for index, (ref, current, time) in enumerate(calls):
            self.assertIs(ref, lr if index < 2 else hr)
            torch.testing.assert_close(current["means"], state["means"])
            self.assertGreater(time.item(), 0)
        calls.clear()
        rng = torch.get_rng_state().clone()
        first = interpolants.sample_flow_model(model, None, 3, 2, self.normalizer, "one_sided", hr, noise_seed=4)
        second = interpolants.sample_flow_model(model, None, 3, 2, self.normalizer, "one_sided", hr, noise_seed=4)
        third = interpolants.sample_flow_model(model, None, 3, 2, self.normalizer, "one_sided", hr, noise_seed=5)
        torch.testing.assert_close(first["means"], second["means"])
        self.assertFalse(torch.equal(first["means"], third["means"]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(ref is hr for ref, _, _ in calls))
        for mode, steps in (("bad", 10), ("linear", 0), ("encoding_decoding", 2), ("encoding_decoding", 5)):
            with self.assertRaises(ValueError):
                interpolants.validate_settings(mode, steps)

    def test_rollout_gradient_includes_previous_steps(self):
        class Model(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.weight = torch.nn.Parameter(torch.tensor(.3))
            def forward(inner, batch_flow_gs, **kwargs):
                return [{key: value * inner.weight for key, value in batch_flow_gs[0].items()}]
        model = Model()
        source = {key: torch.ones_like(value) for key, value in self.source.items()}
        result = interpolants.rollout(model, source, 0, 4, "linear", self.source["means"], None)
        result["means"].sum().backward()
        expected = source["means"].numel() * (1 + .3 / 4) ** 3
        self.assertAlmostEqual(model.weight.grad.item(), expected, places=5)
        self.assertTrue(model.training)

    def test_all_modes_and_mixing_schedules_train_with_gradients(self):
        rendered = []
        def render(gs, cameras):
            rendered.append(gs)
            return [gs["means"].mean().expand(2, 2, 3)], None
        namespace = {"torch": torch, "np": np, "flow": self.flow, "interpolants": interpolants,
                     "SUPPORTED_GS_KEYS": list(STATISTIC_KEYS),
                     "gpu_utils": SimpleNamespace(move_to_device=lambda value, device: value),
                     "gs_utils": SimpleNamespace(rasterize_gaussians_to_multiimgs=render)}
        module = load_functions(ROOT / "overfit-sr-interpolants.py", {"compute_microbatch_loss"}, namespace)
        class Model(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(inner, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
                return [{key: torch.ones_like(value) * inner.weight for key, value in gs.items()} for gs in batch_flow_gs]
        encoded = self.normalizer.encode(self.source)
        for mode in interpolants.MODES:
            for schedule in ("fm-only", "linear", "free-range-gs"):
                for loss_type in ("velocity", "x1"):
                    with self.subTest(mode=mode, schedule=schedule, loss_type=loss_type):
                        model = Model()
                        scene = {"target_gs": self.source, "target_flow_gs": encoded, "scene_idx": 0,
                                 "render_view_count": 1, "target_images": [torch.zeros(2, 2, 3)],
                                 "target_cameras": {"camera_to_worlds": torch.eye(4)[None]}}
                        if mode != "one_sided":
                            scene.update(source_gs=self.source, source_flow_gs=encoded)
                        config = {"flow_t_eps": 1e-4, "flow_noise_std": 1., "loss_type": loss_type,
                                  "interpolant_type": mode, "loss_rollout_steps": 4}
                        rendered.clear()
                        with mock.patch.object(torch.Tensor, "uniform_", lambda value, *args: value.fill_(.25)):
                            with mock.patch.object(interpolants, "rollout", wraps=interpolants.rollout) as rollout:
                                loss, values = module.compute_microbatch_loss(model, [scene], "cpu", config,
                                    {"schedule": schedule}, self.normalizer, 1., 0., None, False)
                        loss.backward()
                        self.assertTrue(torch.isfinite(model.weight.grad))
                        self.assertGreater(model.weight.grad.abs().item(), 0.)
                        self.assertEqual(bool(rendered), schedule != "fm-only")
                        self.assertEqual(rollout.call_count, int(schedule != "fm-only" or loss_type == "x1"))
                        if mode != "one_sided" and rendered:
                            torch.testing.assert_close(rendered[0]["means"], self.source["means"] + .4)

    def test_launcher_forwards_normalization_controls(self):
        env = dict(os.environ, GS_STATISTICS_PATH="/tmp/custom stats.json",
                   NORMALIZATION_VARIANCE_FLOOR="1e-6", FLOW_NOISE_STD="0.7", INTERPOLANT_TYPE="encoding_decoding",
                   LOSS_ROLLOUT_STEPS="6", EVAL_NOISE_SEED="42", FLOW_STEPS="10", FIXED_TRAIN_NOISE="True",
                   TRAIN_NOISE_SEED="9", LR_WARMUP_STEPS="500", LR_WARMUP_START_FACTOR="0.05")
        command = 'python() { printf "%s\\n" "$@"; }; launcher="$1"; shift; source "$launcher"'
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-sr-interpolants.sh")],
                                env=env, capture_output=True, text=True, check=True)
        self.assertIn("overfit-sr-interpolants.py", result.stdout)
        self.assertIn("flow_matching.normalization_variance_floor=1e-6", result.stdout)
        self.assertIn("flow_matching.flow_noise_std=0.7", result.stdout)
        self.assertIn("flow_matching.gs_statistics_path='/tmp/custom stats.json'", result.stdout)
        self.assertNotIn("velocity_variance_source", result.stdout)
        self.assertIn("flow_matching.interpolant_type='encoding_decoding'", result.stdout)
        self.assertIn("flow_matching.flow_steps=10", result.stdout)
        self.assertIn("flow_matching.loss_rollout_steps=6", result.stdout)
        self.assertIn("flow_matching.eval_noise_seed=42", result.stdout)
        self.assertIn("flow_matching.fixed_train_noise=True", result.stdout)
        self.assertIn("flow_matching.train_noise_seed=9", result.stdout)
        self.assertIn("train2D/build_scheduler.warmup_step=500", result.stdout)
        self.assertIn("train2D/build_scheduler.warmup_start_factor=0.05", result.stdout)


if __name__ == "__main__":
    unittest.main()

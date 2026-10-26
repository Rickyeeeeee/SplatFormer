"""Training rotation and PLY-only evaluation checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from utils import metrics
from utils.spatial_coordinates import morton_order


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("overfit_noise_to_gs_rotation_test", ROOT / "overfit-noise-to-gs.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class FakeStandardizer:
    encoded = []

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, gaussians):
        self.encoded.append({key: value.clone() for key, value in gaussians.items()})
        return {key: value.clone() for key, value in gaussians.items()}

    def report(self):
        return {}


class FakeModel(nn.Module):
    coordinate_mode = "morton_rank"
    backbone_type = "DIT"
    sh_degree = 0
    output_features = ("means", "scales", "opacities", "quats", "features_dc")

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.references = []

    def forward(self, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
        self.references.append(batch_reference_means)
        return [{key: torch.zeros_like(value) + self.anchor for key, value in state.items()}
                for state in batch_flow_gs]


class FakeDataset:
    max_gs_num = 1000
    load_tgt_gs = True
    load_tgt_images = True
    tgt_resolution = 32
    image_per_scene = 1

    def __init__(self):
        self.loaded_images = []
        self.gaussians = {
            "means": torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.3, 0.4], [0.2, 0.9, 0.7]]),
            "scales": torch.ones(3, 3), "opacities": torch.ones(3, 1),
            "quats": torch.tensor([[1., 0., 0., 0.]]).expand(3, -1).clone(),
            "features_dc": torch.ones(3, 3), "features_rest": torch.empty(3, 0, 3),
        }

    def load_scene(self, scene_idx):
        self.loaded_images.append(self.load_tgt_images)
        target = {"gs_params": self.gaussians}
        if self.load_tgt_images:
            target.update({"images": [torch.ones(2, 2, 4)], "images_name": ["view.png"],
                           "cameras": {"camera_to_worlds": torch.eye(4).unsqueeze(0),
                                       "background_color": torch.ones(3)}})
        return {"scene_name": f"scene_{scene_idx}", "data": {32: target}}


class RotationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset()
        self.original = {key: value.clone() for key, value in self.dataset.gaussians.items()}
        self.flow_cfg = {"gs_statistics_path": "unused", "normalization_variance_floor": 1e-8,
                         "flow_t_eps": 1e-4, "flow_steps": 1, "eval_noise_seed": 0}
        self.flags = SimpleNamespace(predictor="dit", scene_mode="one", eval_max_views=1,
                                     batch_size=2, grad_accum_steps=1, eval_subdir="eval_final")

    def run_training(self, seed, rotate=True, coordinate_mode="morton_rank", scene_indices=(0,), steps=1,
                     eval_interval=1, save_interval=None):
        torch.manual_seed(seed)
        self.dataset.load_tgt_images = True
        self.dataset.loaded_images = []
        self.flags.scene_mode = "many" if len(scene_indices) > 1 else "one"
        FakeStandardizer.encoded = []
        captured = []
        models = []
        original_path = SCRIPT.noise_to_gs.construct_path

        def capture_path(target, time):
            captured.append({key: value.clone() for key, value in target.items()})
            return original_path(target, time)

        def build_model():
            model = FakeModel()
            model.coordinate_mode = coordinate_mode
            models.append(model)
            return model

        def fake_evaluate(model, scene, standardizer, flow_cfg, output_dir, ground_truth=False, ply_only=False):
            return None if ply_only else {"psnr": 1.0}

        augmentation = {"random_rotate": rotate, "rotation_mode": "gravity_consistent",
                        "rotation_max_degrees": None, "rotation_pivot": (0.5, 0.5, 0.5)}
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT, "FLAGS", self.flags), \
             mock.patch.dict(SCRIPT.PREDICTORS, {"dit": build_model}), \
             mock.patch.object(SCRIPT, "GaussianStandardizer", FakeStandardizer), \
             mock.patch.object(SCRIPT, "flow_matching", return_value=self.flow_cfg), \
             mock.patch.object(SCRIPT, "training_augmentation", return_value=augmentation), \
             mock.patch.object(SCRIPT, "evaluate_scene", side_effect=fake_evaluate) as evaluate_mock, \
             mock.patch.object(SCRIPT.noise_to_gs, "construct_path", side_effect=capture_path), \
             mock.patch.object(SCRIPT, "build_optimizer", side_effect=lambda model: torch.optim.SGD(model.parameters(), lr=0.01)), \
             mock.patch.object(SCRIPT, "build_scheduler", return_value=SimpleNamespace(step=lambda: None)), \
             mock.patch.object(SCRIPT.torch, "save") as save_mock:
            SCRIPT.training(self.dataset, scene_indices, output_dir, mock.Mock(), torch.device("cpu"),
                            total_steps=steps, eval_interval=eval_interval,
                            save_interval=save_interval or steps + 1,
                            log_interval=1, enable_amp=False)
            report_path = Path(output_dir) / "eval_final" / "metrics.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            calls = evaluate_mock.call_args_list[:]
            checkpoints = [call.args[1] for call in save_mock.call_args_list]
            loaded_images = self.dataset.loaded_images[:]
        return captured, FakeStandardizer.encoded[:], models[0], report, calls, checkpoints, loaded_images

    def test_rotates_each_draw_before_encoding_and_resorts_morton_slots(self):
        targets, encoded, _, _, _, _, _ = self.run_training(12)
        self.assertEqual(len(targets), 2)
        self.assertFalse(torch.equal(encoded[1]["means"], encoded[2]["means"]))
        for target, physical in zip(targets, encoded[1:]):
            order = morton_order(physical["means"])
            for key in target:
                torch.testing.assert_close(target[key], physical[key][order])
            self.assertFalse(torch.equal(physical["quats"], self.original["quats"]))
        for key in self.original:
            torch.testing.assert_close(self.dataset.gaussians[key], self.original[key])

        repeated, _, _, _, _, _, _ = self.run_training(12)
        different, _, _, _, _, _, _ = self.run_training(13)
        for first, second in zip(targets, repeated):
            torch.testing.assert_close(first["means"], second["means"])
        self.assertFalse(torch.equal(targets[0]["means"], different[0]["means"]))

    def test_gt_mode_uses_rotated_reference_and_disabled_mode_keeps_format(self):
        targets, _, model, report, _, _, loaded_images = self.run_training(4, coordinate_mode="gt")
        self.assertIsNone(report)
        self.assertEqual(loaded_images, [False])
        for reference, target in zip(model.references[0], targets):
            torch.testing.assert_close(reference, target["means"])
        canonical = self.original["means"][morton_order(self.original["means"])]
        targets, _, _, report, _, _, loaded_images = self.run_training(4, rotate=False)
        for target in targets:
            torch.testing.assert_close(target["means"], canonical)
        self.assertEqual(loaded_images, [True])
        self.assertEqual(report["mean"], {"psnr": 1.0})
        self.assertNotIn("fid", report)

    def test_ply_only_uses_current_snapshot_and_checkpoint_cadences(self):
        for rotate, scene_indices in ((False, (0, 1)), (True, (0,))):
            with self.subTest(rotate=rotate, scene_indices=scene_indices):
                _, _, _, report, calls, checkpoints, loaded_images = self.run_training(
                    6, rotate=rotate, scene_indices=scene_indices, steps=3,
                    eval_interval=2, save_interval=2
                )
                self.assertIsNone(report)
                self.assertEqual(loaded_images, [False] * len(scene_indices))
                self.assertEqual(len(calls), 3 * len(scene_indices))
                self.assertTrue(all(call.kwargs["ply_only"] for call in calls))
                self.assertTrue(all(not call.kwargs.get("ground_truth", False) for call in calls))
                self.assertEqual(sum("eval/00000000" in call.args[4] for call in calls), len(scene_indices))
                self.assertEqual(sum("eval/00000001" in call.args[4] for call in calls), len(scene_indices))
                self.assertEqual(sum("eval_final" in call.args[4] for call in calls), len(scene_indices))
                self.assertTrue(checkpoints[0].endswith("model_00000001.pth"))
                self.assertTrue(checkpoints[1].endswith("model_last.pth"))

    def test_ply_only_exports_without_image_metrics_or_rendering(self):
        model = FakeModel()
        scene = {"scene_idx": 0, "target_gs": self.dataset.gaussians}
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT.noise_to_gs, "sample_flow_model", return_value=self.dataset.gaussians), \
             mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs") as render_mock, \
             mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer") as export_mock, \
             mock.patch.object(metrics, "MetricComputer") as metric_mock:
            SCRIPT.evaluate_scene(model, scene, FakeStandardizer(), self.flow_cfg, output_dir, ply_only=True)
            export_mock.assert_called_once()
            self.assertTrue(export_mock.call_args.args[1].endswith("point_cloud/output.ply"))
            render_mock.assert_not_called()
            metric_mock.assert_not_called()
            self.assertFalse((Path(output_dir) / "metrics.json").exists())

    def test_launcher_exposes_rotation_and_names_its_output(self):
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        environment = dict(os.environ, RANDOM_ROTATE="True", ROTATION_MODE="gravity_consistent",
                           ROTATION_MAX_DEGREES="45",
                           DATASET_ROOT="/tmp/noise-to-gs-test")
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("_rot_gravity_consistent_deg45_", result.stdout)
        self.assertIn("--gin_param=training_augmentation.random_rotate=True", result.stdout)
        self.assertIn("--gin_param=SplatFactoSRDataset.load_tgt_images=False", result.stdout)
        environment["RANDOM_ROTATE"] = "False"
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertNotIn("_rot_gravity_consistent_", result.stdout)
        self.assertIn("--gin_param=SplatFactoSRDataset.load_tgt_images=True", result.stdout)
        environment.update(SCENE_MODE="many", SCENE_COUNT="2")
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("--gin_param=SplatFactoSRDataset.load_tgt_images=False", result.stdout)
        environment["SCENE_COUNT"] = "1"
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("--gin_param=SplatFactoSRDataset.load_tgt_images=True", result.stdout)

    def test_ply_snapshot_is_reproducible_for_a_fixed_noise_seed(self):
        model = FakeModel()
        scene = {"scene_idx": 0, "target_gs": self.dataset.gaussians}
        shapes = {key: value.shape[1:] for key, value in self.dataset.gaussians.items()}
        standardizer = SimpleNamespace(means={key: torch.zeros(shape) for key, shape in shapes.items()},
                                       decode=lambda state: state)
        snapshots = []
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer",
                               side_effect=lambda gaussians, path, sh_degree_zero:
                               snapshots.append(gaussians["means"].clone())):
            for seed in (0, 0, 100):
                flow_cfg = {**self.flow_cfg, "eval_noise_seed": seed}
                SCRIPT.evaluate_scene(model, scene, standardizer, flow_cfg,
                                      str(Path(output_dir) / str(seed)), ply_only=True)
        torch.testing.assert_close(snapshots[0], snapshots[1])
        self.assertFalse(torch.equal(snapshots[0], snapshots[2]))

    def test_one_scene_without_augmentation_keeps_paired_render_metrics(self):
        model = FakeModel()
        scene = {"scene_idx": 0, "target_gs": self.dataset.gaussians, "gt_gs": self.dataset.gaussians,
                 "target_images": [torch.zeros(2, 2, 4)], "target_image_names": ["view.png"],
                 "target_cameras": {"camera_to_worlds": torch.eye(4).unsqueeze(0)}, "eval_chunk_size": 1}
        metric = mock.Mock()
        metric.finalize.return_value = {"psnr": 1.0}
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT.noise_to_gs, "sample_flow_model", return_value=self.dataset.gaussians), \
             mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs",
                               return_value=([torch.full((2, 2, 3), 0.75)], [torch.ones(2, 2, 1)])) as render_mock, \
             mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer"), \
             mock.patch.object(metrics, "MetricComputer", return_value=metric), \
             mock.patch.object(SCRIPT.cv2, "imwrite"):
            result = SCRIPT.evaluate_scene(model, scene, FakeStandardizer(), self.flow_cfg, output_dir)
            self.assertEqual(result, {"psnr": 1.0})
            render_mock.assert_called_once()
            metric.update.assert_called_once()
            self.assertTrue((Path(output_dir) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

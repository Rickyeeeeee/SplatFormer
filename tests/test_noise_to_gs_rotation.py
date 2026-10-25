"""Training rotation and rotated-render FID checks without Inception weights."""

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
        self.gaussians = {
            "means": torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.3, 0.4], [0.2, 0.9, 0.7]]),
            "scales": torch.ones(3, 3), "opacities": torch.ones(3, 1),
            "quats": torch.tensor([[1., 0., 0., 0.]]).expand(3, -1).clone(),
            "features_dc": torch.ones(3, 3), "features_rest": torch.empty(3, 0, 3),
        }

    def load_scene(self, scene_idx):
        return {"scene_name": f"scene_{scene_idx}", "data": {32: {
            "gs_params": self.gaussians,
            "images": [torch.ones(2, 2, 4)], "images_name": ["view.png"],
            "cameras": {"camera_to_worlds": torch.eye(4).unsqueeze(0), "background_color": torch.ones(3)},
        }}}


class FakeFID:
    instances = []

    def __init__(self, device):
        self.real = []
        self.fake = []
        self.history = []
        self.instances.append(self)

    def update(self, images, real):
        (self.real if real else self.fake).append(images.clone())

    def report(self):
        counts = {"real": sum(len(batch) for batch in self.real),
                  "generated": sum(len(batch) for batch in self.fake)}
        self.history.append(counts)
        self.fake.clear()
        return {"fid": 1.0, "fid_counts": counts}


class RotationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset()
        self.original = {key: value.clone() for key, value in self.dataset.gaussians.items()}
        self.flow_cfg = {"gs_statistics_path": "unused", "normalization_variance_floor": 1e-8,
                         "flow_t_eps": 1e-4, "flow_steps": 1, "eval_noise_seed": 0}
        self.flags = SimpleNamespace(predictor="dit", scene_mode="one", eval_max_views=1,
                                     batch_size=2, grad_accum_steps=1, eval_subdir="eval_final")

    def run_training(self, seed, rotate=True, coordinate_mode="morton_rank", scene_indices=(0,), steps=1):
        torch.manual_seed(seed)
        self.flags.scene_mode = "many" if len(scene_indices) > 1 else "one"
        FakeStandardizer.encoded = []
        FakeFID.instances = []
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

        def fake_evaluate(model, scene, standardizer, flow_cfg, output_dir, ground_truth=False,
                          fid=None, rotated_fid=None):
            image = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)
            if fid is not None:
                fid.update(image, real=ground_truth)
            if rotated_fid is not None and not ground_truth:
                rotated_fid.update(image, real=False)
            return {"psnr": 1.0}

        augmentation = {"random_rotate": rotate, "rotation_mode": "gravity_consistent",
                        "rotation_max_degrees": None, "rotation_pivot": (0.5, 0.5, 0.5),
                        "eval_rotation_seed": 7}
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT, "FLAGS", self.flags), \
             mock.patch.dict(SCRIPT.PREDICTORS, {"dit": build_model}), \
             mock.patch.object(SCRIPT, "GaussianStandardizer", FakeStandardizer), \
             mock.patch.object(SCRIPT, "flow_matching", return_value=self.flow_cfg), \
             mock.patch.object(SCRIPT, "training_augmentation", return_value=augmentation), \
             mock.patch.object(SCRIPT, "evaluate_scene", side_effect=fake_evaluate), \
             mock.patch.object(SCRIPT.noise_to_gs, "construct_path", side_effect=capture_path), \
             mock.patch.object(SCRIPT, "PooledFID", FakeFID), \
             mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs",
                               return_value=([torch.ones(2, 2, 3)], [torch.ones(2, 2, 1)])), \
             mock.patch.object(SCRIPT, "build_optimizer", side_effect=lambda model: torch.optim.SGD(model.parameters(), lr=0.01)), \
             mock.patch.object(SCRIPT, "build_scheduler", return_value=SimpleNamespace(step=lambda: None)), \
             mock.patch.object(SCRIPT.torch, "save"):
            SCRIPT.training(self.dataset, scene_indices, output_dir, mock.Mock(), torch.device("cpu"),
                            total_steps=steps, eval_interval=1, save_interval=steps + 1,
                            log_interval=1, enable_amp=False)
            report = json.loads((Path(output_dir) / "eval_final" / "metrics.json").read_text())
        return captured, FakeStandardizer.encoded[:], models[0], report, FakeFID.instances[:]

    def test_rotates_each_draw_before_encoding_and_resorts_morton_slots(self):
        targets, encoded, _, _, _ = self.run_training(12)
        self.assertEqual(len(targets), 2)
        self.assertFalse(torch.equal(encoded[1]["means"], encoded[2]["means"]))
        for target, physical in zip(targets, encoded[1:]):
            order = morton_order(physical["means"])
            for key in target:
                torch.testing.assert_close(target[key], physical[key][order])
            self.assertFalse(torch.equal(physical["quats"], self.original["quats"]))
        for key in self.original:
            torch.testing.assert_close(self.dataset.gaussians[key], self.original[key])

        repeated, _, _, _, _ = self.run_training(12)
        different, _, _, _, _ = self.run_training(13)
        for first, second in zip(targets, repeated):
            torch.testing.assert_close(first["means"], second["means"])
        self.assertFalse(torch.equal(targets[0]["means"], different[0]["means"]))

    def test_gt_mode_uses_rotated_reference_and_disabled_mode_keeps_format(self):
        targets, _, model, _, _ = self.run_training(4, coordinate_mode="gt")
        for reference, target in zip(model.references[0], targets):
            torch.testing.assert_close(reference, target["means"])
        canonical = self.original["means"][morton_order(self.original["means"])]
        targets, _, _, report, _ = self.run_training(4, rotate=False)
        for target in targets:
            torch.testing.assert_close(target["means"], canonical)
        self.assertNotIn("rotated_fid", report)

    def test_many_scene_rotated_real_pool_persists_across_checkpoints(self):
        _, _, _, report, metrics = self.run_training(6, scene_indices=(0, 1), steps=2)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[1].history, [{"real": 2, "generated": 2}] * 2)
        self.assertEqual(report["rotated_fid_counts"], {"real": 2, "generated": 2})
        self.assertEqual(report["fid_counts"], {"real": 2, "generated": 2})
        _, _, _, disabled_report, _ = self.run_training(6, rotate=False, scene_indices=(0, 1))
        self.assertIn("fid", disabled_report)
        self.assertNotIn("rotated_fid", disabled_report)

    def test_rotated_fid_uses_unmasked_generated_render(self):
        model = FakeModel()
        scene = {"scene_idx": 0, "target_gs": self.dataset.gaussians, "gt_gs": self.dataset.gaussians,
                 "target_images": [torch.zeros(2, 2, 4)], "target_image_names": ["view.png"],
                 "target_cameras": {"camera_to_worlds": torch.eye(4).unsqueeze(0)}, "eval_chunk_size": 1}
        canonical_fid, rotated_fid = FakeFID("cpu"), FakeFID("cpu")
        metric = mock.Mock()
        metric.finalize.return_value = {"psnr": 1.0}
        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT.noise_to_gs, "sample_flow_model", return_value=self.dataset.gaussians), \
             mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs",
                               return_value=([torch.full((2, 2, 3), 0.75)], [torch.ones(2, 2, 1)])), \
             mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer"), \
             mock.patch.object(SCRIPT, "MetricComputer", return_value=metric), \
             mock.patch.object(SCRIPT.cv2, "imwrite"):
            SCRIPT.evaluate_scene(model, scene, FakeStandardizer(), self.flow_cfg, output_dir,
                                  fid=canonical_fid, rotated_fid=rotated_fid)
        self.assertEqual(rotated_fid.fake[0][0, 0, 0, 0].item(), 191)
        self.assertEqual(canonical_fid.fake[0][0, 0, 0, 0].item(), 0)

    def test_launcher_exposes_rotation_and_names_its_output(self):
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        environment = dict(os.environ, RANDOM_ROTATE="True", ROTATION_MODE="gravity_consistent",
                           ROTATION_MAX_DEGREES="45", EVAL_ROTATION_SEED="9",
                           DATASET_ROOT="/tmp/noise-to-gs-test")
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("_rot_gravity_consistent_deg45_", result.stdout)
        self.assertIn("--gin_param=training_augmentation.random_rotate=True", result.stdout)
        self.assertIn("--gin_param=training_augmentation.eval_rotation_seed=9", result.stdout)
        environment["RANDOM_ROTATE"] = "False"
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertNotIn("_rot_gravity_consistent_", result.stdout)

    def test_noise_seed_changes_generated_pool_without_changing_real_pool(self):
        model = FakeModel()
        scene = {"scene_idx": 0, "target_gs": self.dataset.gaussians,
                 "target_images": [torch.zeros(2, 2, 3)], "target_image_names": ["view.png"],
                 "target_cameras": {"camera_to_worlds": torch.eye(4).unsqueeze(0)}, "eval_chunk_size": 1}
        shapes = {key: value.shape[1:] for key, value in self.dataset.gaussians.items()}
        standardizer = SimpleNamespace(means={key: torch.zeros(shape) for key, shape in shapes.items()},
                                       decode=lambda state: state)
        rotated_fid = FakeFID("cpu")
        rotated_fid.update(torch.ones(1, 2, 2, 3, dtype=torch.uint8), real=True)
        metric = mock.Mock()
        metric.finalize.return_value = {"psnr": 1.0}

        def render(gaussians, cameras, sh_degree_zero):
            color = torch.sigmoid(gaussians["means"].mean()).expand(2, 2, 3)
            return [color], [torch.ones(2, 2, 1)]

        with tempfile.TemporaryDirectory() as output_dir, \
             mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs", side_effect=render), \
             mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer"), \
             mock.patch.object(SCRIPT, "MetricComputer", return_value=metric), \
             mock.patch.object(SCRIPT.cv2, "imwrite"):
            for seed in (0, 100):
                flow_cfg = {**self.flow_cfg, "eval_noise_seed": seed}
                SCRIPT.evaluate_scene(model, scene, standardizer, flow_cfg,
                                      str(Path(output_dir) / str(seed)), rotated_fid=rotated_fid)
        self.assertFalse(torch.equal(rotated_fid.fake[0], rotated_fid.fake[1]))
        self.assertEqual(len(rotated_fid.real), 1)


if __name__ == "__main__":
    unittest.main()

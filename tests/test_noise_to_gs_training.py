"""Focused checks for unconditional training data and evaluation choices."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models.gaussian_dit_predictor import GaussianDiTPredictor
from utils.noise_to_gs_training import (PAD_OPACITY_LOGIT, evenly_spaced_camera_indices,
                                        evaluation_seed, mean_feature_losses, prepare_fixed_count_target,
                                        select_available_scenes)
from utils.spatial_coordinates import morton_order
from utils.gs_normalization import STATISTIC_KEYS


class NoiseToGSTrainingTests(unittest.TestCase):
    def test_feature_loss_sums_preserve_microbatch_and_rank_averages(self):
        # Two ranks each process two unequal microbatches, three scenes per rank.
        scenes = torch.arange(1., 37.).reshape(6, len(STATISTIC_KEYS))
        rank_sums = [scenes[:1].sum(0) + scenes[1:3].sum(0), scenes[3:5].sum(0) + scenes[5:].sum(0)]
        raw, weighted = mean_feature_losses(sum(rank_sums), 6, {"means": 2.0, "opacities": 0.0})
        for key, expected in zip(STATISTIC_KEYS, scenes.mean(0)):
            self.assertAlmostEqual(raw[key], expected.item())
            self.assertAlmostEqual(weighted[key], expected.item() * {"means": 2.0, "opacities": 0.0}.get(key, 1.0))

    def test_morton_dit_forward_needs_no_scene_or_target_coordinates(self):
        class RecordingBackbone(torch.nn.Module):
            def forward(self, data):
                self.input = data
                return torch.zeros_like(data["feat"])

        with mock.patch("models.gaussian_dit_predictor.GaussianDiT", return_value=RecordingBackbone()):
            model = GaussianDiTPredictor(sh_degree=0, max_gaussians=4, coordinate_mode="morton_rank")
        state = {"means": torch.zeros(4, 3), "scales": torch.zeros(4, 3),
                 "opacities": torch.zeros(4, 1), "quats": torch.zeros(4, 4),
                 "features_dc": torch.zeros(4, 3), "features_rest": torch.empty(4, 0, 3)}
        first = model([state], batch_scene_idx=[123], t=torch.tensor([0.5]))
        second = model([state], batch_scene_idx=[999], t=torch.tensor([0.5]))
        self.assertNotIn("coord", model.backbone.input)
        torch.testing.assert_close(first[0]["means"], second[0]["means"])

    def test_scene_discovery_keeps_only_listed_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "available" / "ckpts").mkdir(parents=True)
            (root / "available" / "ckpts" / "ckpt_9_rank0.pt").touch()
            (root / "03001627" / "model" / "ckpts").mkdir(parents=True)
            (root / "03001627" / "model" / "ckpts" / "ckpt_9_rank0.pt").touch()
            folders = [{"scene_name": name, "resolution_paths": {32: {"gsplat_dir": "old"}}}
                       for name in ("missing", "available", "available", "03001627/model")]
            dataset = SimpleNamespace(folders=folders, tgt_resolution=32)
            skipped = select_available_scenes(dataset, root)
            self.assertEqual(skipped, ["missing"])
            self.assertEqual(len(dataset.folders), 2)
            self.assertEqual(dataset.folders[0]["resolution_paths"][32]["gsplat_dir"], str(root / "available"))
            self.assertEqual(dataset.folders[1]["resolution_paths"][32]["gsplat_dir"], str(root / "03001627/model"))

    def test_scene_limit_selects_first_listed_scenes_and_requires_exact_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second", "third"):
                path = root / name / "ckpts"
                path.mkdir(parents=True)
                (path / "ckpt_9_rank0.pt").touch()
            entries = [{"scene_name": name, "resolution_paths": {32: {}}} for name in ("first", "second", "third")]
            selected = SimpleNamespace(folders=list(entries), tgt_resolution=32)
            self.assertEqual(select_available_scenes(selected, root, 2), [])
            self.assertEqual([entry["scene_name"] for entry in selected.folders], ["first", "second"])
            selected = SimpleNamespace(folders=list(entries), tgt_resolution=32)
            select_available_scenes(selected, root, -1)
            self.assertEqual(len(selected.folders), 3)
            for count in (0, -2, 4):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "scene_count"):
                    select_available_scenes(SimpleNamespace(folders=list(entries), tgt_resolution=32), root, count)
            for names in (["first", "missing", "third"], ["first", "first", "third"]):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, "First 2 listed scenes"):
                    folders = [{"scene_name": name, "resolution_paths": {32: {}}} for name in names]
                    select_available_scenes(SimpleNamespace(folders=folders, tgt_resolution=32), root, 2)

    def test_padding_is_finite_invisible_and_morton_ordered(self):
        gaussians = {
            "means": torch.tensor([[0.9, 0.5, 0.5], [0.1, 0.5, 0.5]]),
            "scales": torch.full((2, 3), -2.0),
            "opacities": torch.zeros(2, 1),
            "quats": torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            "features_dc": torch.zeros(2, 3),
            "features_rest": torch.empty(2, 0, 3),
        }
        padded = prepare_fixed_count_target(gaussians, 4)
        self.assertEqual(padded["means"].shape, (4, 3))
        self.assertEqual((padded["opacities"] == PAD_OPACITY_LOGIT).sum().item(), 2)
        self.assertTrue(torch.isfinite(torch.cat([value.flatten() for value in padded.values()])).all())
        torch.testing.assert_close(padded["means"][morton_order(padded["means"])], padded["means"])
        torch.testing.assert_close(padded["means"][1:3], torch.full((2, 3), 0.5))

    def test_current_coordinates_keep_input_order(self):
        gaussians = {"means": torch.tensor([[0.9, 0.5, 0.5], [0.1, 0.5, 0.5]]),
                     "opacities": torch.zeros(2, 1)}
        prepared = prepare_fixed_count_target(gaussians, 3, "current")
        torch.testing.assert_close(prepared["means"][:2], gaussians["means"])
        torch.testing.assert_close(prepared["means"][2], torch.full((3,), 0.5))

    def test_camera_and_sample_choices_are_stable_across_ranks(self):
        self.assertEqual(evenly_spaced_camera_indices(8), list(range(8)))
        indices = evenly_spaced_camera_indices(128)
        self.assertEqual((indices[0], indices[-1], len(set(indices))), (0, 127, 8))
        seeds = [evaluation_seed(42, slot) for slot in range(16)]
        two_ranks = sorted(evaluation_seed(42, slot) for rank in range(2) for slot in range(rank, 16, 2))
        self.assertEqual(seeds, two_ranks)


if __name__ == "__main__":
    unittest.main()

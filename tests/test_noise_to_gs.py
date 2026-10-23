"""Noise-only Gaussian path, current-coordinate, and launcher checks."""

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor
from models.feature_flow_predictor import GSFlowPredictor
from sr import noise_to_gs
from utils.gs_normalization import STATISTIC_KEYS


ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
    "features_dc": (3,), "features_rest": (3, 3),
}


class UnitStandardizer:
    def __init__(self):
        self.means = {key: torch.zeros(shape) for key, shape in SHAPES.items()}
        self.means["means"] = torch.tensor([2., 3., 4.])
        self.stds = {key: torch.ones(shape) for key, shape in SHAPES.items()}
        self.stds["means"] = torch.tensor([2., 2., 2.])

    def decode(self, state):
        return state


class RecordingPredictor(nn.Module):
    def __init__(self, backbone_type="GDPT"):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.backbone_type = backbone_type
        self.calls = []

    def forward(self, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
        self.calls.append((batch_flow_gs, batch_scene_idx, batch_reference_means, t))
        return [{key: torch.zeros_like(value) + self.anchor for key, value in state.items()}
                for state in batch_flow_gs]


class RecordingBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.output_dim = in_channels
        self.last = None

    def forward(self, data):
        self.last = data
        return SimpleNamespace(feat=data["feat"])


class NoiseToGaussianTests(unittest.TestCase):
    def setUp(self):
        self.standardizer = UnitStandardizer()

    def test_linear_path_uses_noise_and_target_only(self):
        target = {key: torch.full((3,) + shape, 2.) for key, shape in SHAPES.items()}
        noise = {key: torch.full_like(value, -1.) for key, value in target.items()}
        query, velocity = noise_to_gs.construct_path(target, 0.25, noise)
        for key in STATISTIC_KEYS:
            torch.testing.assert_close(query[key], torch.full_like(target[key], -0.25))
            torch.testing.assert_close(velocity[key], torch.full_like(target[key], 3.))

    def test_sampling_is_count_only_seeded_and_restores_mode(self):
        model = RecordingPredictor().train()
        first = noise_to_gs.sample_flow_model(model, 4, 3, 3, self.standardizer, noise_seed=8)
        second = noise_to_gs.sample_flow_model(model, 4, 3, 3, self.standardizer, noise_seed=8)
        third = noise_to_gs.sample_flow_model(model, 4, 3, 3, self.standardizer, noise_seed=9)
        self.assertTrue(model.training)
        self.assertEqual(tuple(first["means"].shape), (4, 3))
        for key in STATISTIC_KEYS:
            torch.testing.assert_close(first[key], second[key])
        self.assertFalse(torch.equal(first["means"], third["means"]))
        for states, indices, references, time in model.calls:
            self.assertEqual(indices, [3])
            self.assertEqual(len(references), 1)
            torch.testing.assert_close(references[0], states[0]["means"] * 2 + self.standardizer.means["means"])
            self.assertEqual(tuple(time.shape), (1,))

    def test_dit_receives_no_reference_coordinates(self):
        model = RecordingPredictor("DIT")
        noise_to_gs.sample_flow_model(model, 2, 1, 2, self.standardizer)
        self.assertTrue(all(reference is None for _, _, reference, _ in model.calls))

    def test_ptv3_and_diffusion_grid_shift_keeps_current_coordinates(self):
        means = torch.tensor([[-1.2, 0.1, -0.3], [0.4, -0.5, 0.2]])
        scene = {"means": means}
        with mock.patch("models.feature_flow_predictor.PointTransformerV3FlowModel", RecordingBackbone):
            ptv3 = GSFlowPredictor(
                sh_degree=1, input_features=["means"], input_feat_to_mlp=False,
                output_features=["means"], output_head_nlayer=1, output_head_type="mlp-relu",
                output_head_width=4, grid_resolution=10, resume_ckpt=None, zeroinit=False,
                res_feature_activation={"means": nn.Identity()}, shift_negative_grid_coords=True,
            )
        other_means = means - torch.tensor([3., 2., 1.])
        ptv3([scene, {"means": other_means}], [0, 1], t=[0.3, 0.7])
        torch.testing.assert_close(ptv3.backbone.last["coord"], torch.cat([means, other_means]))
        grid = ptv3.backbone.last["grid_coord"]
        self.assertTrue((grid >= 0).all())
        self.assertTrue((grid[:2].amin(dim=0) == 0).all())
        self.assertTrue((grid[2:].amin(dim=0) == 0).all())
        with mock.patch("models.diffusion_gaussian_predictor.DiffusionGaussianTransformer", RecordingBackbone):
            dipt = DiffusionGaussianPredictor(
                sh_degree=1, input_features=["means"], input_feat_to_mlp=False,
                output_features=["means"], output_head_nlayer=1, output_head_type="mlp-relu",
                output_head_width=4, grid_resolution=10, resume_ckpt=None, zeroinit=False,
                res_feature_activation={"means": nn.Identity()}, shift_negative_grid_coords=True,
            )
        dipt([scene], [0], t=0.3)
        torch.testing.assert_close(dipt.backbone.last["coord"], means)
        self.assertTrue((dipt.backbone.last["grid_coord"] >= 0).all())
        with mock.patch("models.diffusion_gaussian_predictor.GlobalDiffusionPointTransformer", RecordingBackbone):
            gdpt = DiffusionGaussianPredictor(
                sh_degree=1, input_features=["means"], input_feat_to_mlp=False,
                output_features=["means"], output_head_nlayer=1, output_head_type="mlp-relu",
                output_head_width=4, grid_resolution=10, resume_ckpt=None, zeroinit=False,
                res_feature_activation={"means": nn.Identity()}, shift_negative_grid_coords=True,
                backbone_type="gdpt",
            )
        gdpt([scene], [0], t=0.3)
        torch.testing.assert_close(gdpt.backbone.last["coord"], means)
        self.assertTrue((gdpt.backbone.last["grid_coord"] >= 0).all())

    def test_launcher_selects_all_common_flow_models(self):
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        for predictor in ("ptv3", "dipt", "gdpt", "dit"):
            environment = dict(os.environ, PREDICTOR=predictor, DATASET_ROOT="/tmp/noise-to-gs-test")
            result = subprocess.run(
                ["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                env=environment, capture_output=True, text=True, check=True,
            )
            self.assertIn(f"--predictor={predictor}", result.stdout)
            self.assertIn(f"--gin_file=configs/model/{'ptv3_flow' if predictor == 'ptv3' else predictor + '_gaussian'}.gin",
                          result.stdout)
            self.assertIn("--gin_param=SplatFactoSRDataset.max_gs_num=1000", result.stdout)
            self.assertIn("--gin_param=SplatFactoSRDataset.load_src_gs=False", result.stdout)
            if predictor == "ptv3":
                self.assertIn("--gin_param=GSFlowPredictor.shift_negative_grid_coords=True", result.stdout)
            elif predictor in ("dipt", "gdpt"):
                self.assertIn("--gin_param=DiffusionGaussianPredictor.shift_negative_grid_coords=True", result.stdout)


if __name__ == "__main__":
    unittest.main()

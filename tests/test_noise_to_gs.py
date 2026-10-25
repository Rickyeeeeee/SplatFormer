"""Noise-only Gaussian path, current-coordinate, and launcher checks."""

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from dataset.GS_SR import SplatFactoSRDataset
from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor
from models.feature_flow_predictor import GSFlowPredictor
from sr import interpolants, noise_to_gs
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

    def test_dit_uses_fixed_gt_coordinates_at_every_rollout_step(self):
        model = RecordingPredictor("DIT")
        reference = torch.randn(2, 3)
        noise_to_gs.sample_flow_model(model, 2, 1, 3, self.standardizer, reference_means=reference)
        self.assertEqual(len(model.calls), 3)
        for _, _, references, _ in model.calls:
            self.assertIs(references[0], reference)
        model.calls.clear()
        state = noise_to_gs.sample_noise(2, self.standardizer, torch.device("cpu"))
        noise_to_gs.predict_velocity(model, [state], [1], torch.tensor([0.5]), self.standardizer,
                                     reference_means=[reference])
        self.assertIs(model.calls[0][2][0], reference)

    def test_degree_zero_flow_and_loss_have_empty_rest(self):
        self.standardizer.means["features_rest"] = torch.empty((0, 3))
        self.standardizer.stds["features_rest"] = torch.empty((0, 3))
        target = {key: torch.ones((3,) + tuple(mean.shape)) for key, mean in self.standardizer.means.items()}
        query, velocity = noise_to_gs.construct_path(target, 0.4)
        model = RecordingPredictor("DIT")
        prediction = model([query], [0], None, torch.tensor([0.4]))[0]
        loss, parts, _ = interpolants.attribute_mse(prediction, velocity)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["features_rest"].item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.anchor.grad))
        result = noise_to_gs.sample_flow_model(model, 3, 0, 2, self.standardizer)
        self.assertEqual(tuple(result["features_rest"].shape), (3, 0, 3))

    def test_degree_zero_point_predictors_have_no_rest_head(self):
        scene = {key: torch.randn((2,) + shape) for key, shape in SHAPES.items()}
        scene["features_rest"] = torch.empty((2, 0, 3))
        keys = list(SHAPES)
        settings = dict(sh_degree=0, input_features=keys, input_feat_to_mlp=False,
                        output_features=keys, output_head_nlayer=1, output_head_type="mlp-relu",
                        output_head_width=4, grid_resolution=10, resume_ckpt=None, zeroinit=False,
                        res_feature_activation={key: nn.Identity() for key in keys})
        predictors = (
            ("models.feature_flow_predictor.PointTransformerV3FlowModel", GSFlowPredictor, {}),
            ("models.diffusion_gaussian_predictor.DiffusionGaussianTransformer", DiffusionGaussianPredictor, {}),
            ("models.diffusion_gaussian_predictor.GlobalDiffusionPointTransformer", DiffusionGaussianPredictor,
             {"backbone_type": "gdpt"}),
        )
        for backbone, predictor, options in predictors:
            with self.subTest(backbone=backbone), mock.patch(backbone, RecordingBackbone):
                model = predictor(**settings, **options)
                result = model([scene], [0], t=0.3)[0]
                self.assertNotIn("features_rest", model.input_features)
                self.assertNotIn("features_rest", model.features_outputhead)
                self.assertEqual(tuple(result["features_rest"].shape), (2, 0, 3))
                loss, _, _ = interpolants.attribute_mse(result, scene)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.features_outputhead["means"][-1].weight.grad).all())

    def test_equal_resolution_target_only_scene_loads_gs_and_images(self):
        with mock.patch("dataset.GS_SR.gs_io.read_scene_names", return_value=["fixture"]):
            dataset = SplatFactoSRDataset(
                train_or_test="test", dataset_root="/tmp/fixture", train_scene_list="fixture.csv",
                test_scene_list="fixture.csv", image_per_scene=1, remove_outlier_ndevs=-1,
                max_gs_num=1000, split_across_gpus=False, fit_lr_to_hr_root="/tmp/fixture",
                fit_hr_to_lr_root="/tmp/fixture", src_resolution=32, tgt_resolution=32,
                load_src_gs=False, load_src_images=False, load_tgt_gs=True, load_tgt_images=True,
            )
        gaussians = {key: torch.zeros((2,) + shape) for key, shape in SHAPES.items()}
        gaussians["means"][1] = 1
        gaussians["quats"][:, 0] = 1
        gaussians["features_rest"] = gaussians["features_rest"][:, :0, :]
        camera = torch.eye(4).unsqueeze(0)
        camera_meta = {"meta": {"camera_to_worlds": camera}}
        views = ([torch.zeros((2, 2, 3))], ["view.png"], {"camera_to_worlds": camera})
        with mock.patch("dataset.GS_SR.gs_io.load_gsplat", return_value=(gaussians, "ckpt")), \
             mock.patch.object(dataset, "load_camera_meta", return_value=camera_meta), \
             mock.patch.object(dataset, "load_resolution_views", return_value=views):
            scene = dataset.load_scene(0)
        self.assertEqual(list(scene["data"]), [32])
        self.assertEqual(tuple(scene["data"][32]["gs_params"]["features_rest"].shape), (2, 0, 3))
        self.assertEqual(len(scene["data"][32]["images"]), 1)

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
            environment = dict(os.environ, PREDICTOR=predictor, DATASET_ROOT="/tmp/noise-to-gs-test",
                               SOURCE_GS_ROOT="/tmp/mcmc-500", TARGET_GS_ROOT="/tmp/mcmc-1000", GS_SH_DEGREE="0")
            result = subprocess.run(
                ["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                env=environment, capture_output=True, text=True, check=True,
            )
            self.assertIn(f"--predictor={predictor}", result.stdout)
            self.assertIn(f"--gin_file=configs/model/{'ptv3_flow' if predictor == 'ptv3' else predictor + '_gaussian'}.gin",
                          result.stdout)
            self.assertIn("--gin_param=SplatFactoSRDataset.max_gs_num=1000", result.stdout)
            self.assertIn("--gin_param=SplatFactoSRDataset.load_src_gs=False", result.stdout)
            self.assertIn("--source_gs_root=/tmp/mcmc-500", result.stdout)
            self.assertIn("--target_gs_root=/tmp/mcmc-1000", result.stdout)
            predictor_class = "GaussianDiTPredictor" if predictor == "dit" else (
                "GSFlowPredictor" if predictor == "ptv3" else "DiffusionGaussianPredictor"
            )
            self.assertIn(f"--gin_param={predictor_class}.sh_degree=0", result.stdout)
            if predictor == "ptv3":
                self.assertIn("--gin_param=GSFlowPredictor.shift_negative_grid_coords=True", result.stdout)
            elif predictor in ("dipt", "gdpt"):
                self.assertIn("--gin_param=DiffusionGaussianPredictor.shift_negative_grid_coords=True", result.stdout)

        environment = dict(os.environ, PREDICTOR="dit", DATASET_ROOT="/tmp/noise-to-gs-test",
                           GS_COORDINATE_MODE="gt")
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("--gin_param=GaussianDiTPredictor.coordinate_mode='gt'", result.stdout)

        environment["GS_COORDINATE_MODE"] = "morton_rank"
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-noise-to-gs.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("--gin_param=GaussianDiTPredictor.coordinate_mode='morton_rank'", result.stdout)


if __name__ == "__main__":
    unittest.main()

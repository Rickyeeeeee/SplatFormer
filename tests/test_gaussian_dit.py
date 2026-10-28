"""Focused checks for the standalone FlashAttention Gaussian DiT."""

import math
import unittest
from unittest import mock

import gin
import torch
from torch import nn

from models.gaussian_dit import GaussianDiT, flash_attn
from models.gaussian_dit_predictor import GaussianDiTPredictor
from sr.noise_to_gs import construct_path
from utils.spatial_coordinates import morton_order


SHAPES = {
    "means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
    "features_dc": (3,), "features_rest": (3, 3),
}


class CPUBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, max_gaussians, position_mode="coordinate"):
        super().__init__()
        self.position_mode = position_mode
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, data):
        self.last_input = data
        return self.linear(data["feat"])


class GaussianDiTTests(unittest.TestCase):
    def setUp(self):
        gin.clear_config()
        gin.bind_parameter("GaussianDiT.depth", 2)
        gin.bind_parameter("GaussianDiT.channels", 32)
        gin.bind_parameter("GaussianDiT.num_heads", 4)
        gin.bind_parameter("GaussianDiT.frequency_embedding_size", 32)
        torch.manual_seed(7)

    def tearDown(self):
        gin.clear_config()

    def scene(self, count, device="cpu"):
        return {key: torch.randn((count,) + shape, device=device) for key, shape in SHAPES.items()}

    def test_count_and_time_validation_before_cuda(self):
        model = GaussianDiTPredictor()
        with self.assertRaisesRegex(ValueError, "1 to 1000"):
            model([self.scene(0)], t=0.5)
        with self.assertRaisesRegex(ValueError, "1 to 1000"):
            model([self.scene(1001)], t=0.5)
        with self.assertRaisesRegex(ValueError, "timestep"):
            model([self.scene(2), self.scene(3)], t=[0.1, 0.2, 0.3])
        with self.assertRaisesRegex(RuntimeError, "flash_attn and CUDA"):
            model([self.scene(1000)], t=0.5)

    def test_degree_zero_forward_and_backward_on_cpu(self):
        scene = self.scene(3)
        scene["features_rest"] = scene["features_rest"][:, :0, :]
        with mock.patch("models.gaussian_dit_predictor.GaussianDiT", CPUBackbone):
            model = GaussianDiTPredictor(sh_degree=0)
        self.assertNotIn("features_rest", model.feature_channels)
        prediction = model([scene], t=0.5)[0]
        self.assertEqual(tuple(prediction["features_rest"].shape), (3, 0, 3))
        loss = (prediction["means"] - scene["means"]).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.backbone.linear.weight.grad).all())

    def test_fourier_inputs_and_output_shapes(self):
        for degree in (0, 1):
            scene = self.scene(3)
            if degree == 0:
                scene["features_rest"] = scene["features_rest"][:, :0, :]
            with mock.patch("models.gaussian_dit_predictor.GaussianDiT", CPUBackbone):
                model = GaussianDiTPredictor(sh_degree=degree, fourier_input_features=["means", "quats"],
                                             fourier_num_frequencies={"means": 2, "quats": 1},
                                             fourier_include_raw=False)
            result = model([scene], t=0.5)[0]
            features = model.backbone.last_input["feat"]
            torch.testing.assert_close(features[:, :12], model.fourier_encoders["means"](scene["means"]))
            self.assertEqual(features.shape[1], sum(model.feature_channels.values()) - 7 + 12 + 8)
            for key in scene:
                self.assertEqual(result[key].shape, scene[key].shape)
        with self.assertRaisesRegex(ValueError, "Missing fourier_num_frequencies"):
            GaussianDiTPredictor(fourier_input_features=["means"])

    def test_coordinate_modes_keep_flow_features_and_select_coordinates(self):
        scenes = [self.scene(2), self.scene(3)]
        references = [torch.randn_like(scene["means"]) for scene in scenes]
        with mock.patch("models.gaussian_dit_predictor.GaussianDiT", CPUBackbone):
            current = GaussianDiTPredictor(coordinate_mode="current")
            gt = GaussianDiTPredictor(coordinate_mode="gt")
        current(scenes, t=[0.2, 0.7], batch_reference_means=references)
        gt(scenes, t=[0.2, 0.7], batch_reference_means=references)
        torch.testing.assert_close(current.backbone.last_input["coord"], torch.cat([scene["means"] for scene in scenes]))
        torch.testing.assert_close(gt.backbone.last_input["coord"], torch.cat(references))
        torch.testing.assert_close(gt.backbone.last_input["feat"], current.backbone.last_input["feat"])
        torch.testing.assert_close(gt.backbone.last_input["feat"][:, :3],
                                   torch.cat([scene["means"] for scene in scenes]))

    def test_gt_coordinate_mode_requires_aligned_references(self):
        scenes = [self.scene(2), self.scene(3)]
        with mock.patch("models.gaussian_dit_predictor.GaussianDiT", CPUBackbone):
            with self.assertRaisesRegex(ValueError, "coordinate_mode"):
                GaussianDiTPredictor(coordinate_mode="unknown")
            model = GaussianDiTPredictor(coordinate_mode="gt")
        with self.assertRaisesRegex(ValueError, "one reference means tensor per scene"):
            model(scenes, t=0.2)
        with self.assertRaisesRegex(ValueError, "one reference means tensor per scene"):
            model(scenes, t=0.2, batch_reference_means=[scenes[0]["means"]])
        with self.assertRaisesRegex(ValueError, r"shape \(3, 3\)"):
            model(scenes, t=0.2, batch_reference_means=[scenes[0]["means"], scenes[1]["means"][:2]])

    def test_morton_order_and_attribute_alignment(self):
        means = torch.tensor([[2., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        target = self.scene(3)
        target["means"] = means
        target["features_dc"][:, 0] = torch.tensor([20., 0., 10.])
        order = morton_order(means)
        torch.testing.assert_close(order, torch.tensor([1, 2, 0]))
        sorted_target = {key: value[order] for key, value in target.items()}
        noise = {key: torch.zeros_like(value) for key, value in sorted_target.items()}
        _, velocity = construct_path(sorted_target, 0.5, noise)
        torch.testing.assert_close(velocity["means"][:, 0], torch.tensor([0., 1., 2.]))
        torch.testing.assert_close(velocity["features_dc"][:, 0], torch.tensor([0., 10., 20.]))
        permuted = means[torch.tensor([2, 0, 1])]
        torch.testing.assert_close(permuted[morton_order(permuted)], means[order])

    def test_morton_rank_mode_uses_slots_without_reference_coordinates(self):
        scenes = [self.scene(2), self.scene(3)]
        with mock.patch("models.gaussian_dit_predictor.GaussianDiT", CPUBackbone):
            model = GaussianDiTPredictor(coordinate_mode="morton_rank")
        model(scenes, t=[0.2, 0.7])
        self.assertEqual(model.backbone.position_mode, "morton_rank")
        self.assertNotIn("coord", model.backbone.last_input)
        torch.testing.assert_close(model.backbone.last_input["offset"], torch.tensor([2, 5]))
        torch.testing.assert_close(model.backbone.last_input["feat"][:, :3],
                                   torch.cat([scene["means"] for scene in scenes]))

        backbone = GaussianDiT(in_channels=3, out_channels=3, max_gaussians=5, channels=32,
                               num_heads=4, depth=1, position_mode="morton_rank")
        self.assertFalse(hasattr(backbone, "coordinate_embedding"))
        torch.testing.assert_close(backbone.position_embedding[0, :4], torch.tensor([0., 1., 0., 1.]))
        torch.testing.assert_close(backbone.position_embedding[1, :2], torch.tensor([math.sin(1), math.cos(1)]))

    def test_patch_size_one_keeps_checkpoint_layout(self):
        kwargs = {"in_channels": 3, "out_channels": 3, "max_gaussians": 5, "channels": 32,
                  "num_heads": 4, "depth": 1, "position_mode": "morton_rank"}
        original = GaussianDiT(**kwargs)
        explicit = GaussianDiT(**kwargs, patch_size=1)
        self.assertEqual(set(original.state_dict()), set(explicit.state_dict()))
        self.assertFalse(hasattr(explicit, "patch_embedding"))
        explicit.load_state_dict(original.state_dict(), strict=True)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_patch_size_one_matches_explicit_forward(self):
        kwargs = {"in_channels": 3, "out_channels": 3, "max_gaussians": 5, "channels": 32,
                  "num_heads": 4, "depth": 1, "position_mode": "morton_rank"}
        original = GaussianDiT(**kwargs).cuda().eval()
        explicit = GaussianDiT(**kwargs, patch_size=1).cuda().eval()
        explicit.load_state_dict(original.state_dict(), strict=True)
        data = {"feat": torch.randn(3, 3, device="cuda"), "offset": torch.tensor([3], device="cuda"),
                "timesteps": torch.tensor([0.5], device="cuda")}
        with torch.no_grad():
            torch.testing.assert_close(original(data), explicit(data))

    def test_patch_size_validation(self):
        kwargs = {"in_channels": 3, "out_channels": 3, "position_mode": "morton_rank"}
        for size in (0, -1, 1.5, "2", True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "patch_size"):
                GaussianDiT(**kwargs, patch_size=size)
        with self.assertRaisesRegex(ValueError, "morton_rank"):
            GaussianDiT(in_channels=3, out_channels=3, position_mode="coordinate", patch_size=4)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_patched_morton_rank_shapes_scene_isolation_and_gradients(self):
        gin.bind_parameter("GaussianDiT.patch_size", 4)
        model = GaussianDiTPredictor(coordinate_mode="morton_rank").cuda().train()
        with torch.no_grad():
            torch.nn.init.normal_(model.backbone.final_layer.linear.weight, std=0.02)
        scenes = [self.scene(count, "cuda") for count in (2, 5, 8)]
        times = [0.2, 0.5, 0.8]
        boundaries = []

        def record_boundaries(_, inputs):
            boundaries.append((inputs[3].tolist(), inputs[4]))

        handle = model.backbone.blocks[0].register_forward_pre_hook(record_boundaries)
        try:
            batch = model(scenes, t=times)
        finally:
            handle.remove()
        self.assertEqual(boundaries, [([0, 1, 3, 5], 2)])
        for scene, output in zip(scenes, batch):
            for key in SHAPES:
                self.assertEqual(output[key].shape, scene[key].shape)
        with torch.no_grad():
            separate = [model([scene], t=time)[0] for scene, time in zip(scenes, times)]
        for actual, expected in zip(batch, separate):
            for key in SHAPES:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-3, atol=2e-3)
        sum(output["means"].square().mean() for output in batch).backward()
        for parameter in (model.backbone.feature_embedding[0].weight, model.backbone.patch_embedding.weight,
                          model.backbone.final_layer.linear.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_morton_rank_batch_matches_individual_scenes(self):
        model = GaussianDiTPredictor(coordinate_mode="morton_rank").cuda().eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.backbone.final_layer.linear.weight, std=0.02)
        scenes = [self.scene(3, "cuda"), self.scene(5, "cuda")]
        with torch.no_grad():
            batch = model(scenes, t=[0.2, 0.7])
            separate = [model([scene], t=time)[0] for scene, time in zip(scenes, [0.2, 0.7])]
        for actual, expected in zip(batch, separate):
            for key in SHAPES:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_shapes_scene_isolation_permutation_time_and_reference(self):
        model = GaussianDiTPredictor().cuda().eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.backbone.final_layer.linear.weight, std=0.02)
            torch.nn.init.normal_(model.backbone.final_layer.adaLN_modulation[-1].weight, std=0.02)
        scenes = [self.scene(3, "cuda"), self.scene(5, "cuda")]
        times = torch.tensor([0.2, 0.7], device="cuda")
        with torch.no_grad():
            output = model(scenes, [0, 1], t=times)
            separate = [model([scene], [index], t=times[index])[0] for index, scene in enumerate(scenes)]
            reference = [torch.randn_like(scene["means"]) for scene in scenes]
            changed_reference = model(scenes, [0, 1], t=times, batch_reference_means=reference)
            permutation = torch.randperm(3, device="cuda")
            permuted = model([{key: value[permutation] for key, value in scenes[0].items()}], t=times[:1])[0]
            changed_time = model(scenes, t=torch.tensor([0.9, 0.7], device="cuda"))

        for index, scene in enumerate(scenes):
            for key, shape in SHAPES.items():
                self.assertEqual(tuple(output[index][key].shape), (len(scene["means"]),) + shape)
                torch.testing.assert_close(output[index][key], separate[index][key], rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(output[index][key], changed_reference[index][key])
        for key in SHAPES:
            torch.testing.assert_close(permuted[key], output[0][key][permutation], rtol=2e-3, atol=2e-3)
        self.assertGreater((changed_time[0]["means"] - output[0]["means"]).abs().max().item(), 1e-6)
        for key in SHAPES:
            torch.testing.assert_close(changed_time[1][key], output[1][key], rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_gradients_reach_embedding_after_output_update(self):
        model = GaussianDiTPredictor().cuda().train()
        scene = self.scene(4, "cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(2):
            optimizer.zero_grad()
            prediction = model([scene], t=0.4)[0]["means"]
            loss = (prediction - scene["means"]).square().mean()
            loss.backward()
            if step == 0:
                self.assertGreater(model.backbone.final_layer.linear.weight.grad.abs().sum().item(), 0)
            else:
                gradient = model.backbone.feature_embedding[0].weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_maximum_count_runs(self):
        model = GaussianDiTPredictor().cuda().eval()
        with torch.no_grad():
            output = model([self.scene(1000, "cuda")], t=0.5)[0]
        self.assertEqual(tuple(output["means"].shape), (1000, 3))


if __name__ == "__main__":
    unittest.main()

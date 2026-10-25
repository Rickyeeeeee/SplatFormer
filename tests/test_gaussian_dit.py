"""Focused checks for the standalone FlashAttention Gaussian DiT."""

import unittest
from unittest import mock

import gin
import torch
from torch import nn

from models.gaussian_dit import flash_attn
from models.gaussian_dit_predictor import GaussianDiTPredictor


SHAPES = {
    "means": (3,), "scales": (3,), "opacities": (1,), "quats": (4,),
    "features_dc": (3,), "features_rest": (3, 3),
}


class CPUBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, max_gaussians):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, data):
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

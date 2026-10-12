"""Synthetic correctness checks for scene-wide Gaussian attention."""

import copy
import os
from pathlib import Path
import subprocess
import unittest

import gin
import torch
from torch import nn

from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor
from models.global_diffusion_point_transformer import GlobalDiffusionPointTransformer, flash_attn


ROOT = Path(__file__).resolve().parents[1]


class GlobalDiffusionPointTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def make_input(self, device="cpu"):
        return {
            "feat": torch.randn(8, 7, device=device, requires_grad=True),
            "coord": torch.randn(8, 3, device=device),
            "offset": torch.tensor([3, 8], device=device),
            "timesteps": torch.tensor([0.2, 0.7], device=device),
        }

    def test_sdpa_scene_isolation_permutation_time_and_gradients(self):
        model = GlobalDiffusionPointTransformer(7, depth=2, channels=32, num_head=4,
                                                frequency_embedding_size=32, attention_backend="sdpa").eval()
        state = self.make_input()
        result = model(state).feat
        self.assertEqual(tuple(result.shape), (8, 32))
        torch.testing.assert_close(result[:3], model({**state, "feat": state["feat"][:3],
                                                       "coord": state["coord"][:3], "offset": state["offset"][:1],
                                                       "timesteps": state["timesteps"][:1]}).feat)
        torch.testing.assert_close(result[3:], model({**state, "feat": state["feat"][3:],
                                                       "coord": state["coord"][3:], "offset": torch.tensor([5]),
                                                       "timesteps": state["timesteps"][1:]}).feat)
        permutation = torch.cat([torch.randperm(3), 3 + torch.randperm(5)])
        permuted = model({**state, "feat": state["feat"][permutation],
                          "coord": state["coord"][permutation]}).feat
        torch.testing.assert_close(permuted, result[permutation])
        changed_time = model({**state, "timesteps": torch.tensor([0.9, 0.7])}).feat
        self.assertGreater((changed_time[:3] - result[:3]).abs().max().item(), 1e-6)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(state["feat"].grad).all())
        self.assertGreater(state["feat"].grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_flash_matches_sdpa_and_backpropagates(self):
        state = self.make_input("cuda")
        sdpa = GlobalDiffusionPointTransformer(7, depth=2, channels=32, num_head=4,
                                               frequency_embedding_size=32, attention_backend="sdpa").cuda().eval()
        flash = copy.deepcopy(sdpa)
        for block in flash.blocks:
            block.attention.attention_backend = "flash"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            expected = sdpa(state).feat
            actual = flash(state).feat
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(state["feat"].grad).all())
        self.assertGreater(state["feat"].grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_sparse_conv_flash_scene_isolation_and_gradients(self):
        state = self.make_input("cuda")
        state["coord"] = torch.tensor([
            [0.1, 0.1, 0.1], [0.11, 0.1, 0.1], [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1], [0.11, 0.1, 0.1], [0.3, 0.3, 0.3],
            [0.31, 0.3, 0.3], [0.8, 0.8, 0.8],
        ], device="cuda")
        state["grid_coord"] = torch.floor(state["coord"] * 100).int()
        model = GlobalDiffusionPointTransformer(7, depth=2, channels=32, num_head=4,
                                                frequency_embedding_size=32, attention_backend="flash",
                                                use_sparse_conv=True).cuda().train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model(state).feat
            first = model({**state, "feat": state["feat"][:3], "coord": state["coord"][:3],
                           "grid_coord": state["grid_coord"][:3], "offset": torch.tensor([3], device="cuda"),
                           "timesteps": state["timesteps"][:1]}).feat
            second = model({**state, "feat": state["feat"][3:], "coord": state["coord"][3:],
                            "grid_coord": state["grid_coord"][3:], "offset": torch.tensor([5], device="cuda"),
                            "timesteps": state["timesteps"][1:]}).feat
            permutation = torch.cat([torch.randperm(3, device="cuda"), 3 + torch.randperm(5, device="cuda")])
            permuted = model({**state, "feat": state["feat"][permutation],
                              "coord": state["coord"][permutation],
                              "grid_coord": state["grid_coord"][permutation]}).feat
        torch.testing.assert_close(result[:3], first, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(result[3:], second, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(permuted, result[permutation], rtol=2e-2, atol=2e-2)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(state["feat"].grad).all())
        self.assertTrue(torch.isfinite(model.blocks[0].cpe_conv.weight.grad).all())
        self.assertGreater(model.blocks[0].cpe_conv.weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn is not None, "CUDA FlashAttention is required")
    def test_gin_predictor_optimizer_step(self):
        gin.clear_config()
        try:
            gin.parse_config_file(str(ROOT / "configs/model/gdpt_gaussian.gin"))
            gin.bind_parameter("GlobalDiffusionPointTransformer.depth", 2)
            gin.bind_parameter("GlobalDiffusionPointTransformer.channels", 32)
            gin.bind_parameter("GlobalDiffusionPointTransformer.num_head", 4)
            gin.bind_parameter("GlobalDiffusionPointTransformer.frequency_embedding_size", 32)
            gin.bind_parameter("DiffusionGaussianPredictor.zeroinit", False)
            gin.bind_parameter("DiffusionGaussianPredictor.res_feature_activation", {
                key: nn.Identity() for key in ("means", "scales", "opacities", "quats", "features_dc", "features_rest")
            })
            model = DiffusionGaussianPredictor().cuda().train()
            self.assertEqual(model.backbone_type, "GDPT")
            scenes = []
            for count in (3, 5):
                scenes.append({
                    "means": torch.randn(count, 3, device="cuda"),
                    "scales": torch.randn(count, 3, device="cuda"),
                    "opacities": torch.randn(count, 1, device="cuda"),
                    "quats": torch.randn(count, 4, device="cuda"),
                    "features_dc": torch.randn(count, 3, device="cuda"),
                    "features_rest": torch.randn(count, 3, 3, device="cuda"),
                })
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(scenes, [0, 1], t=torch.tensor([0.2, 0.7], device="cuda"))
                loss = sum((prediction["means"] - scene["means"]).square().mean()
                           for prediction, scene in zip(output, scenes))
            self.assertEqual(tuple(output[1]["features_rest"].shape), (5, 3, 3))
            loss.backward()
            self.assertTrue(torch.isfinite(model.backbone.embedding[0].weight.grad).all())
            self.assertGreater(model.backbone.embedding[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
        finally:
            gin.clear_config()

    def test_launcher_selects_gdpt_and_backend(self):
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        environment = dict(os.environ, PREDICTOR="gdpt", GDPT_ATTENTION_BACKEND="sdpa",
                           GDPT_USE_SPARSE_CONV="True", GS_SHIFT_NEGATIVE_GRID_COORDS="True",
                           quaternion_suffix="")
        result = subprocess.run(["bash", "-c", command, "test", str(ROOT / "scripts/overfit-sr-interpolants.sh")],
                                env=environment, capture_output=True, text=True, check=True)
        self.assertIn("--predictor=gdpt", result.stdout)
        self.assertIn("--gin_file=configs/model/gdpt_gaussian.gin", result.stdout)
        self.assertIn("--gin_param=GlobalDiffusionPointTransformer.attention_backend='sdpa'", result.stdout)
        self.assertIn("--gin_param=GlobalDiffusionPointTransformer.use_sparse_conv=True", result.stdout)
        self.assertIn("--gin_param=DiffusionGaussianPredictor.shift_negative_grid_coords=True", result.stdout)
        self.assertIn("_gdpt", next(line for line in result.stdout.splitlines() if line.startswith("--output_dir=")))


if __name__ == "__main__":
    unittest.main()

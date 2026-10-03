"""GPU smoke checks for shared-weight multi-partition DiPT attention."""

import unittest
from unittest import mock

import torch

from models import diffusion_gaussian_transformer as dipt
from utils.dipt_partitions import partition_views


@unittest.skipUnless(torch.cuda.is_available() and dipt.flash_attn is not None, "CUDA Flash Attention is required")
class MultiPartitionDiPTTests(unittest.TestCase):
    def test_modes_share_parameters_and_backpropagate(self):
        device = "cuda"
        coordinates = torch.tensor([[x, y, 1] for x in range(4) for y in range(4)], device=device)
        reference = coordinates.float() / 16
        original = dipt.DiffusionGaussianTransformer(
            in_channels=6, depth=1, channels=24, num_head=3, patch_size=4,
            order=("z", "z-trans"), drop_path=0, partition_mode="single",
        ).to(device).eval()
        parameter_count = sum(parameter.numel() for parameter in original.parameters())
        for mode in ("single", "rotated", "shifted", "both"):
            model = dipt.DiffusionGaussianTransformer(
                in_channels=6, depth=1, channels=24, num_head=3, patch_size=4,
                order=("z", "z-trans"), drop_path=0, partition_mode=mode,
            ).to(device).eval()
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)
            features = torch.randn(16, 6, device=device, requires_grad=True)
            state = {"coord": reference, "grid_coord": coordinates, "grid_size": torch.full((3,), 1 / 16, device=device),
                     "offset": torch.tensor([16], device=device), "feat": features,
                     "timesteps": torch.tensor([0.5], device=device), "shift_negative_grid_coords": False}
            output = model(state).feat
            self.assertEqual(tuple(output.shape), (16, 24))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            if mode == "both":
                with mock.patch.object(dipt, "partition_views", side_effect=lambda mode, size: list(reversed(partition_views(mode, size)))):
                    reversed_output = model({**state, "feat": features.detach()}).feat
                torch.testing.assert_close(output.detach(), reversed_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

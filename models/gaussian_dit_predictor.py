"""Gaussian attribute velocity predictor using a classic DiT backbone."""

import gin
import torch
from torch import nn

from .gaussian_dit import GaussianDiT


FEATURE_CHANNELS = {
    "means": 3,
    "scales": 3,
    "opacities": 1,
    "quats": 4,
    "features_dc": 3,
    "features_rest": 3,
}


@gin.configurable
class GaussianDiTPredictor(nn.Module):
    """Return one standardized velocity dictionary per Gaussian scene."""

    def __init__(self, sh_degree=1, max_gaussians=1000):
        super().__init__()
        if sh_degree < 1:
            raise ValueError("GaussianDiTPredictor requires sh_degree >= 1")
        self.sh_degree = sh_degree
        self.max_gaussians = max_gaussians
        self.backbone_type = "DIT"
        self.feature_channels = dict(FEATURE_CHANNELS)
        self.feature_channels["features_rest"] = 3 * ((sh_degree + 1) ** 2 - 1)
        self.input_features = list(self.feature_channels)
        self.output_features = list(self.feature_channels)
        width = sum(self.feature_channels.values())
        self.backbone = GaussianDiT(in_channels=width, out_channels=width, max_gaussians=max_gaussians)

    def apply_feature_update(self, feature, value, update, step_scale=1.0):
        del feature
        return value + step_scale * update

    def forward(self, batch_flow_gs, batch_scene_idx=None, t=None, batch_reference_means=None):
        del batch_scene_idx, batch_reference_means
        if not batch_flow_gs:
            raise ValueError("GaussianDiTPredictor requires at least one scene")
        if t is None:
            raise ValueError("GaussianDiTPredictor requires time t")

        counts = [len(gs["means"]) for gs in batch_flow_gs]
        if any(count < 1 or count > self.max_gaussians for count in counts):
            raise ValueError(f"Each scene must contain 1 to {self.max_gaussians} Gaussians")
        packed = []
        for gs, count in zip(batch_flow_gs, counts):
            values = []
            for key, width in self.feature_channels.items():
                value = gs[key]
                shape = (count, width // 3, 3) if key == "features_rest" else (count, width)
                if value.shape != shape:
                    raise ValueError(f"Expected {key} shape {shape}, got {tuple(value.shape)}")
                values.append(value.reshape(count, width))
            packed.append(torch.cat(values, dim=-1))

        device = packed[0].device
        timesteps = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(len(counts))
        if timesteps.numel() != len(counts):
            raise ValueError(f"Expected one timestep per scene, got {timesteps.numel()}")
        offset = torch.tensor(counts, device=device, dtype=torch.long).cumsum(0)
        velocity = self.backbone({
            "feat": torch.cat(packed),
            "coord": torch.cat([gs["means"] for gs in batch_flow_gs]),
            "offset": offset,
            "timesteps": timesteps,
        })

        widths = list(self.feature_channels.values())
        outputs = []
        for scene_velocity in velocity.split(counts):
            attributes = {}
            for key, value in zip(self.output_features, scene_velocity.split(widths, dim=-1)):
                attributes[key] = value.reshape(len(scene_velocity), -1, 3) if key == "features_rest" else value
            outputs.append(attributes)
        return outputs

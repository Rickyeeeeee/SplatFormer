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
    """Return scene velocities; Morton-rank mode expects ordered training targets."""

    def __init__(self, sh_degree=1, max_gaussians=1000, coordinate_mode="current"):
        super().__init__()
        if sh_degree < 0:
            raise ValueError("GaussianDiTPredictor requires sh_degree >= 0")
        if coordinate_mode not in ("current", "gt", "morton_rank"):
            raise ValueError(f"Unsupported coordinate_mode={coordinate_mode!r}; expected 'current', 'gt', or 'morton_rank'")
        self.sh_degree = sh_degree
        self.max_gaussians = max_gaussians
        self.coordinate_mode = coordinate_mode
        self.backbone_type = "DIT"
        self.feature_channels = dict(FEATURE_CHANNELS)
        if sh_degree:
            self.feature_channels["features_rest"] = 3 * ((sh_degree + 1) ** 2 - 1)
        else:
            del self.feature_channels["features_rest"]
        self.input_features = list(self.feature_channels)
        self.output_features = list(self.feature_channels)
        width = sum(self.feature_channels.values())
        position_mode = "morton_rank" if coordinate_mode == "morton_rank" else "coordinate"
        self.backbone = GaussianDiT(in_channels=width, out_channels=width, max_gaussians=max_gaussians,
                                    position_mode=position_mode)

    def apply_feature_update(self, feature, value, update, step_scale=1.0):
        del feature
        return value + step_scale * update

    def forward(self, batch_flow_gs, batch_scene_idx=None, t=None, batch_reference_means=None):
        del batch_scene_idx
        if not batch_flow_gs:
            raise ValueError("GaussianDiTPredictor requires at least one scene")
        if t is None:
            raise ValueError("GaussianDiTPredictor requires time t")

        counts = [len(gs["means"]) for gs in batch_flow_gs]
        if any(count < 1 or count > self.max_gaussians for count in counts):
            raise ValueError(f"Each scene must contain 1 to {self.max_gaussians} Gaussians")
        # GT mode fixes the coordinate embedding while features follow the flow state.
        coordinates = [gs["means"] for gs in batch_flow_gs]
        if self.coordinate_mode == "gt":
            if batch_reference_means is None or len(batch_reference_means) != len(counts):
                raise ValueError("GT coordinate mode requires one reference means tensor per scene")
            for reference, gs, count in zip(batch_reference_means, batch_flow_gs, counts):
                if not torch.is_tensor(reference) or reference.shape != (count, 3):
                    raise ValueError(f"GT reference means must have shape ({count}, 3) per scene")
                if reference.device != gs["means"].device:
                    raise ValueError("GT reference means must be on the same device as the flow state")
            coordinates = batch_reference_means
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
        model_input = {
            "feat": torch.cat(packed),
            "offset": offset,
            "timesteps": timesteps,
        }
        if self.coordinate_mode != "morton_rank":
            model_input["coord"] = torch.cat(coordinates)
        velocity = self.backbone(model_input)

        widths = list(self.feature_channels.values())
        outputs = []
        for scene_velocity, gs in zip(velocity.split(counts), batch_flow_gs):
            attributes = {}
            for key, value in zip(self.output_features, scene_velocity.split(widths, dim=-1)):
                attributes[key] = value.reshape(len(scene_velocity), -1, 3) if key == "features_rest" else value
            if self.sh_degree == 0:
                attributes["features_rest"] = torch.zeros_like(gs["features_rest"])
            outputs.append(attributes)
        return outputs

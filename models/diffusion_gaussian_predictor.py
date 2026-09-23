"""Gaussian velocity heads around a time-conditioned DiPT backbone."""

from collections import OrderedDict
from typing import List

import gin
import torch
import torch.nn as nn

from .diffusion_gaussian_transformer import DiffusionGaussianTransformer
from utils.fourier_features import build_fourier_feature_encoders

gin.external_configurable(torch.nn.Identity)
gin.external_configurable(torch.nn.Tanh)
gin.external_configurable(torch.nn.Sigmoid)


FEATURE2CHANNEL = {
    "means": 3,
    "features_dc": 3,
    "features_rest": 3,
    "opacities": 1,
    "scales": 3,
    "quats": 4,
}
ALL_FEATURES = list(FEATURE2CHANNEL)


@gin.configurable
class DiffusionGaussianPredictor(nn.Module):
    """Predict standardized Gaussian interpolant velocities for each scene."""

    def __init__(
        self,
        sh_degree,
        input_features,
        input_feat_to_mlp,
        output_features,
        output_head_nlayer,
        output_head_type,
        output_head_width,
        grid_resolution,
        resume_ckpt,
        zeroinit,
        res_feature_activation,
        quat_residual_mode="add",
        fourier_input_features=(),
        fourier_num_frequencies=None,
        fourier_include_raw=True,
        fourier_log_sampling=True,
        fourier_max_frequency_log2=None,
    ):
        super().__init__()
        if quat_residual_mode != "add":
            raise ValueError("Standardized Gaussian velocities require additive quaternion updates")
        self.sh_degree = sh_degree
        self.feature2channel = dict(FEATURE2CHANNEL)
        self.feature2channel["features_rest"] = 3 * ((sh_degree + 1) ** 2 - 1)
        self.input_features = list(input_features)
        self.output_features = list(output_features)
        self.input_feat_to_mlp = input_feat_to_mlp
        self.grid_resolution = grid_resolution
        self.resume_ckpt = resume_ckpt
        self.quat_residual_mode = quat_residual_mode
        self.backbone_type = "DIPT"
        self.res_feature_activation = nn.ModuleDict(res_feature_activation)

        self.fourier_encoders, input_feature_channels = build_fourier_feature_encoders(
            self.feature2channel, self.input_features, fourier_input_features,
            fourier_num_frequencies, fourier_include_raw, fourier_log_sampling,
            fourier_max_frequency_log2,
        )
        in_channels = sum(input_feature_channels[feature] for feature in self.input_features)
        self.gs_features_dim = in_channels
        self.backbone = DiffusionGaussianTransformer(in_channels=in_channels)
        head_input_dim = self.backbone.output_dim + (in_channels if input_feat_to_mlp else 0)

        self.features_outputhead = nn.ModuleDict()
        for feature in self.output_features:
            if output_head_type != "mlp-relu":
                raise NotImplementedError(f"Unsupported output head: {output_head_type}")
            layers = []
            for index in range(output_head_nlayer - 1):
                layers.extend([
                    nn.Linear(head_input_dim if index == 0 else output_head_width, output_head_width),
                    nn.ReLU(),
                ])
            layers.append(nn.Linear(output_head_width if output_head_nlayer > 1 else head_input_dim,
                                    self.feature2channel[feature]))
            self.features_outputhead[feature] = nn.Sequential(*layers)

        if zeroinit:
            for head in self.features_outputhead.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def _input_feature_tensor(self, gs, key):
        value = gs[key]
        if key == "features_rest":
            value = value.reshape(value.shape[0], -1)
        if key in self.fourier_encoders:
            value = self.fourier_encoders[key](value)
        return value

    def apply_feature_update(self, feature, value, update, step_scale=1.0):
        del feature
        return value + step_scale * update

    def forward(
        self,
        batch_flow_gs: List[dict],
        batch_scene_idx: List[int],
        t=None,
        batch_reference_means=None,
        **kwargs,
    ):
        del batch_scene_idx, kwargs
        if t is None:
            raise ValueError("DiffusionGaussianPredictor.forward requires `t`")

        device = batch_flow_gs[0]["means"].device
        counts = [gs["means"].shape[0] for gs in batch_flow_gs]
        offset = torch.tensor(counts, device=device, dtype=torch.long).cumsum(0)
        feat = torch.cat([
            torch.cat([self._input_feature_tensor(gs, key) for key in self.input_features], dim=1)
            for gs in batch_flow_gs
        ], dim=0)

        if batch_reference_means is None:
            batch_reference_means = [gs["means"] for gs in batch_flow_gs]
        if len(batch_reference_means) != len(batch_flow_gs):
            raise ValueError("Expected one reference means tensor per scene")
        coord = torch.cat(batch_reference_means, dim=0)
        if coord.shape != (feat.shape[0], 3):
            raise ValueError(f"Expected reference means shape ({feat.shape[0]}, 3), got {tuple(coord.shape)}")

        timesteps = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(len(batch_flow_gs))
        if timesteps.numel() != len(batch_flow_gs):
            raise ValueError(f"Expected one timestep per scene, got {timesteps.numel()}")

        # Reference means fix spatial serialization while feat carries the noisy state.
        model_input = {
            "coord": coord,
            "grid_coord": torch.floor(coord * self.grid_resolution).int(),
            "grid_size": coord.new_full((3,), 1.0 / self.grid_resolution),
            "offset": offset,
            "feat": feat,
            "timesteps": timesteps,
        }
        point = self.backbone(model_input)
        hidden = point.feat
        if self.input_feat_to_mlp:
            hidden = torch.cat([hidden, feat], dim=1)

        output = OrderedDict()
        for feature in self.output_features:
            velocity = self.res_feature_activation[feature](self.features_outputhead[feature](hidden))
            if feature == "features_rest":
                velocity = velocity.reshape(velocity.shape[0], -1, 3)
            output[feature] = velocity

        out_batch_flow = []
        left = 0
        for right, in_gs in zip(offset.tolist(), batch_flow_gs):
            out_gs = {feature: output[feature][left:right] for feature in self.output_features}
            for feature in ALL_FEATURES:
                if self.sh_degree == 0 and feature == "features_rest":
                    continue
                if feature not in out_gs and feature in in_gs:
                    out_gs[feature] = torch.zeros_like(in_gs[feature])
            out_batch_flow.append(out_gs)
            left = right
        return out_batch_flow

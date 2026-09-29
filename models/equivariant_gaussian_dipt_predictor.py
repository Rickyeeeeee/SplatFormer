"""Gaussian attribute velocities around EDiPT; no training/rollout integration.

Color coefficients are ordinary features with no SH basis conversion.
use_features_rest=False removes higher-order SH from both inputs and heads. Attribute
velocities use caller coordinates; position and quaternion derivatives use the
explicit physical geometry. Quaternion derivatives need compatible tangent
training targets, not the existing additive quaternion endpoint differences.
"""
import math

import gin
import torch
from torch import nn

from .equivariant_gaussian_dipt import EquivariantGaussianDiPT
from utils.data_augmentation import quaternion_multiply, quaternion_to_rotation_matrix


@gin.configurable
class EquivariantGaussianDiPTPredictor(nn.Module):
    def __init__(self, sh_degree=1, input_feat_to_mlp=True, output_head_width=64,
                 grid_resolution=1536, zeroinit=True, use_features_rest=True):
        super().__init__()
        if not isinstance(sh_degree, int) or sh_degree < 0:
            raise ValueError("sh_degree must be a nonnegative integer")
        if not math.isfinite(grid_resolution) or grid_resolution <= 0:
            raise ValueError("grid_resolution must be finite and positive")
        self.sh_degree = sh_degree
        self.use_features_rest = use_features_rest
        self.grid_resolution = grid_resolution
        self.input_feat_to_mlp = input_feat_to_mlp
        self.backbone_type = 'EDIPT'
        self.feature_channels = {'scales': 3, 'opacities': 1, 'features_dc': 3}
        if sh_degree and use_features_rest:
            self.feature_channels['features_rest'] = 3 * ((sh_degree + 1) ** 2 - 1)
        self.input_features = list(self.feature_channels)
        self.output_features = ['means', 'quats', *self.input_features]
        self.gs_features_dim = sum(self.feature_channels.values())
        self.backbone = EquivariantGaussianDiPT(in_channels=self.gs_features_dim)
        head_width = self.backbone.output_dim + (self.gs_features_dim if input_feat_to_mlp else 0)
        widths = {'means': 3, 'quats': 3, **self.feature_channels}
        self.features_outputhead = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(head_width, output_head_width), nn.ReLU(), nn.Linear(output_head_width, width))
            for key, width in widths.items()
        })
        if zeroinit:
            for head in self.features_outputhead.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, batch_flow_gs, batch_scene_idx=None, t=None, *, batch_geometry, batch_reference_means=None):
        del batch_scene_idx
        if t is None:
            raise ValueError("EDiPT requires time t")
        scenes = len(batch_flow_gs)
        if not scenes or len(batch_geometry) != scenes:
            raise ValueError("Expected one geometry dictionary per nonempty scene batch")
        if batch_reference_means is None:
            batch_reference_means = [geometry['means'] for geometry in batch_geometry]
        if len(batch_reference_means) != scenes:
            raise ValueError("Expected one reference means tensor per scene")

        features, positions, quaternions, grids, counts = [], [], [], [], []
        for gs, geometry, reference in zip(batch_flow_gs, batch_geometry, batch_reference_means):
            means, quats = geometry['means'], geometry['quats']
            count = len(means)
            if count == 0 or means.shape != (count, 3) or quats.shape != (count, 4) or reference.shape != (count, 3):
                raise ValueError("Expected nonempty means/reference [N,3] and quats [N,4]")
            # Geometry stays physical and FP32 even when feature operations use AMP.
            means, quats = means.float(), quats.float()
            norms = quats.norm(dim=-1, keepdim=True)
            if not torch.isfinite(quats).all() or not torch.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Geometry quaternions must be finite and nonzero")
            if not torch.isfinite(means).all() or not torch.isfinite(reference).all():
                raise ValueError("Current and reference positions must be finite")
            values = []
            for key, width in self.feature_channels.items():
                value = gs[key]
                shape = (count, width // 3, 3) if key == 'features_rest' else (count, width)
                if value.shape != shape:
                    raise ValueError(f"Expected {key} shape {shape}, got {tuple(value.shape)}")
                values.append(value.reshape(count, width))
            features.append(torch.cat(values, dim=-1))
            positions.append(means)
            quaternions.append(quats / norms)
            counts.append(count)
            # Validate before integer conversion so large coordinates cannot wrap.
            grid = torch.floor(reference.float() * self.grid_resolution)
            grid = grid - grid.amin(dim=0, keepdim=True).clamp(max=0)
            if not torch.isfinite(grid).all() or (grid >= 65536).any():
                raise ValueError("Reference grid exceeds Pointcept's 16-bit coordinate range")
            grids.append(grid.int())

        feat, coord, quats = torch.cat(features), torch.cat(positions), torch.cat(quaternions)
        with torch.autocast(device_type=coord.device.type, enabled=False):
            rotations = quaternion_to_rotation_matrix(quats)
        timesteps = torch.as_tensor(t, device=coord.device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(scenes)
        if timesteps.numel() != scenes or not torch.isfinite(timesteps).all():
            raise ValueError("Expected a finite scalar time or one time per scene")
        point = self.backbone({
            'feat': feat, 'coord': coord, 'rotations': rotations,
            'grid_coord': torch.cat(grids),
            'offset': torch.tensor(counts, device=coord.device, dtype=torch.long).cumsum(0),
            'timesteps': timesteps,
        })
        hidden = torch.cat([point.feat, feat], dim=-1) if self.input_feat_to_mlp else point.feat
        output = {key: head(hidden) for key, head in self.features_outputhead.items()}
        with torch.autocast(device_type=coord.device.type, enabled=False):
            output['means'] = torch.einsum('nij,nj->ni', rotations, output['means'].float())
            angular = output['quats'].float()
            output['quats'] = 0.5 * quaternion_multiply(quats, torch.cat([torch.zeros_like(angular[:, :1]), angular], dim=-1))
        if self.sh_degree and self.use_features_rest:
            output['features_rest'] = output['features_rest'].reshape(len(feat), -1, 3)
        splits = {key: value.split(counts) for key, value in output.items()}
        return [{key: pieces[index] for key, pieces in splits.items()} for index in range(scenes)]

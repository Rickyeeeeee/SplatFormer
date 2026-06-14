from collections import OrderedDict
from typing import List

import gin
import torch
import torch.nn as nn

from models.pointtransformer_v3_flow import PointTransformerV3FlowModel


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


def timestep_embedding(timesteps, dim, max_period=10000):
    timesteps = timesteps.float().view(-1)
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=timesteps.device, dtype=torch.float32))
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


@gin.configurable
@gin.configurable("GSFlowFeaturePredictor")
class GSFlowModel(nn.Module):
    def __init__(
        self,
        sh_degree,
        input_features,
        input_feat_to_mlp,
        output_features,
        output_head_nlayer,
        output_head_type,
        output_head_width,
        res_feature_activation,
        max_scale_normalized,
        grid_resolution,
        resume_ckpt,
        input_embed_to_mlp,
        zeroinit,
    ):
        super(GSFlowModel, self).__init__()
        self.sh_degree = sh_degree
        sh_dim = (sh_degree + 1) ** 2 - 1
        FEATURE2CHANNEL["features_rest"] = sh_dim * 3
        self.input_features = input_features
        self.input_feat_to_mlp = input_feat_to_mlp
        in_channels = sum([FEATURE2CHANNEL[feature] for feature in input_features])
        self.gs_features_dim = in_channels
        self.output_features = output_features
        if max_scale_normalized <= 0:
            print("Setting max_scale_normalized <0, turning off scale clamping")
        self.max_scale_normalized = max_scale_normalized
        self.backbone_type = "PT_FLOW"
        self.grid_resolution = grid_resolution
        self.resume_ckpt = resume_ckpt
        self.res_feature_activation = res_feature_activation
        self.input_embed_to_mlp = input_embed_to_mlp

        self.backbone = PointTransformerV3FlowModel(in_channels=in_channels)
        head_input_dim = self.backbone.output_dim
        if self.input_feat_to_mlp:
            head_input_dim += in_channels

        self.features_outputhead = nn.ModuleDict()
        for feature in output_features:
            if output_head_type == "mlp-relu":
                module_list = nn.ModuleList()
                for i in range(output_head_nlayer - 1):
                    module_list.extend(
                        [
                            nn.Linear(head_input_dim if i == 0 else output_head_width, output_head_width),
                            nn.ReLU(),
                        ]
                    )
                output_dim = FEATURE2CHANNEL[feature]
                module_list.append(
                    nn.Linear(output_head_width if output_head_nlayer > 1 else head_input_dim, output_dim)
                )
                self.features_outputhead[feature] = nn.Sequential(*module_list)
            else:
                raise NotImplementedError
        if zeroinit:
            for module in self.features_outputhead.values():
                module[-1].weight.data.zero_()
                module[-1].bias.data.zero_()

    def _prepare_t_emb(self, batch_t_emb, batch_timestep, timestep, batch_size, device):
        T_dim = getattr(self.backbone, "T_dim", -1)
        if T_dim == -1:
            return None
        if batch_t_emb is not None:
            if isinstance(batch_t_emb, (list, tuple)):
                batch_t_emb = torch.stack(
                    [torch.as_tensor(t, device=device, dtype=torch.float32).view(-1) for t in batch_t_emb],
                    dim=0,
                )
            else:
                batch_t_emb = torch.as_tensor(batch_t_emb, device=device, dtype=torch.float32)
            if batch_t_emb.dim() == 1:
                batch_t_emb = batch_t_emb.unsqueeze(0)
            if batch_t_emb.shape[0] == 1 and batch_size > 1:
                batch_t_emb = batch_t_emb.expand(batch_size, -1)
            if batch_t_emb.shape[0] != batch_size:
                raise ValueError(f"batch_t_emb batch dimension {batch_t_emb.shape[0]} != batch size {batch_size}")
            if batch_t_emb.shape[-1] != T_dim:
                raise ValueError(f"batch_t_emb dim {batch_t_emb.shape[-1]} != T_dim {T_dim}")
            return batch_t_emb

        if batch_timestep is None:
            batch_timestep = timestep
        if batch_timestep is None:
            return None
        batch_timestep = torch.as_tensor(batch_timestep, device=device, dtype=torch.float32).view(-1)
        if batch_timestep.numel() == 1 and batch_size > 1:
            batch_timestep = batch_timestep.expand(batch_size)
        if batch_timestep.numel() != batch_size:
            raise ValueError(f"batch_timestep length {batch_timestep.numel()} != batch size {batch_size}")
        return timestep_embedding(batch_timestep, T_dim)

    def forward(
        self,
        batch_normalized_gs: List,
        batch_scene_idx=None,
        batch_t_emb=None,
        batch_timestep=None,
        timestep=None,
        **kwargs,
    ):
        del batch_scene_idx, kwargs
        device = batch_normalized_gs[0]["means"].device
        counts = [gs["means"].shape[0] for gs in batch_normalized_gs]
        offset = torch.tensor(counts).cumsum(0)
        feat = []

        for gs in batch_normalized_gs:
            feat_list = []
            for key in self.input_features:
                if key == "means":
                    feat_list.append(gs[key])
                elif key == "features_rest":
                    feat_list.append(gs[key].view(gs[key].shape[0], -1))
                else:
                    feat_list.append(gs[key])
            feat.append(torch.cat(feat_list, dim=1))
        feat = torch.cat(feat, dim=0)

        model_input = {
            "coord": torch.cat([gs["means"] for gs in batch_normalized_gs], dim=0),
            "grid_size": torch.ones([3], device=device) * 1.0 / self.grid_resolution,
            "offset": offset.to(device),
            "feat": feat,
        }
        model_input["grid_coord"] = torch.floor(model_input["coord"] * self.grid_resolution).int()
        t_emb = self._prepare_t_emb(
            batch_t_emb=batch_t_emb,
            batch_timestep=batch_timestep,
            timestep=timestep,
            batch_size=len(batch_normalized_gs),
            device=device,
        )
        if t_emb is not None:
            model_input["t_emb"] = t_emb

        y = self.backbone(model_input)["feat"]
        if self.input_feat_to_mlp:
            y = torch.cat([y, feat], dim=1)

        output = OrderedDict()
        for feature in self.output_features:
            feature_o = self.features_outputhead[feature](y)
            feature_delta = self.res_feature_activation[feature](feature_o[:, : FEATURE2CHANNEL[feature]])
            if feature == "features_rest":
                feature_delta = feature_delta.view(feature_delta.shape[0], -1, 3)
            output[feature] = feature_delta

        out_batch_residuals = []
        left = 0
        for right in offset.tolist():
            out_residual = {}
            for feature in self.output_features:
                out_residual[feature] = output[feature][left:right]
            out_batch_residuals.append(out_residual)
            left = right

        assert len(out_batch_residuals) == 1, "Now only support batch size 1"
        return out_batch_residuals


GSFlowFeaturePredictor = GSFlowModel

from collections import OrderedDict
from typing import List

import gin
import torch
import torch.nn as nn

from .pointtransformer_v3_flow import PointTransformerV3FlowModel

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
ALL_FEATURES = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]


def _identity_quat_like(quats):
    identity = torch.zeros_like(quats)
    identity[..., 0] = 1.0
    return identity


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@gin.configurable
class GSFlowPredictor(nn.Module):
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
    ):
        super().__init__()
        if quat_residual_mode not in ["add", "mul"]:
            raise ValueError(
                f"Unsupported quat_residual_mode={quat_residual_mode}; expected 'add' or 'mul'"
            )
        self.sh_degree = sh_degree
        sh_dim = (sh_degree + 1) ** 2 - 1
        self.feature2channel = dict(FEATURE2CHANNEL)
        self.feature2channel["features_rest"] = sh_dim * 3
        self.input_features = list(input_features)
        self.output_features = list(output_features)
        self.input_feat_to_mlp = input_feat_to_mlp
        self.grid_resolution = grid_resolution
        self.resume_ckpt = resume_ckpt
        self.res_feature_activation = res_feature_activation
        self.quat_residual_mode = quat_residual_mode
        self.backbone_type = "PT_FLOW"

        in_channels = sum(self.feature2channel[feature] for feature in self.input_features)
        self.gs_features_dim = in_channels
        self.backbone = PointTransformerV3FlowModel(in_channels=in_channels)

        head_input_dim = self.backbone.output_dim
        if self.input_feat_to_mlp:
            head_input_dim += in_channels

        self.features_outputhead = nn.ModuleDict()
        for feature in self.output_features:
            if output_head_type != "mlp-relu":
                raise NotImplementedError(f"Unsupported output head: {output_head_type}")
            module_list = nn.ModuleList()
            for layer_idx in range(output_head_nlayer - 1):
                module_list.extend(
                    [
                        nn.Linear(head_input_dim if layer_idx == 0 else output_head_width, output_head_width),
                        nn.ReLU(),
                    ]
                )
            output_dim = self.feature2channel[feature]
            module_list.append(
                nn.Linear(output_head_width if output_head_nlayer > 1 else head_input_dim, output_dim)
            )
            self.features_outputhead[feature] = nn.Sequential(*module_list)

        if zeroinit:
            for module in self.features_outputhead.values():
                module[-1].weight.data.zero_()
                module[-1].bias.data.zero_()

    def _feature_tensor(self, gs, key):
        value = gs[key]
        if key == "features_rest":
            return value.view(value.shape[0], -1)
        return value

    def _time_embedding(self, t, batch_size, device):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch_size)
        if t.numel() != batch_size:
            raise ValueError(f"Expected one t per scene or one scalar, got {t.numel()} for batch {batch_size}")
        return torch.stack([t, torch.sin(t), torch.cos(t)], dim=-1)

    def apply_feature_update(self, feature, value, update, step_scale=1.0):
        if torch.is_tensor(step_scale):
            scale = step_scale.to(device=update.device, dtype=update.dtype)
        else:
            scale = float(step_scale)
        if feature == "quats" and self.quat_residual_mode == "mul":
            delta_quat = torch.nn.functional.normalize(
                _identity_quat_like(update) + scale * update, dim=-1
            )
            input_quat = torch.nn.functional.normalize(value, dim=-1)
            return torch.nn.functional.normalize(_quat_multiply(delta_quat, input_quat), dim=-1)
        return value + scale * update

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
            raise ValueError("GSFlowPredictor.forward requires a time tensor `t`")

        device = batch_flow_gs[0]["means"].device
        counts = [gs["means"].shape[0] for gs in batch_flow_gs]
        offset = torch.tensor(counts, device=device, dtype=torch.long).cumsum(0)
        feat = []
        for gs in batch_flow_gs:
            feat_list = [self._feature_tensor(gs, key) for key in self.input_features]
            feat.append(torch.cat(feat_list, dim=1))
        feat = torch.cat(feat, dim=0)

        if batch_reference_means is None:
            batch_reference_means = [gs["means"] for gs in batch_flow_gs]
        coord = torch.cat(batch_reference_means, dim=0)
        model_input = {
            "coord": coord,
            "grid_size": torch.ones([3], device=device) * 1.0 / self.grid_resolution,
            "offset": offset,
            "feat": feat,
            "t_emb": self._time_embedding(t, len(batch_flow_gs), device),
        }
        model_input["grid_coord"] = torch.floor(coord * self.grid_resolution).int()

        y = self.backbone(model_input)
        if isinstance(y, dict):
            y = y["feat"]
        else:
            y = y.feat
        if self.input_feat_to_mlp:
            y = torch.cat([y, feat], dim=1)

        output = OrderedDict()
        for feature in self.output_features:
            feature_o = self.features_outputhead[feature](y)
            feature_o = self.res_feature_activation[feature](feature_o)
            if feature == "features_rest":
                feature_o = feature_o.view(feature_o.shape[0], -1, 3)
            output[feature] = feature_o

        out_batch_flow = []
        left = 0
        for right, in_gs in zip(offset.tolist(), batch_flow_gs):
            out_gs = {}
            for feature in self.output_features:
                out_gs[feature] = output[feature][left:right]
            for key in ALL_FEATURES:
                if self.sh_degree == 0 and key == "features_rest":
                    continue
                if key not in out_gs and key in in_gs:
                    out_gs[key] = torch.zeros_like(in_gs[key])
            out_batch_flow.append(out_gs)
            left = right

        return out_batch_flow

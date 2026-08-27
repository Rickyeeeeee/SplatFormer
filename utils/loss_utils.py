import json
from dataclasses import dataclass

import gin
import torch
import torch.nn.functional as F

SUPPORTED_GS_KEYS = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]


@dataclass(frozen=True)
class GaussianAttributeLossConfig:
    """Fixed options for the all-attribute Gaussian loss."""

    loss_weights: dict
    quat_direct_mse: bool
    means_loss_reduction: str

def load_gs_statistics_normalizers(
    stats_path,
    target_resolution,
    loss_features,
    target_gs,
    min_std=1e-6,
    alignment=None,
    resolutions=None,
):
    """Load channel-wise standard deviations from legacy or gsplat reports."""
    with open(stats_path, "r") as stats_file:
        report = json.load(stats_file)

    if "factors" in report:
        factor = int(target_resolution)
        if resolutions is not None:
            max_resolution = max(int(resolution) for resolution in resolutions)
            if max_resolution % int(target_resolution) != 0:
                raise ValueError(
                    f"Cannot map resolution {target_resolution} to a legacy factor"
                )
            factor = max_resolution // int(target_resolution)
        report_key = f"df-{factor}"
        factors = report.get("factors", {})
        if report_key not in factors:
            raise ValueError(
                f"GS statistics report does not contain factor '{report_key}'"
            )
        parameters = factors[report_key].get("parameters", {})
        key_map = {key: key for key in loss_features}
        schema = "legacy"
    elif "aggregate" in report:
        if alignment != "fit_lr_to_hr":
            raise ValueError(
                "Aggregate gsplat output statistics require alignment='fit_lr_to_hr'"
            )
        report_key = "aggregate.output"
        parameters = report.get("aggregate", {}).get("output", {})
        key_map = {
            "features_dc": "sh0",
            "features_rest": "shN",
            **{
                key: key
                for key in loss_features
                if key not in ("features_dc", "features_rest")
            },
        }
        schema = "gsplat"
    else:
        raise ValueError("Unrecognized GS statistics report schema")

    normalizers = {}
    selected_statistics = {}
    for key in loss_features:
        report_parameter = key_map[key]
        parameter_stats = parameters.get(report_parameter)
        if parameter_stats is None:
            raise ValueError(
                f"GS statistics report {report_key} is missing parameter "
                f"'{report_parameter}' for '{key}'"
            )
        expected_shape = tuple(target_gs[key].shape[1:])
        std_values = parameter_stats.get("std")
        if std_values is None:
            raise ValueError(
                f"GS statistics report {report_key} is missing std for '{report_parameter}'"
            )
        std = torch.as_tensor(
            std_values,
            dtype=target_gs[key].dtype,
            device=target_gs[key].device,
        )
        if schema == "legacy":
            reported_shape = tuple(parameter_stats.get("channel_shape") or [])
            if reported_shape != expected_shape:
                raise ValueError(
                    f"GS statistics shape mismatch for '{key}': "
                    f"report {reported_shape} vs target {expected_shape}"
                )
        else:
            if key == "features_dc" and std.ndim == 2 and std.shape[0] == 1:
                std = std.squeeze(0)
            if key == "opacities" and std.ndim == 0 and expected_shape == (1,):
                std = std.unsqueeze(0)
        if tuple(std.shape) != expected_shape:
            raise ValueError(
                f"GS statistics std shape mismatch for '{key}': "
                f"report {tuple(std.shape)} vs target {expected_shape}"
            )
        if not torch.isfinite(std).all() or torch.any(std < 0):
            raise ValueError(
                f"GS statistics std for '{key}' must be finite and nonnegative"
            )
        normalizers[key] = std.clamp_min(float(min_std))
        selected_statistics[key] = {
            **parameter_stats,
            "report_parameter": report_parameter,
            "report_group": report_key,
        }
    return normalizers, selected_statistics


def feature_loss_value(
    key,
    pred: torch.Tensor,
    target: torch.Tensor,
    post_activate_loss=False,
    quat_direct_mse=False,
    means_loss_reduction="mean",
    component_normalizer=None,
):
    if component_normalizer is not None:
        if post_activate_loss:
            raise ValueError("Component-normalized loss does not support post_activate_loss")
        normalized_error = (pred - target) / component_normalizer
        if key == "means":
            normalized_error = normalized_error.abs()
            if means_loss_reduction == "mean":
                return normalized_error.mean()
            if means_loss_reduction == "sum":
                return normalized_error.sum()
        return normalized_error.square().mean()
    if key == "means":
        # error = (pred - target).square()
        error = (pred - target).abs()
        if means_loss_reduction == "mean":
            return error.mean()
        if means_loss_reduction == "sum":
            return error.sum()


    if key == "opacities" and post_activate_loss:
        return F.mse_loss(torch.sigmoid(pred), torch.sigmoid(target))

    if key == "quats":
        if quat_direct_mse or not post_activate_loss:
            return F.mse_loss(pred, target)
        pred_quat = F.normalize(pred, dim=-1)
        target_quat = F.normalize(target, dim=-1)
        cosine_sq = (pred_quat * target_quat).sum(dim=-1).square().clamp(max=1.0)
        return (1.0 - cosine_sq).mean()

    return F.mse_loss(pred, target)


@gin.configurable
def gaussian_attribute_loss_config(
    loss_weights=None,
    quat_direct_mse=False,
    means_loss_reduction="mean",
):
    """Return fixed options for the all-attribute Gaussian loss."""
    resolved_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
    if loss_weights is not None:
        resolved_weights.update(loss_weights)
    return GaussianAttributeLossConfig(
        loss_weights=resolved_weights,
        quat_direct_mse=quat_direct_mse,
        means_loss_reduction=means_loss_reduction,
    )


def compute_gaussian_attribute_loss(
    out_gs,
    target_gs,
    loss_weights,
    post_activate_loss=False,
    quat_direct_mse=False,
    means_loss_reduction="mean",
    component_normalizers=None,
):
    """Compute and weight every Gaussian attribute loss."""
    losses = {}
    weighted_losses = {}
    total_loss = 0.0
    for key in SUPPORTED_GS_KEYS:
        pred = out_gs[key]
        target = target_gs[key].to(device=pred.device, dtype=pred.dtype)
        normalizer = None
        if component_normalizers is not None:
            normalizer = component_normalizers.get(key)
            if normalizer is not None:
                normalizer = normalizer.to(device=pred.device, dtype=pred.dtype)
        loss = feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=post_activate_loss,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=means_loss_reduction,
            component_normalizer=normalizer,
        )
        weighted_loss = loss_weights[key] * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = total_loss + weighted_loss
    return total_loss, losses, weighted_losses


class lpips_loss_fn():
    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def __call__(self, x, y):
        # x  B,H,W,C [0,1]
        # y  B,H,W,C [0,1]
        loss = self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)#.mean()
        return loss

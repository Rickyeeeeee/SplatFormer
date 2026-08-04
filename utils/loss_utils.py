import json

import torch
import torch.nn.functional as F

SUPPORTED_GS_KEYS = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]
def load_gs_statistics_normalizers(stats_path, target_factor, loss_features, target_gs, min_std=1e-6):
    """Load channel-wise GS standard deviations for loss normalization."""
    with open(stats_path, "r") as stats_file:
        report = json.load(stats_file)
    factor_key = f"df-{target_factor}"
    factors = report.get("factors", {})
    if factor_key not in factors:
        raise ValueError(f"GS statistics report does not contain factor '{factor_key}'")
    parameters = factors[factor_key].get("parameters", {})
    normalizers = {}
    selected_statistics = {}
    for key in loss_features:
        parameter_stats = parameters.get(key)
        if parameter_stats is None:
            raise ValueError(f"GS statistics report {factor_key} is missing parameter '{key}'")
        expected_shape = tuple(target_gs[key].shape[1:])
        reported_shape = tuple(parameter_stats.get("channel_shape") or [])
        if reported_shape != expected_shape:
            raise ValueError(f"GS statistics shape mismatch for '{key}': report {reported_shape} vs target {expected_shape}")
        std_values = parameter_stats.get("std")
        if std_values is None:
            raise ValueError(f"GS statistics report {factor_key} is missing std for '{key}'")
        std = torch.as_tensor(std_values, dtype=target_gs[key].dtype, device=target_gs[key].device)
        if tuple(std.shape) != expected_shape:
            raise ValueError(f"GS statistics std shape mismatch for '{key}': report {tuple(std.shape)} vs target {expected_shape}")
        if not torch.isfinite(std).all() or torch.any(std < 0):
            raise ValueError(f"GS statistics std for '{key}' must be finite and nonnegative")
        normalizers[key] = std.clamp_min(float(min_std))
        selected_statistics[key] = parameter_stats
    return normalizers, selected_statistics


def unique_preserve_order(values):
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def parse_loss_features(
    raw_loss_features,
    model,
    target_gs,
    supported_keys=SUPPORTED_GS_KEYS,
    no_features_message="No Gaussian attributes selected for MSE loss",
):
    if raw_loss_features is None or raw_loss_features.strip() == "":
        loss_features = list(getattr(model, "output_features", []))
    else:
        loss_features = [feature.strip() for feature in raw_loss_features.split(",") if feature.strip()]

    loss_features = unique_preserve_order(loss_features)
    if len(loss_features) == 0:
        raise ValueError(no_features_message)

    unsupported = [feature for feature in loss_features if feature not in supported_keys]
    if unsupported:
        raise ValueError(
            f"Unsupported loss feature(s): {unsupported}. "
            f"Supported features are: {list(supported_keys)}"
        )

    missing = [feature for feature in loss_features if feature not in target_gs]
    if missing:
        raise ValueError(f"Selected loss feature(s) missing from target GS: {missing}")

    return loss_features


def fixed_attribute_keys(loss_features, target_gs, supported_keys=SUPPORTED_GS_KEYS):
    loss_set = set(loss_features)
    return [key for key in supported_keys if key in target_gs and key not in loss_set]


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
            raise ValueError(
                f"Unsupported means_loss_reduction={means_loss_reduction}; expected 'mean' or 'sum'"
            )
        return normalized_error.square().mean()
    if key == "means":
        # error = (pred - target).square()
        error = (pred - target).abs()
        if means_loss_reduction == "mean":
            return error.mean()
        if means_loss_reduction == "sum":
            return error.sum()
        raise ValueError(
            f"Unsupported means_loss_reduction={means_loss_reduction}; expected 'mean' or 'sum'"
        )



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


def feature_mse_loss(
    out_gs,
    target_gs,
    loss_features,
    loss_weights=None,
    post_activate_loss=False,
    quat_direct_mse=False,
    means_loss_reduction="mean",
    component_normalizers=None,
    output_label="model output",
    total_loss_error="No MSE losses were computed",
    grad_error=(
        "Selected loss features do not receive gradients. "
        "Make sure FeaturePredictor.output_features includes at least one selected loss feature."
    ),
):
    losses = {}
    weighted_losses = {}
    total_loss = None
    if loss_weights is None:
        loss_weights = {key: 1.0 for key in loss_features}

    for key in loss_features:
        if key not in out_gs:
            raise ValueError(f"Selected loss feature '{key}' missing from {output_label}")
        if key not in target_gs:
            raise ValueError(f"Selected loss feature '{key}' missing from target GS")
        if out_gs[key].shape != target_gs[key].shape:
            raise ValueError(
                f"Shape mismatch for loss feature '{key}': "
                f"output {tuple(out_gs[key].shape)} vs target {tuple(target_gs[key].shape)}"
            )
        pred = out_gs[key]
        component_normalizer = None
        if component_normalizers is not None and key in component_normalizers:
            component_normalizer = component_normalizers[key].to(device=pred.device, dtype=pred.dtype)
            if component_normalizer.shape != pred.shape[1:]:
                raise ValueError(
                    f"Normalizer shape mismatch for loss feature '{key}': "
                    f"normalizer {tuple(component_normalizer.shape)} vs feature {tuple(pred.shape[1:])}"
                )
        target = target_gs[key].to(device=pred.device, dtype=pred.dtype)
        loss = feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=post_activate_loss,
            quat_direct_mse=quat_direct_mse,
            component_normalizer=component_normalizer,
            means_loss_reduction=means_loss_reduction,
        )
        weighted_loss = float(loss_weights.get(key, 1.0)) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

    if total_loss is None:
        raise ValueError(total_loss_error)
    if not total_loss.requires_grad:
        raise ValueError(grad_error)
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

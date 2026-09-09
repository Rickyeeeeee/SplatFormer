import json

import torch

from utils import gs_utils
from utils.loss_utils import SUPPORTED_GS_KEYS


FLOW_EPSILON = 1e-6


STATISTIC_KEYS = {
    "means": "means",
    "features_dc": "sh0",
    "features_rest": "shN",
    "opacities": "opacities",
    "scales": "scales",
    "quats": "quats",
}


def delta_statistic_tensor(
    delta_statistics, key, statistic, device=None, dtype=None
):
    """Load a stored delta statistic in the runtime Gaussian feature shape."""
    value = torch.as_tensor(
        delta_statistics[STATISTIC_KEYS[key]][statistic],
        device=device,
        dtype=dtype,
    )
    # Statistics retain SH0's coefficient axis, while runtime features_dc is [N, 3].
    if key == "features_dc" and value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def delta_velocity_variances(delta_statistics, variance_floor, device=None, dtype=None):
    """Convert stored standard deviations to raw and floored variances."""
    raw_variances = {}
    effective_variances = {}
    for key in SUPPORTED_GS_KEYS:
        std = delta_statistic_tensor(
            delta_statistics, key, "std", device=device, dtype=dtype
        )
        raw_variances[key] = std.square()
        effective_variances[key] = raw_variances[key].clamp_min(
            float(variance_floor)
        )
    return raw_variances, effective_variances


def load_aggregate_velocity_variances(stats_path, variance_floor, device=None, dtype=None):
    """Load fixed global velocity variances from aggregate delta statistics."""
    with open(stats_path) as statistics_file:
        delta_statistics = json.load(statistics_file)["aggregate"]["delta"]

    return delta_velocity_variances(delta_statistics, variance_floor, device=device, dtype=dtype)


def loss_mix_weights(t, schedule):
    if schedule == "linear":
        return 1.0 - t, t
    if schedule == "free-range-gs":
        render_weight = 50.0 * torch.clamp(t / 0.9, max=1.0).pow(5)
        return torch.ones_like(t), render_weight
    if schedule == "fm-only":
        return torch.ones_like(t), torch.zeros_like(t)
    raise ValueError(
        f"Unsupported loss_mixing.schedule={schedule!r}; "
        "expected one of ('linear', 'free-range-gs', 'fm-only')"
    )


def sample_stochastic_interpolant(
    source_flow_gs, target_flow_gs, t, noise_scale
):
    missing = [
        key
        for key in SUPPORTED_GS_KEYS
        if key not in source_flow_gs or key not in target_flow_gs
    ]
    if missing:
        raise ValueError(
            f"Flow interpolation requires every Gaussian attribute; missing {missing}"
        )

    alpha = t.view(1, 1)
    gamma_base = torch.sqrt(
        torch.clamp(2.0 * t * (1.0 - t), min=FLOW_EPSILON)
    )
    gamma = (float(noise_scale) * gamma_base).view(1, 1)
    gamma_dot = (
        float(noise_scale) * (1.0 - 2.0 * t) / gamma_base
    ).view(1, 1)

    query_flow_gs = {}
    flow_noise = {}
    for key in SUPPORTED_GS_KEYS:
        source_value = source_flow_gs[key]
        target_value = target_flow_gs[key]
        noise = (
            torch.randn_like(source_value)
            if float(noise_scale) > 0.0
            else torch.zeros_like(source_value)
        )
        flow_noise[key] = noise
        query_flow_gs[key] = (
            (1.0 - alpha) * source_value
            + alpha * target_value
            + gamma * noise
        )
    return query_flow_gs, flow_noise, gamma, gamma_dot


def compute_matching_velocity_variances(
    source_flow_gs, target_flow_gs, variance_floor
):
    raw_variances = {}
    effective_variances = {}
    for key in SUPPORTED_GS_KEYS:
        if key not in source_flow_gs or key not in target_flow_gs:
            raise ValueError(
                f"Cannot compute velocity variance for missing feature {key!r}"
            )
        if source_flow_gs[key].shape != target_flow_gs[key].shape:
            raise ValueError(
                f"Velocity variance shape mismatch for {key!r}: "
                f"source {tuple(source_flow_gs[key].shape)} vs "
                f"target {tuple(target_flow_gs[key].shape)}"
            )
        velocity = (
            target_flow_gs[key] - source_flow_gs[key]
        ).detach().float()
        variance = velocity.var(dim=0, unbiased=False)
        raw_variances[key] = variance
        effective_variances[key] = variance.clamp_min(
            float(variance_floor)
        )
    return raw_variances, effective_variances


def compute_variance_normalized_velocity_loss(
    pred_vel,
    source_flow_gs,
    target_flow_gs,
    flow_noise,
    gamma_dot,
    velocity_variances,
    loss_weights,
):
    missing = [key for key in SUPPORTED_GS_KEYS if key not in pred_vel]
    if missing:
        raise ValueError(
            f"GSFlowPredictor must output every Gaussian attribute; missing {missing}"
        )

    losses = {}
    weighted_losses = {}
    total_loss = None
    for key in SUPPORTED_GS_KEYS:
        pred = pred_vel[key].float()
        target = (target_flow_gs[key] - source_flow_gs[key]).to(
            device=pred.device, dtype=pred.dtype
        )
        noise = flow_noise[key].to(device=pred.device, dtype=pred.dtype)
        target = target + gamma_dot.to(
            device=pred.device, dtype=pred.dtype
        ) * noise
        variance = velocity_variances[key].to(
            device=pred.device, dtype=pred.dtype
        )
        loss = ((pred - target).square() / variance).mean()
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    if not total_loss.requires_grad:
        raise ValueError(
            "GSFlowPredictor outputs do not receive gradients for the "
            "all-attribute velocity loss"
        )
    return total_loss, losses, weighted_losses


def apply_feature_update(model, feature, value, update, step_scale=1.0):
    if hasattr(model, "apply_feature_update"):
        return model.apply_feature_update(
            feature, value, update, step_scale
        )
    if torch.is_tensor(step_scale):
        step_scale = step_scale.to(
            device=update.device, dtype=update.dtype
        )
    else:
        step_scale = float(step_scale)
    return value + step_scale * update


def predict_x1_from_velocity(
    model,
    source_flow_gs,
    query_flow_gs,
    pred_vel,
    flow_noise,
    gamma,
    gamma_dot,
    t,
    source_anchored=True,
):
    missing = [key for key in SUPPORTED_GS_KEYS if key not in pred_vel]
    if missing:
        raise ValueError(
            f"GSFlowPredictor must output every Gaussian attribute; missing {missing}"
        )

    one_minus_t = (1.0 - t).view(1, 1)
    predicted_target = {}
    for key in SUPPORTED_GS_KEYS:
        query_value = query_flow_gs[key]
        noise = flow_noise[key]
        clean_query = query_value - gamma * noise
        base_value = (
            source_flow_gs[key] if source_anchored else clean_query
        )
        step_scale = 1.0 if source_anchored else one_minus_t
        clean_velocity = pred_vel[key] - gamma_dot * noise
        predicted_target[key] = apply_feature_update(
            model,
            key,
            base_value,
            clean_velocity,
            step_scale,
        )
    return predicted_target


def sample_flow_model(model, source_flow_gs, scene_idx, flow_steps):
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    model.eval()
    state = gs_utils.clone_gaussians(source_flow_gs)
    device = state["means"].device
    with torch.no_grad():
        for step in range(flow_steps):
            time = torch.full(
                (1,),
                float(step) / float(flow_steps),
                device=device,
            )
            predicted_velocity = model(
                batch_flow_gs=[state],
                batch_scene_idx=[scene_idx],
                batch_reference_means=[source_flow_gs["means"]],
                t=time,
            )[0]
            missing = [
                key
                for key in SUPPORTED_GS_KEYS
                if key not in predicted_velocity
            ]
            if missing:
                raise ValueError(
                    f"GSFlowPredictor must output every Gaussian attribute; "
                    f"missing {missing}"
                )
            state = {
                key: apply_feature_update(
                    model,
                    key,
                    state[key],
                    predicted_velocity[key],
                    1.0 / float(flow_steps),
                )
                for key in SUPPORTED_GS_KEYS
            }
    return gs_utils.clone_gaussians(state)

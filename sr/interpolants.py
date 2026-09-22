"""Euclidean losses and sampling for standardized Gaussian attributes."""
import torch

from utils.gs_normalization import STATISTIC_KEYS


def attribute_mse(prediction, target):
    losses = {key: (prediction[key].float() - target[key].float()).square().mean() for key in STATISTIC_KEYS}
    return sum(losses.values()), losses, dict(losses)


@torch.no_grad()
def sample_flow_model(model, source, scene_idx, flow_steps, standardizer):
    """Accept and return scene-frame attributes; integrate only standardized states."""
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    model.eval()
    state = standardizer.encode(source)
    for step in range(flow_steps):
        time = torch.full((1,), step / flow_steps, device=source["means"].device)
        velocity = model(batch_flow_gs=[state], batch_scene_idx=[scene_idx],
                         batch_reference_means=[source["means"]], t=time)[0]
        state = {key: state[key] + velocity[key] / flow_steps for key in STATISTIC_KEYS}
    return standardizer.decode(state)

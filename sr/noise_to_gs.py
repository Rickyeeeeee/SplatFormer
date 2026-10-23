"""Unconditional Gaussian flow matching and count-only sampling."""

import torch

from utils.gs_normalization import STATISTIC_KEYS


def sample_noise(count, standardizer, device, generator=None):
    if count < 1:
        raise ValueError("Gaussian count must be positive")
    return {
        key: torch.randn((count,) + tuple(mean.shape), device=device, dtype=torch.float32, generator=generator)
        for key, mean in standardizer.means.items()
    }


def construct_path(target, time, noise=None):
    """Return the one-sided linear query and its exact standardized velocity."""
    if noise is None:
        noise = {key: torch.randn_like(target[key]) for key in STATISTIC_KEYS}
    time = torch.as_tensor(time, device=target["means"].device, dtype=target["means"].dtype).reshape(())
    state = {key: (1 - time) * noise[key] + time * target[key] for key in STATISTIC_KEYS}
    velocity = {key: target[key] - noise[key] for key in STATISTIC_KEYS}
    return state, velocity


def predict_velocity(model, states, scene_indices, time, standardizer):
    """Use only the evolving state to supply spatial coordinates to point backbones."""
    references = None
    if model.backbone_type != "DIT":
        mean = standardizer.means["means"].to(states[0]["means"])
        std = standardizer.stds["means"].to(states[0]["means"])
        references = [state["means"].float() * std + mean for state in states]
    return model(batch_flow_gs=states, batch_scene_idx=scene_indices,
                 batch_reference_means=references, t=time)


def rollout(model, noise, scene_idx, steps, standardizer, t_eps=1e-4):
    if steps < 1 or not 0 < t_eps < 0.5:
        raise ValueError("Expected positive flow steps and 0 < t_eps < 0.5")
    state = noise
    for step in range(steps):
        time = state["means"].new_tensor([min(max(step / steps, t_eps), 1 - t_eps)])
        velocity = predict_velocity(model, [state], [scene_idx], time, standardizer)[0]
        state = {key: state[key] + velocity[key] / steps for key in STATISTIC_KEYS}
    return state


@torch.no_grad()
def sample_flow_model(model, count, scene_idx, steps, standardizer, noise_seed=0, t_eps=1e-4):
    """Generate a scene from its count and a deterministic noise seed."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(noise_seed) + int(scene_idx))
    noise = sample_noise(count, standardizer, device, generator)
    was_training = model.training
    model.eval()
    try:
        endpoint = rollout(model, noise, scene_idx, steps, standardizer, t_eps)
    finally:
        model.train(was_training)
    return standardizer.decode(endpoint)

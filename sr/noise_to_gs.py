"""Gaussian flow matching with optional GT-coordinate conditioning."""

import gin
import torch

from utils.gs_normalization import STATISTIC_KEYS


@gin.configurable
def flow_matching(gs_statistics_path, flow_steps=10, flow_t_eps=1e-4, eval_noise_seed=0,
                  normalization_variance_floor=1e-8):
    if flow_steps < 1 or not 0 < flow_t_eps < 0.5:
        raise ValueError("Expected positive flow_steps and 0 < flow_t_eps < 0.5")
    return {"gs_statistics_path": gs_statistics_path, "flow_steps": flow_steps,
            "flow_t_eps": flow_t_eps, "eval_noise_seed": eval_noise_seed,
            "normalization_variance_floor": normalization_variance_floor}


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


def predict_velocity(model, states, scene_indices, time, standardizer, reference_means=None):
    """Supply physical state coordinates to point backbones or fixed DiT references."""
    references = reference_means
    if model.backbone_type != "DIT":
        mean = standardizer.means["means"].to(states[0]["means"])
        std = standardizer.stds["means"].to(states[0]["means"])
        references = [state["means"].float() * std + mean for state in states]
    return model(batch_flow_gs=states, batch_scene_idx=scene_indices,
                 batch_reference_means=references, t=time)


def rollout(model, noise, scene_idx, steps, standardizer, t_eps=1e-4, reference_means=None, on_state=None):
    if steps < 1 or not 0 < t_eps < 0.5:
        raise ValueError("Expected positive flow steps and 0 < t_eps < 0.5")
    state = noise
    references = [reference_means] if reference_means is not None else None
    if on_state is not None:
        on_state(0, state)
    for step in range(steps):
        time = state["means"].new_tensor([min(max(step / steps, t_eps), 1 - t_eps)])
        velocity = predict_velocity(model, [state], [scene_idx], time, standardizer, references)[0]
        state = {key: state[key] + velocity[key] / steps for key in STATISTIC_KEYS}
        if on_state is not None:
            on_state(step + 1, state)
    return state


@torch.no_grad()
def sample_flow_model(model, count, scene_idx, steps, standardizer, noise_seed=0, t_eps=1e-4,
                      reference_means=None):
    """Generate a scene from noise, optionally conditioned on fixed GT coordinates."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(noise_seed) + int(scene_idx))
    noise = sample_noise(count, standardizer, device, generator)
    was_training = model.training
    model.eval()
    try:
        endpoint = rollout(model, noise, scene_idx, steps, standardizer, t_eps, reference_means)
    finally:
        model.train(was_training)
    return standardizer.decode(endpoint)

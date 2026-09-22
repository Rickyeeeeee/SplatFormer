"""Normalized Gaussian paths, Euclidean losses, and differentiable Euler flows."""
import math

import torch

from utils.gs_normalization import STATISTIC_KEYS


MODES = ("linear", "latent", "encoding_decoding", "one_sided")


def validate_settings(mode, steps, t_eps=1e-4):
    if mode not in MODES:
        raise ValueError(f"Unknown interpolant_type {mode!r}; expected {MODES}")
    if steps <= 0 or int(steps) != steps:
        raise ValueError("flow steps must be a positive integer")
    if mode == "encoding_decoding" and (steps < 4 or steps % 2):
        raise ValueError("encoding_decoding requires an even step count of at least four")
    if not 0 < t_eps < .5:
        raise ValueError("flow_t_eps must be between zero and 0.5")


def reference_means(mode, t, source_means, target_means):
    use_target = mode == "one_sided" or (mode == "encoding_decoding" and float(t) >= .5)
    reference = target_means if use_target else source_means
    if reference is None:
        raise ValueError("The selected interpolant stage is missing its reference means")
    return reference


def construct_path(source, target, t, mode, source_means, target_means, noise_scale=1.0, noise=None):
    """Construct one scene's state and exact pathwise velocity in standardized space."""
    if mode not in MODES:
        raise ValueError(f"Unknown interpolant_type {mode!r}")
    time = torch.as_tensor(t, device=target["means"].device, dtype=target["means"].dtype).reshape(())
    if not 0 <= float(time) <= 1:
        raise ValueError("Interpolant time must be in [0, 1]")
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("flow_noise_std must be finite and nonnegative")
    if noise is None:
        noise = {key: torch.zeros_like(value) if mode == "linear" else torch.randn_like(value)
                 for key, value in target.items()}
    gamma = time.new_zeros(())
    gamma_dot = time.new_zeros(())
    if mode in ("latent", "encoding_decoding"):
        base = (2 * time * (1 - time)).sqrt()
        gamma = noise_scale * base
        # The square-root path has no finite endpoint derivative; training excludes endpoints.
        if float(base) == 0 and noise_scale > 0:
            raise ValueError("Noisy path velocities require 0 < t < 1")
        gamma_dot = noise_scale * (1 - 2 * time) / base if noise_scale > 0 else gamma_dot
    state, velocity = {}, {}
    for key in STATISTIC_KEYS:
        if mode == "linear":
            state[key] = (1 - time) * source[key] + time * target[key]
            velocity[key] = target[key] - source[key]
        elif mode == "latent":
            state[key] = (1 - time) * source[key] + time * target[key] + gamma * noise[key]
            velocity[key] = target[key] - source[key] + gamma_dot * noise[key]
        elif mode == "encoding_decoding":
            endpoint = source[key] if float(time) < .5 else target[key]
            # Make the shared midpoint exact despite floating-point trigonometric error.
            coefficient = time.new_zeros(()) if float(time) == .5 else torch.cos(math.pi * time).square()
            derivative = time.new_zeros(()) if float(time) == .5 else -math.pi * torch.sin(2 * math.pi * time)
            state[key] = coefficient * endpoint + gamma * noise[key]
            velocity[key] = derivative * endpoint + gamma_dot * noise[key]
        else:
            state[key] = (1 - time) * noise[key] + time * target[key]
            velocity[key] = target[key] - noise[key]
    return {"query": state, "velocity": velocity, "noise": noise, "gamma": gamma, "gamma_dot": gamma_dot,
            "reference_means": reference_means(mode, time, source_means, target_means)}


def attribute_mse(prediction, target):
    losses = {key: (prediction[key].float() - target[key].float()).square().mean() for key in STATISTIC_KEYS}
    return sum(losses.values()), losses, dict(losses)


def rollout(model, source, scene_idx, steps, mode, source_means, target_means, initial_noise=None, t_eps=1e-4):
    """Return a standardized endpoint without changing model mode or disabling gradients."""
    validate_settings(mode, steps, t_eps)
    if mode == "one_sided":
        if initial_noise is None:
            raise ValueError("one_sided rollout requires initial noise")
        state = {key: value for key, value in initial_noise.items()}
    else:
        state = {key: value for key, value in source.items()}
    for step in range(steps):
        # Even step counts place the encoding/decoding switch exactly on a grid boundary.
        stage_time = step / steps
        time = state["means"].new_tensor([min(max(stage_time, t_eps), 1 - t_eps)])
        reference = reference_means(mode, stage_time, source_means, target_means)
        velocity = model(batch_flow_gs=[state], batch_scene_idx=[scene_idx],
                         batch_reference_means=[reference], t=time)[0]
        state = {key: state[key] + velocity[key] / steps for key in STATISTIC_KEYS}
    return state


@torch.no_grad()
def sample_flow_model(model, source, scene_idx, flow_steps, standardizer, mode="linear",
                      target_means=None, t_eps=1e-4, noise_seed=0):
    """Accept/return scene-frame attributes; one-sided sampling accepts source=None."""
    validate_settings(mode, flow_steps, t_eps)
    model.eval()
    noise = None
    if mode == "one_sided":
        if target_means is None:
            raise ValueError("one_sided sampling requires target reference means")
        generator = torch.Generator(device=target_means.device).manual_seed(int(noise_seed) + int(scene_idx))
        noise = {key: torch.randn((len(target_means),) + tuple(mean.shape), device=target_means.device,
                                 dtype=torch.float32, generator=generator) for key, mean in standardizer.means.items()}
        state, source_means = None, None
    else:
        state, source_means = standardizer.encode(source), source["means"]
    endpoint = rollout(model, state, scene_idx, flow_steps, mode, source_means, target_means, noise, t_eps)
    return standardizer.decode(endpoint)

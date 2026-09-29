"""EDiPT mixed-coordinate paths and manifold integration.

Non-rotation attributes use the original standardized Euclidean paths. Rotation
noise is a Gaussian rotation vector around world identity, not a rotation-
invariant prior. SH coefficients remain ordinary standardized attributes.
"""
import math

import torch
from torch.nn import functional as F

from sr.interpolants import MODES, reference_means, validate_settings
from utils.data_augmentation import quaternion_multiply, quaternion_to_rotation_matrix
from utils.rotation_flow import (integrate_rotation, quaternion_derivative_to_body,
                                 relative_rotation, rotation_exp)


def evaluation_steps(mode):
    return [4, 6, 10] if mode == 'encoding_decoding' else [1, 5, 10]


def sample_noise(reference, generator=None):
    return {'euclidean': {key: torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
                          for key, value in reference.items() if key != 'quats'},
            'rotation': torch.randn(reference['quats'].shape[:-1] + (3,), dtype=torch.float32,
                                    device=reference['quats'].device, generator=generator)}


def seeded_noise_like(reference, seed):
    generator = torch.Generator(device=reference['means'].device).manual_seed(int(seed))
    return sample_noise(reference, generator)


def rotation_path(source, target, time, mode, noise, noise_std):
    """Return a continuous quaternion lift and analytic body angular velocity."""
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('rotation_noise_std must be finite and nonnegative')
    target = F.normalize(target, dim=-1)
    prior = rotation_exp(noise_std * noise)
    if mode == 'one_sided':
        delta = relative_rotation(prior, target)
        return quaternion_multiply(prior, rotation_exp(time * delta)), delta
    source = F.normalize(source, dim=-1)
    delta = relative_rotation(source, target)
    base = quaternion_multiply(source, rotation_exp(time * delta))
    if mode == 'linear':
        return base, delta
    if mode == 'latent':
        if noise_std == 0:
            return base, delta
        if not 0 < float(time) < 1:
            raise ValueError('Noisy latent rotation velocities require 0 < t < 1')
        scale = (2 * time * (1 - time)).sqrt()
        perturbation = rotation_exp(noise_std * scale * noise)
        rotation = quaternion_to_rotation_matrix(perturbation)
        angular = torch.einsum('nji,nj->ni', rotation, delta) + noise_std * (1 - 2 * time) / scale * noise
        return quaternion_multiply(base, perturbation), angular
    # Choose the midpoint lift by following the first shortest geodesic.
    to_prior = relative_rotation(source, prior)
    midpoint = quaternion_multiply(source, rotation_exp(to_prior))
    if float(time) == .5:
        return midpoint, torch.zeros_like(noise)
    if float(time) < .5:
        coefficient = torch.sin(math.pi * time).square()
        angular = math.pi * torch.sin(2 * math.pi * time) * to_prior
        return quaternion_multiply(source, rotation_exp(coefficient * to_prior)), angular
    to_target = relative_rotation(midpoint, target)
    coefficient = torch.cos(math.pi * time).square()
    angular = -math.pi * torch.sin(2 * math.pi * time) * to_target
    return quaternion_multiply(midpoint, rotation_exp(coefficient * to_target)), angular


def construct_path(source, target, t, mode, source_means, target_means, noise_scale=1.0,
                   noise=None, rotation_noise_std=.3):
    if mode not in MODES:
        raise ValueError(f'Unknown interpolant mode {mode!r}')
    time = torch.as_tensor(t, device=target['means'].device, dtype=torch.float32).reshape(())
    if not 0 <= float(time) <= 1 or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError('Expected time in [0,1] and finite nonnegative flow_noise_std')
    if noise is None:
        noise = sample_noise(target) if mode != 'linear' else {
            'euclidean': {k: torch.zeros_like(v) for k, v in target.items() if k != 'quats'},
            'rotation': torch.zeros_like(target['quats'][..., 1:])}
    gamma = time.new_zeros(())
    gamma_dot = time.new_zeros(())
    if mode in ('latent', 'encoding_decoding') and noise_scale > 0:
        if not 0 < float(time) < 1:
            raise ValueError('Noisy Euclidean paths require 0 < t < 1')
        base = (2 * time * (1 - time)).sqrt()
        gamma, gamma_dot = noise_scale * base, noise_scale * (1 - 2 * time) / base
    state, velocity = {}, {}
    for key in target:
        if key == 'quats':
            continue
        z = noise['euclidean'][key]
        if mode in ('linear', 'latent'):
            state[key] = (1 - time) * source[key] + time * target[key] + gamma * z
            velocity[key] = target[key] - source[key] + gamma_dot * z
        elif mode == 'encoding_decoding':
            endpoint = source[key] if float(time) < .5 else target[key]
            coefficient = time.new_zeros(()) if float(time) == .5 else torch.cos(math.pi * time).square()
            derivative = time.new_zeros(()) if float(time) == .5 else -math.pi * torch.sin(2 * math.pi * time)
            state[key] = coefficient * endpoint + gamma * z
            velocity[key] = derivative * endpoint + gamma_dot * z
        else:
            state[key] = (1 - time) * z + time * target[key]
            velocity[key] = target[key] - z
    with torch.autocast(device_type=time.device.type, enabled=False):
        state['quats'], velocity['quats'] = rotation_path(
            None if source is None else source['quats'].float(), target['quats'].float(), time,
            mode, noise['rotation'].float(), rotation_noise_std)
    return {'query': state, 'velocity': velocity, 'noise': noise, 'gamma': gamma, 'gamma_dot': gamma_dot,
            'reference_means': reference_means(mode, time, source_means, target_means)}


def predict_velocity(model, batch_flow_gs, batch_scene_idx, batch_reference_means, t, standardizer):
    """Adapt physical EDiPT derivatives into standardized and angular velocities."""
    if standardizer.quaternion_representation != 'unit_unstandardized':
        raise ValueError('EDiPT requires unit_unstandardized quaternion state')
    geometry = []
    for state in batch_flow_gs:
        means = state['means'].float()
        geometry.append({'means': means * standardizer.stds['means'].to(means) + standardizer.means['means'].to(means),
                         'quats': F.normalize(state['quats'].float(), dim=-1)})
    predictions = model(batch_flow_gs=batch_flow_gs, batch_scene_idx=batch_scene_idx,
                        batch_geometry=geometry, batch_reference_means=batch_reference_means, t=t)
    result = []
    with torch.autocast(device_type=batch_flow_gs[0]['means'].device.type, enabled=False):
        for prediction, geom in zip(predictions, geometry):
            converted = dict(prediction)
            converted['means'] = prediction['means'].float() / standardizer.stds['means'].to(prediction['means'].float())
            converted['quats'] = quaternion_derivative_to_body(geom['quats'], prediction['quats'].float())
            result.append(converted)
    return result


def attribute_mse(prediction, target, loss_type='velocity', rotation_loss_weight=1.0):
    losses = {key: (prediction[key].float() - value.float()).square().mean()
              for key, value in target.items() if key != 'quats' and value.numel()}
    with torch.autocast(device_type=prediction['quats'].device.type, enabled=False):
        if loss_type == 'velocity':
            losses['quats'] = (prediction['quats'].float() - target['quats'].float()).square().mean()
        elif loss_type == 'x1':
            losses['quats'] = relative_rotation(prediction['quats'].float(), target['quats'].float()).square().sum(-1).mean()
        else:
            raise ValueError(f'Unknown loss_type {loss_type!r}')
    weighted = {**losses, 'quats': rotation_loss_weight * losses['quats']}
    return sum(weighted.values()), losses, weighted


def rollout(model, source, scene_idx, steps, mode, source_means, target_means, initial_noise=None,
            t_eps=1e-4, *, standardizer, rotation_noise_std=.3):
    validate_settings(mode, steps, t_eps)
    if mode == 'one_sided':
        if initial_noise is None:
            raise ValueError('one_sided rollout requires initial noise')
        state = {**initial_noise['euclidean'], 'quats': rotation_exp(rotation_noise_std * initial_noise['rotation'])}
    else:
        state = {**source, 'quats': F.normalize(source['quats'].float(), dim=-1)}
    for step in range(steps):
        stage_time = step / steps
        time = state['means'].new_tensor([min(max(stage_time, t_eps), 1 - t_eps)])
        reference = reference_means(mode, stage_time, source_means, target_means)
        velocity = predict_velocity(model, [state], [scene_idx], [reference], time, standardizer)[0]
        with torch.autocast(device_type=state['means'].device.type, enabled=False):
            next_state = {key: value + velocity[key] / steps for key, value in state.items() if key != 'quats' and value.numel()}
            next_state['quats'] = integrate_rotation(state['quats'].float(), velocity['quats'].float(), 1 / steps)
        state = next_state
    return state


@torch.no_grad()
def sample_flow_model(model, source, scene_idx, flow_steps, standardizer, mode='linear',
                      target_means=None, t_eps=1e-4, noise_seed=0, rotation_noise_std=.3):
    validate_settings(mode, flow_steps, t_eps)
    model.eval()
    noise = None
    if mode == 'one_sided':
        if target_means is None:
            raise ValueError('one_sided sampling requires target reference means')
        template = {key: torch.empty((len(target_means),) + tuple(mean.shape), device=target_means.device)
                    for key, mean in standardizer.means.items()}
        noise = seeded_noise_like(template, int(noise_seed) + int(scene_idx))
        state, source_means = None, None
    else:
        source, _ = standardizer.prepare_endpoints(source)
        state, source_means = standardizer.encode(source), source['means']
    endpoint = rollout(model, state, scene_idx, flow_steps, mode, source_means, target_means, noise, t_eps,
                       standardizer=standardizer, rotation_noise_std=rotation_noise_std)
    return standardizer.decode(endpoint)

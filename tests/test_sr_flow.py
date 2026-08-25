import torch

from sr import flow
from utils.loss_utils import SUPPORTED_GS_KEYS


FEATURE_SHAPES = {
    "means": (3,),
    "features_dc": (3,),
    "features_rest": (1, 3),
    "opacities": (1,),
    "scales": (3,),
    "quats": (4,),
}


def gaussian_batch(value, requires_grad=False):
    return {
        key: torch.full(
            (2, *shape),
            float(value),
            requires_grad=requires_grad,
        )
        for key, shape in FEATURE_SHAPES.items()
    }


def test_stochastic_interpolant_without_noise_is_linear():
    source = gaussian_batch(0.0)
    target = gaussian_batch(2.0)
    time = torch.tensor([0.25])

    query, noise, gamma, gamma_dot = flow.sample_stochastic_interpolant(
        source, target, time, noise_scale=0.0
    )

    for key in SUPPORTED_GS_KEYS:
        torch.testing.assert_close(
            query[key], torch.full_like(query[key], 0.5)
        )
        torch.testing.assert_close(noise[key], torch.zeros_like(noise[key]))
    torch.testing.assert_close(gamma, torch.zeros_like(gamma))
    torch.testing.assert_close(gamma_dot, torch.zeros_like(gamma_dot))


def test_velocity_loss_uses_every_attribute_and_weights():
    source = gaussian_batch(0.0)
    target = gaussian_batch(1.0)
    prediction = gaussian_batch(0.0, requires_grad=True)
    noise = gaussian_batch(0.0)
    variances = {
        key: torch.ones(shape) for key, shape in FEATURE_SHAPES.items()
    }
    weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
    weights["means"] = 2.0

    total, losses, weighted = (
        flow.compute_variance_normalized_velocity_loss(
            pred_vel=prediction,
            source_flow_gs=source,
            target_flow_gs=target,
            flow_noise=noise,
            gamma_dot=torch.zeros((1, 1)),
            velocity_variances=variances,
            loss_weights=weights,
        )
    )

    assert list(losses) == SUPPORTED_GS_KEYS
    assert list(weighted) == SUPPORTED_GS_KEYS
    torch.testing.assert_close(total, torch.tensor(7.0))


def test_velocity_variance_floor_and_loss_mixing():
    source = gaussian_batch(0.0)
    target = gaussian_batch(1.0)
    raw, effective = flow.compute_matching_velocity_variances(
        source, target, variance_floor=0.25
    )
    for key in SUPPORTED_GS_KEYS:
        torch.testing.assert_close(raw[key], torch.zeros_like(raw[key]))
        torch.testing.assert_close(
            effective[key], torch.full_like(effective[key], 0.25)
        )

    time = torch.tensor([0.2])
    fm_weight, render_weight = flow.loss_mix_weights(time, "linear")
    torch.testing.assert_close(fm_weight, torch.tensor([0.8]))
    torch.testing.assert_close(render_weight, torch.tensor([0.2]))
    fm_weight, render_weight = flow.loss_mix_weights(time, "fm-only")
    torch.testing.assert_close(fm_weight, torch.ones_like(time))
    torch.testing.assert_close(render_weight, torch.zeros_like(time))


class ConstantVelocityModel:
    def eval(self):
        return self

    def __call__(self, batch_flow_gs, **kwargs):
        del kwargs
        return [
            {
                key: torch.ones_like(value)
                for key, value in batch_flow_gs[0].items()
            }
        ]


def test_multistep_flow_sampling_updates_every_attribute():
    source = gaussian_batch(0.0)
    sampled = flow.sample_flow_model(
        ConstantVelocityModel(), source, scene_idx=0, flow_steps=2
    )
    for key in SUPPORTED_GS_KEYS:
        torch.testing.assert_close(sampled[key], torch.ones_like(sampled[key]))

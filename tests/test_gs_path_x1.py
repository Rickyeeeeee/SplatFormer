from unittest import mock

import torch

from gs_path import GSPath


def _state():
    return {
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "scales": torch.tensor([[0.5, 0.5, 0.5]]),
    }


def test_x1_extrapolation_formulas_and_unpredicted_fields():
    x_t = _state()
    prediction = {"means": torch.tensor([[2.0, 4.0, 6.0]])}

    residual = GSPath.extrapolate_x1(
        x_t, prediction, torch.tensor([0.25]), ["means"], "residual"
    )
    velocity = GSPath.extrapolate_x1(
        x_t,
        prediction,
        torch.tensor([0.25]),
        ["means"],
        "velocity_extrapolation",
    )

    torch.testing.assert_close(residual["means"], x_t["means"] + prediction["means"])
    torch.testing.assert_close(
        velocity["means"], x_t["means"] + 0.75 * prediction["means"]
    )
    assert residual["scales"] is x_t["scales"]
    assert velocity["scales"] is x_t["scales"]

    residual_t0 = GSPath.extrapolate_x1(
        x_t, prediction, torch.tensor([0.0]), ["means"], "residual"
    )
    velocity_t0 = GSPath.extrapolate_x1(
        x_t,
        prediction,
        torch.tensor([0.0]),
        ["means"],
        "velocity_extrapolation",
    )
    torch.testing.assert_close(residual_t0["means"], velocity_t0["means"])


def test_exact_x1_has_zero_point_mse():
    target = _state()
    pred = {key: value.clone() for key, value in target.items()}
    loss, feature_losses = GSPath.x1_point_mse_loss(
        pred, target, ["means", "scales"]
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert set(feature_losses) == {"means", "scales"}


def test_optimized_target_is_retrained_for_each_call():
    path = GSPath(flow_num_timesteps=1, flow_segment_steps=1)
    calls = []

    def fake_optimize(input_gs, images, cameras, flow_keys, optim_steps, log_prefix):
        calls.append(input_gs["means"].detach().clone())
        target = GSPath.detach_gs(input_gs)
        target["means"] = target["means"] + float(len(calls))
        return target, torch.tensor(float(len(calls)))

    path.optimize_gs = fake_optimize
    source = _state()

    x0_first, x1_first = path.optimize_target_discrete(
        source, [], {}, ["means"], 1
    )
    x0_second, x1_second = path.optimize_target_discrete(
        source, [], {}, ["means"], 1
    )

    assert len(calls) == 2
    torch.testing.assert_close(x0_first["means"], source["means"])
    torch.testing.assert_close(x0_second["means"], source["means"])
    assert not torch.equal(x1_first["means"], x1_second["means"])


def test_render_l1_backpropagates_through_extrapolated_x1():
    raw_prediction = torch.ones((1, 3), requires_grad=True)
    x0 = _state()
    pred_x1 = GSPath.extrapolate_x1(
        x0,
        {"means": raw_prediction},
        torch.tensor([0.0]),
        ["means"],
        "residual",
    )

    def fake_rasterizer(gs, cameras):
        image = gs["means"].sum().expand(1, 1, 3)
        return [image], None

    with mock.patch(
        "gs_path.gs_utils.rasterize_gaussians_to_multiimgs",
        side_effect=fake_rasterizer,
    ):
        loss = GSPath.render_l1_loss(
            pred_x1,
            [torch.zeros((1, 1, 3))],
            {},
        )
        loss.backward()

    assert raw_prediction.grad is not None
    assert torch.count_nonzero(raw_prediction.grad).item() == raw_prediction.numel()


if __name__ == "__main__":
    test_x1_extrapolation_formulas_and_unpredicted_fields()
    test_exact_x1_has_zero_point_mse()
    test_optimized_target_is_retrained_for_each_call()
    test_render_l1_backpropagates_through_extrapolated_x1()
    print("x1 tests passed")

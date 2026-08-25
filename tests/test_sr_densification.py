import math

import torch

from sr import densification


class IdentityScaler:
    scale_ = torch.tensor(1.0)
    trans_ = torch.zeros(3)


FEATURE_SHAPES = {
    "means": (3,),
    "features_dc": (3,),
    "features_rest": (1, 3),
    "opacities": (1,),
    "scales": (3,),
    "quats": (4,),
}


def gaussian_batch(count, offset=0.0):
    gaussians = {
        key: torch.full((count, *shape), float(offset))
        for key, shape in FEATURE_SHAPES.items()
    }
    gaussians["means"] = (
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
        + offset
    )
    gaussians["quats"][:, 0] = 1.0
    return gaussians


def test_midpoint_noop_returns_independent_gaussians():
    source = gaussian_batch(2)
    output = densification.midpoint_interpolate_gaussians(source, 2)

    for key in source:
        torch.testing.assert_close(output[key], source[key])
        assert output[key] is not source[key]


def test_nearest_alignment_maps_each_target_to_a_source():
    source_means = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    target_means = torch.tensor([[1.8, 0.0, 0.0], [0.1, 0.0, 0.0]])

    indices = densification.align_nearest_target_to_source(
        source_means, target_means
    )

    assert indices.tolist() == [1, 0]


def test_3dgs_initialization_resets_scale_quaternion_and_opacity():
    gaussians = gaussian_batch(2)
    gaussians["means"] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    initialized = densification.initialize_3dgs_attributes(gaussians)

    torch.testing.assert_close(
        initialized["scales"], torch.zeros((2, 3))
    )
    torch.testing.assert_close(
        initialized["quats"],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    )
    torch.testing.assert_close(
        initialized["opacities"],
        torch.full((2, 1), math.log(0.1 / 0.9)),
    )


def test_target_cardinality_construction_preserves_stage_schema():
    source = gaussian_batch(2, offset=0.0)
    target = gaussian_batch(2, offset=10.0)
    source_entry = {"gs_params": source, "scaler": IdentityScaler()}
    target_entry = {"gs_params": target, "scaler": IdentityScaler()}

    output, stages = densification.build_densified_input(
        input_factor_dict=source_entry,
        target_factor_dict=target_entry,
        alignment="none",
        attribute_init="aligned",
        emd_eps=0.01,
        emd_iters=100,
        device=torch.device("cpu"),
        return_stages=True,
    )

    assert output["means"].shape[0] == target["means"].shape[0]
    assert set(output) == set(FEATURE_SHAPES)
    assert list(stages) == [
        "00_low_res_gs.ply",
        "01_interpolated_high_res_gs.ply",
        "02_gt_high_res_gs.ply",
        "03_input_high_res_gs.ply",
    ]

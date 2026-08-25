import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.loss_utils import SUPPORTED_GS_KEYS, feature_loss_value


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SHAPES = {
    "means": (3,),
    "features_dc": (3,),
    "features_rest": (1, 3),
    "opacities": (1,),
    "scales": (3,),
    "quats": (4,),
}


def load_local_loss_function(filename):
    path = REPOSITORY_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_all_feature_mse_loss"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "torch": torch,
        "SUPPORTED_GS_KEYS": SUPPORTED_GS_KEYS,
        "MEANS_LOSS_REDUCTION": "mean",
        "feature_loss_value": feature_loss_value,
        "loss_utils": SimpleNamespace(feature_loss_value=feature_loss_value),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["compute_all_feature_mse_loss"]


def gaussian_pair():
    predictions = {}
    targets = {}
    for index, (key, shape) in enumerate(FEATURE_SHAPES.items(), start=1):
        predictions[key] = torch.full(
            (2, *shape),
            0.1 * index,
            requires_grad=True,
        )
        targets[key] = torch.zeros((2, *shape))
    return predictions, targets


@pytest.mark.parametrize(
    "filename", ["overfit-sr-mse.py", "overfit-sr-gsfm.py"]
)
def test_overfit_loss_always_includes_and_weights_all_attributes(filename):
    loss_function = load_local_loss_function(filename)
    predictions, targets = gaussian_pair()
    weights = {
        key: float(index)
        for index, key in enumerate(SUPPORTED_GS_KEYS, start=1)
    }
    arguments = {
        "out_gs": predictions,
        "target_gs": targets,
        "loss_weights": weights,
        "quat_direct_mse": True,
    }
    if filename == "overfit-sr-mse.py":
        arguments.update(
            post_activate_loss=False,
            component_normalizers=None,
        )

    total, losses, weighted = loss_function(**arguments)

    assert list(losses) == SUPPORTED_GS_KEYS
    assert list(weighted) == SUPPORTED_GS_KEYS
    for key in SUPPORTED_GS_KEYS:
        torch.testing.assert_close(weighted[key], weights[key] * losses[key])
    torch.testing.assert_close(total, sum(weighted.values()))

    total.backward()
    for key in SUPPORTED_GS_KEYS:
        assert predictions[key].grad is not None

import ast
import re
from pathlib import Path

import torch

from utils.loss_utils import SUPPORTED_GS_KEYS, feature_loss_value


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = ("train-sr-mse.py", "train-sr.py")
FEATURE_SHAPES = {
    "means": (3,),
    "features_dc": (3,),
    "features_rest": (1, 3),
    "opacities": (1,),
    "scales": (3,),
    "quats": (4,),
}


def parse_file(filename):
    path = REPOSITORY_ROOT / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def load_train_mse_loss():
    path = REPOSITORY_ROOT / "train-sr-mse.py"
    tree = parse_file("train-sr-mse.py")
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
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["compute_all_feature_mse_loss"]


def gaussian_pair(predicted_keys=None):
    if predicted_keys is None:
        predicted_keys = set(SUPPORTED_GS_KEYS)
    predictions = {}
    targets = {}
    for index, (key, shape) in enumerate(FEATURE_SHAPES.items(), start=1):
        predictions[key] = torch.full(
            (2, *shape),
            0.1 * index,
            requires_grad=key in predicted_keys,
        )
        targets[key] = torch.zeros((2, *shape))
    return predictions, targets


def test_train_mse_loss_includes_and_weights_all_six_attributes():
    loss_function = load_train_mse_loss()
    predictions, targets = gaussian_pair()
    weights = {
        key: float(index)
        for index, key in enumerate(SUPPORTED_GS_KEYS, start=1)
    }

    total, losses, weighted = loss_function(
        out_gs=predictions,
        target_gs=targets,
        loss_weights=weights,
        post_activate_loss=False,
        quat_direct_mse=True,
    )

    assert list(losses) == SUPPORTED_GS_KEYS
    assert list(weighted) == SUPPORTED_GS_KEYS
    for key in SUPPORTED_GS_KEYS:
        torch.testing.assert_close(weighted[key], weights[key] * losses[key])
    torch.testing.assert_close(total, sum(weighted.values()))

    total.backward()
    for key in SUPPORTED_GS_KEYS:
        assert predictions[key].grad is not None


def test_train_mse_loss_only_requires_gradients_for_predicted_attributes():
    loss_function = load_train_mse_loss()
    predicted_keys = {"means", "features_dc"}
    predictions, targets = gaussian_pair(predicted_keys)

    total, _, _ = loss_function(
        out_gs=predictions,
        target_gs=targets,
        loss_weights={key: 1.0 for key in SUPPORTED_GS_KEYS},
        post_activate_loss=False,
        quat_direct_mse=True,
    )
    total.backward()

    for key in SUPPORTED_GS_KEYS:
        if key in predicted_keys:
            assert predictions[key].grad is not None
        else:
            assert predictions[key].grad is None


def test_train_scripts_use_public_helpers_and_no_deleted_sr_utilities():
    for filename in TRAIN_SCRIPTS:
        tree = parse_file(filename)
        private_functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_")
            and not node.name.startswith("__")
        ]
        legacy_imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("utils.sr_")
        ]
        assert private_functions == [], f"{filename}: {private_functions}"
        assert legacy_imports == [], f"{filename}: {legacy_imports}"


def test_train_mse_has_no_loss_selection_or_fixed_attribute_plumbing():
    source = (REPOSITORY_ROOT / "train-sr-mse.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "loss_features",
        "fixed_attribute_keys",
        "parse_loss_features",
        "select_fixed_attribute_keys",
        "copy_gt_attributes",
    )
    assert all(name not in source for name in forbidden)


def test_train_sr_uses_canonical_frame_conversion_twice():
    tree = parse_file("train-sr.py")
    conversions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gs_utils"
        and node.func.attr == "convert_gaussian_frame"
    ]
    assert len(conversions) == 2


def test_densification_has_no_gt_attribute_override_argument():
    tree = parse_file("sr/densification.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_densified_input"
    )
    arguments = [argument.arg for argument in function.args.args]
    assert "gt_attribute_keys" not in arguments


def configured_list(config_text, binding):
    match = re.search(
        rf"^{re.escape(binding)}\s*=\s*(\[[^\n]+\])",
        config_text,
        flags=re.MULTILINE,
    )
    assert match is not None, binding
    return ast.literal_eval(match.group(1))


def test_ptv3_config_keeps_all_six_model_inputs_and_outputs():
    config = (REPOSITORY_ROOT / "configs/model/ptv3.gin").read_text(
        encoding="utf-8"
    )
    expected = set(SUPPORTED_GS_KEYS)
    for binding in (
        "FeaturePredictor.input_features",
        "FeaturePredictor.output_features",
    ):
        configured = configured_list(config, binding)
        assert len(configured) == len(expected)
        assert set(configured) == expected


def test_mse_launcher_removed_and_shifted_loss_feature_argument():
    launcher = (
        REPOSITORY_ROOT / "scripts/train-sr-mse-on-objaverse.sh"
    ).read_text(encoding="utf-8")
    assert "LOSS_FEATURES" not in launcher
    assert "--loss_features" not in launcher
    assert "POST_ACTIVATE_LOSS=${9" in launcher
    assert "DIRECT_PREDICTION=${10" in launcher
    assert "MEANS_ORIGIN_SCALE=${11" in launcher


def test_trainers_and_launchers_use_gs_sr_resolution_contract():
    trainer_sources = {
        filename: (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
        for filename in TRAIN_SCRIPTS
    }
    launcher_paths = (
        "scripts/train-sr-on-objaverse.sh",
        "scripts/train-sr-mse-on-objaverse.sh",
    )
    launcher_sources = {
        filename: (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
        for filename in launcher_paths
    }
    forbidden = (
        "dataset.GS_multi",
        "SplatFactoMultiLevelDataset",
        "input_factor",
        "target_factor",
        "factor_data",
        "multilevel",
        "load_factor_views",
    )
    for filename, source in {**trainer_sources, **launcher_sources}.items():
        assert all(token not in source for token in forbidden), filename
        assert "input_resolution" in source
        assert "target_resolution" in source

    for filename in ("configs/train/sr.gin", "configs/train/sr_mse.gin"):
        config = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
        assert "SplatFactoMultiLevelDataset" not in config


def test_launchers_load_shared_dataset_config_and_bind_manifests():
    for filename in (
        "scripts/train-sr-on-objaverse.sh",
        "scripts/train-sr-mse-on-objaverse.sh",
    ):
        launcher = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
        assert "--gin_file=configs/dataset/objaverse-sr.gin" in launcher
        assert "DATASET_ROOT" in launcher
        assert "TRAIN_SCENE_LIST" in launcher
        assert "TEST_SCENE_LIST" in launcher
        assert "SplatFactoSRDataset.resolutions" in launcher
        assert "SplatFactoSRDataset.fit_source_resolution" in launcher
        assert "SplatFactoSRDataset.fit_target_resolution" in launcher


def test_mse_trainer_exposes_reference_alignment_modes():
    source = (REPOSITORY_ROOT / "train-sr-mse.py").read_text(
        encoding="utf-8"
    )
    for mode in ("emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"):
        assert f"\"{mode}\"" in source
    assert "prepare_alignment" in source
    assert "force_matching_fit=False" in source
    assert "write_artifacts=False" in source

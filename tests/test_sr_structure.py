import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SR_FILES = {
    "alignment.py",
    "densification.py",
    "flow.py",
    "matching.py",
}


def function_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def loaded_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }


def test_sr_package_contains_only_the_four_domain_modules():
    actual = {
        path.name
        for path in (REPOSITORY_ROOT / "sr").glob("*.py")
    }
    assert actual == SR_FILES


def test_sr_path_has_no_single_leading_underscore_functions():
    paths = [
        REPOSITORY_ROOT / "dataset" / "GS_SR.py",
        REPOSITORY_ROOT / "dataset" / "gs_io.py",
        REPOSITORY_ROOT / "dataset" / "gs_processing.py",
        *[
            REPOSITORY_ROOT / "sr" / filename
            for filename in sorted(SR_FILES)
        ],
    ]
    for path in paths:
        private_names = [
            name
            for name in function_names(path)
            if name.startswith("_") and not name.startswith("__")
        ]
        assert private_names == [], f"{path}: {private_names}"


def test_overfit_scripts_do_not_load_selectable_loss_feature_names():
    forbidden = {"attribute_keys", "loss_features", "fixed_attribute_keys"}
    for filename in ("overfit-sr-mse.py", "overfit-sr-gsfm.py"):
        names = loaded_names(REPOSITORY_ROOT / filename)
        assert names.isdisjoint(forbidden), f"{filename}: {names & forbidden}"


def test_obsolete_sr_utility_modules_are_removed():
    for filename in (
        "sr_dataset_utils.py",
        "sr_densify_utils.py",
        "sr_matching_utils.py",
    ):
        assert not (REPOSITORY_ROOT / "utils" / filename).exists()

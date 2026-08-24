#!/usr/bin/env python3
"""Run the untouched current overfit code with a selectable dataset backend."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from absl import app


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CURRENT_ENTRYPOINT = REPO_ROOT / "overfit-sr-mse.py"
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compat import LegacyResolutionDatasetAdapter  # noqa: E402


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for the nerfstudio_factor backend")
    return value


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _load_current_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "sr_refactor_isolation_current_overfit", CURRENT_ENTRYPOINT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import current entrypoint: {CURRENT_ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_dataset_adapter():
    from dataset.GS_multi import SplatFactoMultiLevelDataset

    input_factor = _int_env("SR_ABLATION_INPUT_FACTOR", 4)
    target_factor = _int_env("SR_ABLATION_TARGET_FACTOR", 1)
    input_resolution = _int_env("SR_ABLATION_INPUT_RESOLUTION", 128)
    target_resolution = _int_env("SR_ABLATION_TARGET_RESOLUTION", 512)
    legacy = SplatFactoMultiLevelDataset(
        train_or_test="test",
        nerfstudio_folder=_required_env("SR_ABLATION_NERFSTUDIO_ROOT"),
        colmap_folder=_required_env("SR_ABLATION_COLMAP_ROOT"),
        load_pose_src="nerfstudio",
        sample_ratio_test=None,
        image_per_scene=None,
        remove_outlier_ndevs=-1,
        max_gs_num=1_000_000,
        split_across_gpus=False,
        factors=[target_factor, input_factor],
        background_color=[0, 0, 0],
        skip_invalid_scenes=False,
    )
    return LegacyResolutionDatasetAdapter(
        legacy,
        input_factor=input_factor,
        target_factor=target_factor,
        input_resolution=input_resolution,
        target_resolution=target_resolution,
    )


def main() -> None:
    backend = os.environ.get("SR_ABLATION_BACKEND", "gsplat_native").strip()
    if backend not in ("gsplat_native", "nerfstudio_factor"):
        raise ValueError(
            "SR_ABLATION_BACKEND must be gsplat_native or nerfstudio_factor"
        )

    os.chdir(REPO_ROOT)
    current = _load_current_entrypoint()
    if backend == "nerfstudio_factor":
        # Register legacy Gin configurables before current.main() locks config.
        import dataset.GS_multi  # noqa: F401

    original_build_dataset = current.build_dataset
    if backend == "nerfstudio_factor":
        current.build_dataset = lambda scope="test_dataset": _legacy_dataset_adapter()
    else:
        current.build_dataset = original_build_dataset

    def delegated_main(argv):
        output_dir = Path(current.FLAGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        backend_report = {
            "version": 1,
            "backend": backend,
            "current_entrypoint": str(CURRENT_ENTRYPOINT),
            "current_entrypoint_mtime_ns": CURRENT_ENTRYPOINT.stat().st_mtime_ns,
        }
        if backend == "nerfstudio_factor":
            backend_report.update(
                {
                    "nerfstudio_root": _required_env(
                        "SR_ABLATION_NERFSTUDIO_ROOT"
                    ),
                    "colmap_root": _required_env("SR_ABLATION_COLMAP_ROOT"),
                    "input_factor": _int_env("SR_ABLATION_INPUT_FACTOR", 4),
                    "target_factor": _int_env("SR_ABLATION_TARGET_FACTOR", 1),
                    "input_resolution": _int_env(
                        "SR_ABLATION_INPUT_RESOLUTION", 128
                    ),
                    "target_resolution": _int_env(
                        "SR_ABLATION_TARGET_RESOLUTION", 512
                    ),
                }
            )
        (output_dir / "ablation_backend.json").write_text(
            json.dumps(backend_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return current.main(argv)

    app.run(delegated_main)


if __name__ == "__main__":
    main()

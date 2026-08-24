#!/usr/bin/env python3
"""Build and verify an isolated legacy-format fixture for one SR scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compat import (  # noqa: E402
    compare_camera_metadata,
    compare_canonical_gs,
    compare_normalized_backends,
    create_legacy_fixture,
    load_gsplat_checkpoint,
    load_nerfstudio_checkpoint,
    render_checkpoint_psnr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native gsplat/Nerfstudio outputs into the legacy df-1/df-4 "
            "layout without changing Gaussian parameterization."
        )
    )
    parser.add_argument("--scene-name", required=True)
    parser.add_argument(
        "--source-format", choices=("gsplat", "nerfstudio"), required=True
    )
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--input-camera-metadata", type=Path, required=True)
    parser.add_argument("--target-camera-metadata", type=Path, required=True)
    parser.add_argument("--input-colmap-scene", type=Path, required=True)
    parser.add_argument("--target-colmap-scene", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-factor", type=int, default=4)
    parser.add_argument("--target-factor", type=int, default=1)
    parser.add_argument(
        "--camera-tolerance",
        type=float,
        default=1e-6,
        help="Maximum absolute camera/intrinsics difference.",
    )
    parser.add_argument(
        "--verify-render",
        action="store_true",
        help="Rerender both native checkpoints after conversion on CUDA.",
    )
    parser.add_argument("--render-device", default="cuda")
    parser.add_argument(
        "--max-render-views",
        type=int,
        default=0,
        help="Zero verifies all views; positive values limit verification.",
    )
    parser.add_argument("--input-reference-stats", type=Path)
    parser.add_argument("--target-reference-stats", type=Path)
    parser.add_argument("--render-psnr-tolerance", type=float, default=0.2)
    return parser.parse_args()


def _canonical_loader(source_format: str):
    return (
        load_gsplat_checkpoint
        if source_format == "gsplat"
        else load_nerfstudio_checkpoint
    )


def _verify_converted_checkpoint(
    source_format: str,
    source_checkpoint: Path,
    converted_checkpoint: Path,
):
    source = _canonical_loader(source_format)(source_checkpoint)
    converted = load_nerfstudio_checkpoint(converted_checkpoint)
    return {
        "raw_tensor_max_abs": compare_canonical_gs(source, converted),
        "normalized": compare_normalized_backends(source, converted),
    }


def _reference_psnr(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "psnr" not in report:
        raise ValueError(f"Reference statistics have no psnr value: {path}")
    return float(report["psnr"])


def _render_verification(
    *,
    label: str,
    checkpoint: Path,
    colmap_scene: Path,
    reference_stats: Path,
    device: str,
    max_views: int,
    tolerance: float,
):
    canonical = load_nerfstudio_checkpoint(checkpoint)
    metrics = render_checkpoint_psnr(
        canonical,
        colmap_scene,
        device=device,
        max_views=max_views or None,
    )
    if reference_stats is not None and max_views == 0:
        reference = _reference_psnr(reference_stats)
        metrics["reference_psnr"] = reference
        metrics["reference_difference"] = abs(metrics["psnr"] - reference)
        if metrics["reference_difference"] > tolerance:
            raise ValueError(
                f"{label} rerender PSNR differs from producer statistics by "
                f"{metrics['reference_difference']:.4f} dB (tolerance {tolerance})"
            )
    return metrics


def main() -> int:
    args = parse_args()
    manifest = create_legacy_fixture(
        scene_name=args.scene_name,
        source_format=args.source_format,
        input_checkpoint=args.input_checkpoint,
        target_checkpoint=args.target_checkpoint,
        input_camera_metadata=args.input_camera_metadata,
        target_camera_metadata=args.target_camera_metadata,
        input_colmap_scene=args.input_colmap_scene,
        target_colmap_scene=args.target_colmap_scene,
        output_root=args.output_root,
        input_factor=args.input_factor,
        target_factor=args.target_factor,
    )

    fixture_ns_scene = args.output_root.resolve() / "nerfstudio" / args.scene_name
    input_converted = (
        fixture_ns_scene
        / f"df-{args.input_factor}"
        / "splatfacto"
        / "nerfstudio_models"
        / "step-000015001.ckpt"
    )
    target_converted = (
        fixture_ns_scene
        / f"df-{args.target_factor}"
        / "splatfacto"
        / "nerfstudio_models"
        / "step-000015001.ckpt"
    )
    verification = {
        "tensor_max_abs": {
            "input": _verify_converted_checkpoint(
                args.source_format, args.input_checkpoint, input_converted
            ),
            "target": _verify_converted_checkpoint(
                args.source_format, args.target_checkpoint, target_converted
            ),
        },
        "camera": {
            "input": compare_camera_metadata(
                args.input_camera_metadata,
                args.input_colmap_scene,
                tolerance=args.camera_tolerance,
            ),
            "target": compare_camera_metadata(
                args.target_camera_metadata,
                args.target_colmap_scene,
                tolerance=args.camera_tolerance,
            ),
        },
    }
    if args.verify_render:
        verification["render"] = {
            "input": _render_verification(
                label="input",
                checkpoint=input_converted,
                colmap_scene=args.input_colmap_scene,
                reference_stats=args.input_reference_stats,
                device=args.render_device,
                max_views=args.max_render_views,
                tolerance=args.render_psnr_tolerance,
            ),
            "target": _render_verification(
                label="target",
                checkpoint=target_converted,
                colmap_scene=args.target_colmap_scene,
                reference_stats=args.target_reference_stats,
                device=args.render_device,
                max_views=args.max_render_views,
                tolerance=args.render_psnr_tolerance,
            ),
        }

    verification_path = args.output_root.resolve() / "verification.json"
    report = {"fixture": manifest, "verification": verification}
    verification_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

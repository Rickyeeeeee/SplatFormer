"""Shared compatibility helpers for the standalone SR refactor ablation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


GS_KEYS = (
    "means",
    "features_dc",
    "features_rest",
    "opacities",
    "scales",
    "quats",
)
GSPLAT_TO_CANONICAL = {
    "means": "means",
    "sh0": "features_dc",
    "shN": "features_rest",
    "opacities": "opacities",
    "scales": "scales",
    "quats": "quats",
}
NERFSTUDIO_PREFIX = "_model.gauss_params."
DEFAULT_NERFSTUDIO_CHECKPOINT_NAME = "step-000015001.ckpt"


def load_torch_file(path: os.PathLike) -> object:
    """Load a trusted local tensor checkpoint on CPU."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_gsplat_checkpoint(path: os.PathLike) -> Dict[str, torch.Tensor]:
    checkpoint = load_torch_file(path)
    splats = checkpoint.get("splats") if isinstance(checkpoint, dict) else None
    if not isinstance(splats, Mapping):
        raise ValueError(f"Checkpoint has no splats mapping: {path}")
    missing = sorted(set(GSPLAT_TO_CANONICAL) - set(splats))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing gsplat keys {missing}")

    canonical = {
        target: splats[source].detach().float().cpu().clone()
        for source, target in GSPLAT_TO_CANONICAL.items()
    }
    if canonical["features_dc"].ndim != 3 or canonical["features_dc"].shape[1] != 1:
        raise ValueError(
            "gsplat sh0 must have shape [N, 1, 3], got "
            f"{tuple(canonical['features_dc'].shape)}"
        )
    canonical["features_dc"] = canonical["features_dc"].squeeze(1)
    canonical["opacities"] = canonical["opacities"].reshape(-1, 1)
    validate_canonical_gs(canonical, context=f"gsplat checkpoint {path}")
    return canonical


def load_nerfstudio_checkpoint(path: os.PathLike) -> Dict[str, torch.Tensor]:
    checkpoint = load_torch_file(path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Nerfstudio checkpoint is not a mapping: {path}")
    canonical = {
        str(key).replace(NERFSTUDIO_PREFIX, ""): value.detach().float().cpu().clone()
        for key, value in checkpoint.items()
        if str(key).startswith(NERFSTUDIO_PREFIX)
    }
    missing = sorted(set(GS_KEYS) - set(canonical))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing Nerfstudio keys {missing}")
    canonical = {key: canonical[key] for key in GS_KEYS}
    canonical["opacities"] = canonical["opacities"].reshape(-1, 1)
    validate_canonical_gs(canonical, context=f"Nerfstudio checkpoint {path}")
    return canonical


def canonical_to_nerfstudio_state(
    canonical: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    validate_canonical_gs(canonical, context="canonical input")
    return {
        f"{NERFSTUDIO_PREFIX}{key}": canonical[key].detach().cpu().clone()
        for key in GS_KEYS
    }


def canonical_to_gsplat_splats(
    canonical: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    validate_canonical_gs(canonical, context="canonical input")
    return {
        "means": canonical["means"].detach().cpu().clone(),
        "sh0": canonical["features_dc"].detach().cpu().unsqueeze(1).clone(),
        "shN": canonical["features_rest"].detach().cpu().clone(),
        "opacities": canonical["opacities"].detach().cpu().reshape(-1).clone(),
        "scales": canonical["scales"].detach().cpu().clone(),
        "quats": canonical["quats"].detach().cpu().clone(),
    }


def validate_canonical_gs(
    canonical: Mapping[str, torch.Tensor], context: str = "Gaussian parameters"
) -> None:
    missing = sorted(set(GS_KEYS) - set(canonical))
    if missing:
        raise KeyError(f"{context} is missing keys {missing}")
    count = canonical["means"].shape[0]
    expected_shapes = {
        "means": (count, 3),
        "features_dc": (count, 3),
        "opacities": (count, 1),
        "scales": (count, 3),
        "quats": (count, 4),
    }
    for key in GS_KEYS:
        value = canonical[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{context} key {key} is not a tensor")
        if value.shape[0] != count:
            raise ValueError(
                f"{context} has inconsistent {key} count: {value.shape[0]} vs {count}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{context} key {key} contains non-finite values")
    for key, shape in expected_shapes.items():
        if tuple(canonical[key].shape) != shape:
            raise ValueError(
                f"{context} key {key} has shape {tuple(canonical[key].shape)}, "
                f"expected {shape}"
            )
    if canonical["features_rest"].ndim != 3:
        raise ValueError(
            f"{context} features_rest must be rank 3, got "
            f"{tuple(canonical['features_rest'].shape)}"
        )


def compare_canonical_gs(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Dict[str, float]:
    validate_canonical_gs(left, context="left Gaussian parameters")
    validate_canonical_gs(right, context="right Gaussian parameters")
    report = {}
    for key in GS_KEYS:
        if tuple(left[key].shape) != tuple(right[key].shape):
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(left[key].shape)} vs "
                f"{tuple(right[key].shape)}"
            )
        difference = (left[key] - right[key]).abs()
        report[key] = float(difference.max()) if difference.numel() else 0.0
        torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol)
    return report


def normalized_gs(
    canonical: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], object]:
    from utils.transform_utils import MinMaxScaler

    normalized = {key: value.clone() for key, value in canonical.items()}
    scaler = MinMaxScaler()
    normalized["means"] = scaler.fit_transform(normalized["means"])
    normalized["scales"] = normalized["scales"] + torch.log(scaler.scale_)
    return normalized, scaler


def compare_normalized_backends(
    source: Mapping[str, torch.Tensor],
    converted: Mapping[str, torch.Tensor],
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Compare normalized tensors and fitted scaler state for two backends."""
    source_normalized, source_scaler = normalized_gs(source)
    converted_normalized, converted_scaler = normalized_gs(converted)
    tensor_differences = compare_canonical_gs(
        source_normalized,
        converted_normalized,
        atol=tolerance,
        rtol=0.0,
    )
    scaler_differences = {}
    for name in ("scale_", "trans_", "data_min_", "data_max_", "data_range_"):
        source_value = torch.as_tensor(getattr(source_scaler, name))
        converted_value = torch.as_tensor(getattr(converted_scaler, name))
        difference = (source_value - converted_value).abs()
        maximum = float(difference.max()) if difference.numel() else 0.0
        scaler_differences[name] = maximum
        torch.testing.assert_close(
            source_value, converted_value, atol=tolerance, rtol=0.0
        )
    return {
        "normalized_tensor_max_abs": tensor_differences,
        "scaler_max_abs": scaler_differences,
    }


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(canonical: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
            "std": float(value.std()),
        }
        for key, value in canonical.items()
    }


def ensure_symlink(source: os.PathLike, destination: os.PathLike) -> None:
    source = Path(source).resolve()
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(
            f"Refusing to replace mismatched symlink {destination} -> "
            f"{destination.resolve()}"
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _write_converted_checkpoint(
    source_checkpoint: os.PathLike,
    output_checkpoint: os.PathLike,
) -> Dict[str, torch.Tensor]:
    canonical = load_gsplat_checkpoint(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if output_checkpoint.exists():
        existing = load_nerfstudio_checkpoint(output_checkpoint)
        compare_canonical_gs(canonical, existing)
    else:
        torch.save(canonical_to_nerfstudio_state(canonical), output_checkpoint)
    converted = load_nerfstudio_checkpoint(output_checkpoint)
    compare_canonical_gs(canonical, converted)
    return converted


def _prepare_factor(
    source_format: str,
    source_checkpoint: os.PathLike,
    camera_metadata: os.PathLike,
    factor_dir: Path,
) -> Dict[str, torch.Tensor]:
    models_dir = factor_dir / "splatfacto" / "nerfstudio_models"
    output_checkpoint = models_dir / DEFAULT_NERFSTUDIO_CHECKPOINT_NAME
    output_camera = factor_dir / "splatfacto" / "camera_for-3d-denoise.pkl"
    if source_format == "gsplat":
        canonical = _write_converted_checkpoint(source_checkpoint, output_checkpoint)
    elif source_format == "nerfstudio":
        ensure_symlink(source_checkpoint, output_checkpoint)
        canonical = load_nerfstudio_checkpoint(source_checkpoint)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")
    ensure_symlink(camera_metadata, output_camera)
    return canonical


def create_legacy_fixture(
    *,
    scene_name: str,
    source_format: str,
    input_checkpoint: os.PathLike,
    target_checkpoint: os.PathLike,
    input_camera_metadata: os.PathLike,
    target_camera_metadata: os.PathLike,
    input_colmap_scene: os.PathLike,
    target_colmap_scene: os.PathLike,
    output_root: os.PathLike,
    input_factor: int = 4,
    target_factor: int = 1,
) -> Dict[str, object]:
    output_root = Path(output_root).resolve()
    input_colmap_scene = Path(input_colmap_scene).resolve()
    target_colmap_scene = Path(target_colmap_scene).resolve()
    fixture_ns_scene = output_root / "nerfstudio" / scene_name
    fixture_colmap_scene = output_root / "colmap" / scene_name

    input_gs = _prepare_factor(
        source_format,
        input_checkpoint,
        input_camera_metadata,
        fixture_ns_scene / f"df-{input_factor}",
    )
    target_gs = _prepare_factor(
        source_format,
        target_checkpoint,
        target_camera_metadata,
        fixture_ns_scene / f"df-{target_factor}",
    )
    ensure_symlink(target_colmap_scene / "images", fixture_colmap_scene / "images")
    ensure_symlink(
        input_colmap_scene / "images",
        fixture_colmap_scene / f"images_{input_factor}",
    )

    manifest = {
        "version": 1,
        "scene_name": scene_name,
        "source_format": source_format,
        "input_factor": int(input_factor),
        "target_factor": int(target_factor),
        "fixture": {
            "root": str(output_root),
            "nerfstudio_root": str(output_root / "nerfstudio"),
            "colmap_root": str(output_root / "colmap"),
        },
        "input": _fixture_source_record(
            input_checkpoint,
            input_camera_metadata,
            input_colmap_scene,
            input_gs,
        ),
        "target": _fixture_source_record(
            target_checkpoint,
            target_camera_metadata,
            target_colmap_scene,
            target_gs,
        ),
    }
    manifest_path = output_root / "fixture_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                f"Existing fixture manifest does not match requested fixture: {manifest_path}"
            )
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _fixture_source_record(
    checkpoint: os.PathLike,
    camera_metadata: os.PathLike,
    colmap_scene: os.PathLike,
    canonical: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    checkpoint = Path(checkpoint).resolve()
    camera_metadata = Path(camera_metadata).resolve()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "camera_metadata": str(camera_metadata),
        "camera_metadata_sha256": sha256_file(camera_metadata),
        "colmap_scene": str(Path(colmap_scene).resolve()),
        "gaussians": tensor_summary(canonical),
    }


def compare_camera_metadata(
    camera_metadata: os.PathLike,
    colmap_scene: os.PathLike,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    from dataset.GS_SR import SplatFactoSRDataset

    with open(camera_metadata, "rb") as handle:
        legacy = pickle.load(handle)
    dataset = object.__new__(SplatFactoSRDataset)
    native, image_paths = dataset.load_images_cameras_fromcolmap(str(colmap_scene))

    legacy_cameras = torch.as_tensor(
        legacy["train_camera_to_worlds"], dtype=torch.float32
    )
    native_cameras = native["camera_to_worlds"][:, :3, :4]
    if legacy_cameras.shape != native_cameras.shape:
        raise ValueError(
            f"Camera shape mismatch: {tuple(legacy_cameras.shape)} vs "
            f"{tuple(native_cameras.shape)}"
        )
    camera_max_abs = float((legacy_cameras - native_cameras).abs().max())
    if camera_max_abs > tolerance:
        raise ValueError(
            f"Camera matrices differ by {camera_max_abs}, tolerance is {tolerance}"
        )

    scalar_differences = {}
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        difference = abs(float(np.asarray(legacy[key])) - float(native[key]))
        scalar_differences[key] = difference
        if difference > tolerance:
            raise ValueError(
                f"Camera metadata {key} differs by {difference}, tolerance is {tolerance}"
            )
    legacy_names = sorted(
        name for name in os.listdir(Path(colmap_scene) / "images")
        if name.lower().endswith(".png")
    )
    native_names = [Path(path).name for path in image_paths]
    if legacy_names != native_names:
        raise ValueError("Legacy and COLMAP image-name ordering differs")
    return {
        "camera_max_abs": camera_max_abs,
        "scalar_max_abs": max(scalar_differences.values()),
        "scalar_differences": scalar_differences,
        "image_count": len(native_names),
    }


def render_checkpoint_psnr(
    canonical: Mapping[str, torch.Tensor],
    colmap_scene: os.PathLike,
    *,
    device: str = "cuda",
    max_views: Optional[int] = None,
) -> Dict[str, float]:
    from dataset.GS_SR import SplatFactoSRDataset
    from utils import gpu_utils, gs_utils

    dataset = object.__new__(SplatFactoSRDataset)
    meta, image_paths = dataset.load_images_cameras_fromcolmap(str(colmap_scene))
    if max_views is not None and max_views > 0:
        image_paths = image_paths[:max_views]
        meta["camera_to_worlds"] = meta["camera_to_worlds"][:max_views]
    cameras = {
        **meta,
        "background_color": torch.zeros(3, dtype=torch.float32),
    }
    torch_device = torch.device(device)
    render_gs = gpu_utils.move_to_device(dict(canonical), torch_device)
    cameras = gpu_utils.move_to_device(cameras, torch_device)
    view_psnr = []
    with torch.no_grad():
        for index, image_path in enumerate(image_paths):
            prediction, _ = gs_utils.rasterize_gaussians_to_singleimg(
                render_gs,
                cameras["camera_to_worlds"][index],
                **cameras,
            )
            ground_truth = _read_black_composited_image(image_path).to(torch_device)
            mse = torch.mean((prediction - ground_truth) ** 2)
            view_psnr.append(float(-10.0 * torch.log10(mse.clamp_min(1e-12))))
    return {
        "psnr": float(np.mean(view_psnr)),
        "min_psnr": float(np.min(view_psnr)),
        "max_psnr": float(np.max(view_psnr)),
        "view_count": len(view_psnr),
    }


def _read_black_composited_image(path: os.PathLike) -> torch.Tensor:
    image = np.asarray(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4]
    return torch.from_numpy(image[..., :3])


class LegacyResolutionDatasetAdapter:
    """Expose a legacy factor dataset through the current resolution interface."""

    def __init__(
        self,
        legacy_dataset,
        *,
        input_factor: int = 4,
        target_factor: int = 1,
        input_resolution: int = 128,
        target_resolution: int = 512,
    ):
        self.legacy_dataset = legacy_dataset
        self.input_factor = int(input_factor)
        self.target_factor = int(target_factor)
        self.input_resolution = int(input_resolution)
        self.target_resolution = int(target_resolution)
        self.resolutions = [self.input_resolution, self.target_resolution]
        self.primary_resolution = self.input_resolution
        self.folders = legacy_dataset.folders
        self.image_per_scene = legacy_dataset.image_per_scene
        self.background_color = legacy_dataset.background_color

    def load_scene(self, scene_idx: int) -> Dict[str, object]:
        legacy_scene = self.legacy_dataset.load_scene(scene_idx)
        factor_data = legacy_scene["factor_data"]
        return {
            "idx": legacy_scene["idx"],
            "scene_name": legacy_scene["scene_name"],
            "resolution_data": {
                self.input_resolution: factor_data[self.input_factor],
                self.target_resolution: factor_data[self.target_factor],
            },
        }

    def load_resolution_views(
        self,
        resolution_entry,
        camera_ids: Optional[Sequence[int]] = None,
        background=None,
    ):
        return self.legacy_dataset.load_factor_views(
            resolution_entry,
            cam_ids=camera_ids,
            background=background,
        )

    def __getattr__(self, name: str):
        return getattr(self.legacy_dataset, name)

    def __len__(self) -> int:
        return len(self.legacy_dataset)

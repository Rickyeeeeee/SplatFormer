"""Focused tests for the standalone compatibility fixture and adapter."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

try:
    import pytest
except ImportError:
    class _Raises:
        def __init__(self, exception):
            self.exception = exception

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exception.__name__}")
            return issubclass(exc_type, self.exception)

    class _PytestFallback:
        def raises(self, exception):
            return _Raises(exception)

    pytest = _PytestFallback()


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from compat import (  # noqa: E402
    GS_KEYS,
    LegacyResolutionDatasetAdapter,
    canonical_to_gsplat_splats,
    canonical_to_nerfstudio_state,
    compare_camera_metadata,
    compare_canonical_gs,
    create_legacy_fixture,
    load_gsplat_checkpoint,
    load_nerfstudio_checkpoint,
    normalized_gs,
)


def canonical_gaussians(count: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4242)
    return {
        "means": torch.randn(count, 3, generator=generator) * 2.0 - 0.4,
        "features_dc": torch.randn(count, 3, generator=generator) * 3.0,
        "features_rest": torch.randn(count, 3, 3, generator=generator),
        "opacities": torch.linspace(-7.0, 3.0, count).reshape(-1, 1),
        "scales": torch.randn(count, 3, generator=generator) - 2.0,
        "quats": torch.randn(count, 4, generator=generator) * 4.0,
    }


def save_gsplat(path: Path, canonical: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": 14999, "splats": canonical_to_gsplat_splats(canonical)},
        path,
    )


def save_nerfstudio(path: Path, canonical: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(canonical_to_nerfstudio_state(canonical), path)


def test_checkpoint_conversion_preserves_raw_parameters_and_singletons(tmp_path: Path):
    expected = canonical_gaussians()
    gsplat_path = tmp_path / "native.pt"
    legacy_path = tmp_path / "legacy.ckpt"
    save_gsplat(gsplat_path, expected)

    loaded_native = load_gsplat_checkpoint(gsplat_path)
    save_nerfstudio(legacy_path, loaded_native)
    loaded_legacy = load_nerfstudio_checkpoint(legacy_path)

    assert compare_canonical_gs(expected, loaded_native) == {key: 0.0 for key in GS_KEYS}
    assert compare_canonical_gs(expected, loaded_legacy) == {key: 0.0 for key in GS_KEYS}
    raw = torch.load(gsplat_path, map_location="cpu", weights_only=True)["splats"]
    assert raw["sh0"].shape == (5, 1, 3)
    assert raw["opacities"].shape == (5,)
    assert loaded_legacy["features_dc"].shape == (5, 3)
    assert loaded_legacy["opacities"].shape == (5, 1)
    torch.testing.assert_close(loaded_legacy["opacities"], expected["opacities"])
    torch.testing.assert_close(loaded_legacy["scales"], expected["scales"])
    torch.testing.assert_close(loaded_legacy["quats"], expected["quats"])
    assert not torch.allclose(
        torch.linalg.vector_norm(loaded_legacy["quats"], dim=-1),
        torch.ones(5),
    )


def test_fixture_layout_maps_native_128_to_factor4_and_512_to_factor1(tmp_path: Path):
    input_gs = canonical_gaussians(4)
    target_gs = canonical_gaussians(7)
    input_checkpoint = tmp_path / "source" / "128.pt"
    target_checkpoint = tmp_path / "source" / "512.pt"
    save_gsplat(input_checkpoint, input_gs)
    save_gsplat(target_checkpoint, target_gs)

    input_colmap = tmp_path / "native" / "128" / "scene"
    target_colmap = tmp_path / "native" / "512" / "scene"
    for scene, size in ((input_colmap, 128), (target_colmap, 512)):
        images = scene / "images"
        images.mkdir(parents=True)
        Image.new("RGB", (size, size)).save(images / "000.png")
    input_camera = tmp_path / "source" / "camera128.pkl"
    target_camera = tmp_path / "source" / "camera512.pkl"
    input_camera.write_bytes(b"camera-128")
    target_camera.write_bytes(b"camera-512")

    fixture = tmp_path / "fixture"
    manifest = create_legacy_fixture(
        scene_name="example",
        source_format="gsplat",
        input_checkpoint=input_checkpoint,
        target_checkpoint=target_checkpoint,
        input_camera_metadata=input_camera,
        target_camera_metadata=target_camera,
        input_colmap_scene=input_colmap,
        target_colmap_scene=target_colmap,
        output_root=fixture,
        input_factor=4,
        target_factor=1,
    )

    df4 = fixture / "nerfstudio/example/df-4/splatfacto"
    df1 = fixture / "nerfstudio/example/df-1/splatfacto"
    loaded_df4 = load_nerfstudio_checkpoint(
        df4 / "nerfstudio_models/step-000015001.ckpt"
    )
    loaded_df1 = load_nerfstudio_checkpoint(
        df1 / "nerfstudio_models/step-000015001.ckpt"
    )
    compare_canonical_gs(input_gs, loaded_df4)
    compare_canonical_gs(target_gs, loaded_df1)
    assert (fixture / "colmap/example/images_4").resolve() == input_colmap / "images"
    assert (fixture / "colmap/example/images").resolve() == target_colmap / "images"
    assert (df4 / "camera_for-3d-denoise.pkl").resolve() == input_camera
    assert (df1 / "camera_for-3d-denoise.pkl").resolve() == target_camera
    assert manifest["input_factor"] == 4
    assert manifest["target_factor"] == 1
    assert json.loads((fixture / "fixture_manifest.json").read_text())["source_format"] == "gsplat"


def _write_colmap_text_scene(scene: Path) -> None:
    sparse = scene / "sparse" / "0"
    images = scene / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 16 8 10.0 11.0 8.0 4.0\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list\n"
        "2 1 0 0 0 1 2 3 1 b.png\n\n"
        "1 1 0 0 0 0 0 0 1 a.png\n\n",
        encoding="utf-8",
    )
    Image.new("RGB", (16, 8), color=(20, 30, 40)).save(images / "a.png")
    Image.new("RGB", (16, 8), color=(50, 60, 70)).save(images / "b.png")


def test_camera_matrices_intrinsics_and_image_order_match(tmp_path: Path):
    from dataset.GS_SR import SplatFactoSRDataset

    scene = tmp_path / "colmap_scene"
    _write_colmap_text_scene(scene)
    dataset = object.__new__(SplatFactoSRDataset)
    native, paths = dataset.load_images_cameras_fromcolmap(str(scene))
    camera_file = tmp_path / "camera.pkl"
    legacy = {
        "train_camera_to_worlds": native["camera_to_worlds"][:, :3, :4].clone(),
        **{key: native[key].clone() for key in ("fx", "fy", "cx", "cy", "width", "height")},
    }
    with camera_file.open("wb") as stream:
        pickle.dump(legacy, stream)

    report = compare_camera_metadata(camera_file, scene, tolerance=1e-6)
    assert report["camera_max_abs"] == 0.0
    assert report["scalar_max_abs"] == 0.0
    assert report["image_count"] == 2
    assert [Path(path).name for path in paths] == ["a.png", "b.png"]


class FakeLegacyDataset:
    def __init__(self):
        self.folders = [{"scene": "fixture"}]
        self.image_per_scene = None
        self.background_color = [0, 0, 0]

    def load_scene(self, scene_idx: int):
        return {
            "idx": scene_idx,
            "scene_name": "fixture",
            "factor_data": {4: "native-128", 1: "native-512"},
        }

    def load_factor_views(self, entry, cam_ids=None, background=None):
        return entry, cam_ids, background

    def __len__(self):
        return 1


def test_factor_to_resolution_adapter_mapping_and_view_delegation():
    adapter = LegacyResolutionDatasetAdapter(FakeLegacyDataset())
    scene = adapter.load_scene(0)
    assert scene["resolution_data"] == {128: "native-128", 512: "native-512"}
    assert adapter.resolutions == [128, 512]
    assert adapter.primary_resolution == 128
    assert adapter.load_resolution_views("native-128", [3], "black") == (
        "native-128",
        [3],
        "black",
    )
    assert len(adapter) == 1


def test_backend_checkpoint_normalization_and_scalers_are_identical(tmp_path: Path):
    expected = canonical_gaussians(8)
    gsplat_path = tmp_path / "native.pt"
    nerfstudio_path = tmp_path / "converted.ckpt"
    save_gsplat(gsplat_path, expected)
    save_nerfstudio(nerfstudio_path, expected)
    native = load_gsplat_checkpoint(gsplat_path)
    legacy = load_nerfstudio_checkpoint(nerfstudio_path)

    native_normalized, native_scaler = normalized_gs(native)
    legacy_normalized, legacy_scaler = normalized_gs(legacy)
    compare_canonical_gs(native_normalized, legacy_normalized, atol=1e-6, rtol=0.0)
    for key in ("scale_", "trans_", "data_min_", "data_max_", "data_range_"):
        torch.testing.assert_close(
            getattr(native_scaler, key),
            getattr(legacy_scaler, key),
            atol=1e-6,
            rtol=0.0,
        )


def test_fixture_creation_refuses_to_replace_mismatched_sources(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.symlink_to(source, target_is_directory=True)
    other = tmp_path / "other"
    other.mkdir()
    from compat import ensure_symlink

    with pytest.raises(FileExistsError):
        ensure_symlink(other, destination)


def _run_without_pytest() -> int:
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    failures = []
    for name, test in tests:
        try:
            if test.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"PASS {name}")
        except Exception as error:
            failures.append((name, error))
            print(f"FAIL {name}: {error}")
    print(f"Ran {len(tests)} tests; {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())

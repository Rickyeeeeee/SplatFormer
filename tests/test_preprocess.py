import argparse
import csv
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from dataset import preprocess


def _camera_metadata(view_count):
    return {
        "train_camera_to_worlds": torch.zeros((view_count, 3, 4)),
        "fx": torch.tensor(100.0),
        "fy": torch.tensor(100.0),
        "cx": torch.tensor(50.0),
        "cy": torch.tensor(50.0),
        "width": torch.tensor(100.0),
        "height": torch.tensor(100.0),
    }


def _write_resolution(root, scene, resolution, names=("000.png", "001.png")):
    image_dir = root / str(resolution) / "colmap" / scene / "images"
    image_dir.mkdir(parents=True)
    for name in names:
        Image.new("RGB", (16, 16), color=(resolution % 255, 0, 0)).save(
            image_dir / name
        )

    splatfacto_dir = (
        root / str(resolution) / "nerfstudio" / scene / "splatfacto"
    )
    model_dir = splatfacto_dir / "nerfstudio_models"
    model_dir.mkdir(parents=True)
    (model_dir / "step-000000010.ckpt").write_bytes(b"old")
    (model_dir / "step-000000200.ckpt").write_bytes(b"new")
    with (splatfacto_dir / preprocess.CAMERA_METADATA_NAME).open("wb") as handle:
        pickle.dump(_camera_metadata(len(names)), handle)


def _output_args(root):
    return {
        "structure_report": root / "structure_report.csv",
        "structural_scene_list": root / "structurally_valid_scenes.txt",
        "metrics_csv": root / "scene_metrics.csv",
        "selection_report": root / "selection_report.csv",
        "valid_scene_list": root / "valid_scenes.txt",
        "pretrain_status": root / "pretrain_status.csv",
        "cache_root": root / "pretrained_gaussians",
        "statistics_json": root / "gs_statistics.json",
        "statistics_records": root / "gs_statistics_scenes.jsonl",

    }

def _make_gs(means):
    means = torch.as_tensor(means, dtype=torch.float32)
    count = means.shape[0]
    return {
        "means": means,
        "scales": torch.zeros((count, 3), dtype=torch.float32),
        "opacities": torch.zeros((count, 1), dtype=torch.float32),
        "quats": torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "features_dc": torch.zeros((count, 3), dtype=torch.float32),
        "features_rest": torch.zeros((count, 15, 3), dtype=torch.float32),
    }


def _scaler(scale, translation):
    return SimpleNamespace(
        scale_=torch.tensor(float(scale)),
        trans_=torch.tensor(translation, dtype=torch.float32),
    )


class PreprocessStructureTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_discovery_validation_and_latest_checkpoint(self):
        _write_resolution(self.root, "scene_b", 512)
        _write_resolution(self.root, "scene_b", 128)
        _write_resolution(self.root, "scene_a", 512)

        self.assertEqual(
            preprocess.discover_scenes(self.root, [512, 128]),
            ["scene_a", "scene_b"],
        )
        valid, reason, _ = preprocess.validate_scene_structure(
            self.root, "scene_b", [512, 128], 2
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")

        paths = preprocess.scene_resolution_paths(self.root, "scene_b", 512)
        self.assertEqual(
            preprocess.latest_checkpoint(paths["nerfstudio_dir"]).name,
            "step-000000200.ckpt",
        )

        valid, reason, _ = preprocess.validate_scene_structure(
            self.root, "scene_a", [512, 128], 2
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "missing_colmap_scene")

    def test_validation_rejects_cross_resolution_names_and_pose_counts(self):
        _write_resolution(self.root, "names", 512)
        _write_resolution(
            self.root,
            "names",
            128,
            names=("000.png", "different.png"),
        )
        valid, reason, _ = preprocess.validate_scene_structure(
            self.root, "names", [512, 128], 2
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "cross_resolution_image_mismatch")

        _write_resolution(self.root, "poses", 512)
        _write_resolution(self.root, "poses", 128)
        camera_path = preprocess.scene_resolution_paths(
            self.root, "poses", 128
        )["camera_path"]
        with camera_path.open("wb") as handle:
            pickle.dump(_camera_metadata(1), handle)
        valid, reason, _ = preprocess.validate_scene_structure(
            self.root, "poses", [512, 128], 2
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "image_pose_count_mismatch")

    def test_validate_writes_sorted_manifest_and_report(self):
        _write_resolution(self.root, "scene_b", 512)
        _write_resolution(self.root, "scene_b", 128)
        _write_resolution(self.root, "scene_a", 512)
        _write_resolution(self.root, "scene_a", 128)
        outputs = _output_args(self.root)
        args = SimpleNamespace(
            dataset_root=self.root,
            resolutions=[512, 128],
            expected_images=2,
            max_scenes=None,
            candidate_scene_list=None,
            **outputs,
        )

        selected = preprocess.run_validate(args)

        self.assertEqual(selected, ["scene_a", "scene_b"])
        self.assertEqual(
            preprocess.read_scene_list(outputs["structural_scene_list"]),
            ["scene_a", "scene_b"],
        )
        with outputs["structure_report"].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["scene"] for row in rows], ["scene_a", "scene_b"])
        self.assertTrue(all(row["status"] == "valid" for row in rows))


class PreprocessEvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.outputs = _output_args(self.root)
        preprocess.atomic_write_scene_list(
            self.outputs["structural_scene_list"], ["scene_a", "scene_b"]
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _evaluation_args(self, retry_errors=False):
        return SimpleNamespace(
            dataset_root=self.root,
            resolutions=[512, 128],
            expected_images=2,
            max_scenes=None,
            scene_list=None,
            device="cuda",
            render_chunk_size=2,
            retry_errors=retry_errors,
            **self.outputs,
        )

    def test_evaluation_writes_rows_and_resumes_completed_scenes(self):
        calls = []

        def load_bundle(dataset_root, scene, resolution, device):
            calls.append((scene, resolution))
            return {
                "checkpoint_path": Path("/tmp") / scene / ("%d.ckpt" % resolution),
                "image_paths": [Path("000.png"), Path("001.png")],
                "gs_count": resolution,
            }

        with mock.patch.object(preprocess, "require_cuda", return_value=torch.device("cpu")), \
            mock.patch.object(preprocess, "build_lpips_model", return_value=object()), \
            mock.patch.object(preprocess, "load_resolution_bundle", side_effect=load_bundle), \
            mock.patch.object(
                preprocess,
                "evaluate_bundle",
                return_value={"psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
            ), \
            mock.patch.object(torch.cuda, "empty_cache"):
            preprocess.run_evaluate(self._evaluation_args())
            preprocess.run_evaluate(self._evaluation_args())

        self.assertEqual(
            calls,
            [
                ("scene_a", 512),
                ("scene_a", 128),
                ("scene_b", 512),
                ("scene_b", 128),
            ],
        )
        latest = preprocess._read_latest_rows(
            self.outputs["metrics_csv"], ("scene", "resolution")
        )
        self.assertEqual(len(latest), 4)
        self.assertTrue(all(row["status"] == "ok" for row in latest.values()))

    def test_evaluation_error_is_retried_only_when_requested(self):
        preprocess.atomic_write_scene_list(
            self.outputs["structural_scene_list"], ["scene_a"]
        )
        call_count = {"value": 0}

        def load_bundle(*unused):
            call_count["value"] += 1
            raise RuntimeError("broken checkpoint")

        patches = [
            mock.patch.object(preprocess, "require_cuda", return_value=torch.device("cpu")),
            mock.patch.object(preprocess, "build_lpips_model", return_value=object()),
            mock.patch.object(preprocess, "load_resolution_bundle", side_effect=load_bundle),
            mock.patch.object(torch.cuda, "empty_cache"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        preprocess.run_evaluate(self._evaluation_args(retry_errors=False))
        first_count = call_count["value"]
        preprocess.run_evaluate(self._evaluation_args(retry_errors=False))
        self.assertEqual(call_count["value"], first_count)
        preprocess.run_evaluate(self._evaluation_args(retry_errors=True))
        self.assertGreater(call_count["value"], first_count)

    def test_selection_requires_every_resolution_and_uses_inclusive_minima(self):
        rows = [
            {
                "scene": "scene_a",
                "resolution": 512,
                "status": "ok",
                "gs_count": 100,
                "psnr": 25.0,
            },
            {
                "scene": "scene_a",
                "resolution": 128,
                "status": "ok",
                "gs_count": 100,
                "psnr": 25.0,
            },
            {
                "scene": "scene_b",
                "resolution": 512,
                "status": "ok",
                "gs_count": 99,
                "psnr": 30.0,
            },
        ]
        preprocess.atomic_write_csv(
            self.outputs["metrics_csv"], preprocess.METRIC_COLUMNS, rows
        )
        args = SimpleNamespace(
            dataset_root=self.root,
            resolutions=[512, 128],
            scene_list=None,
            min_psnr=25.0,
            min_gs_count=100,
            **self.outputs,
        )

        selected = preprocess.run_select(args)

        self.assertEqual(selected, ["scene_a"])
        self.assertEqual(
            preprocess.read_scene_list(self.outputs["valid_scene_list"]),
            ["scene_a"],
        )
        with self.outputs["selection_report"].open(newline="") as handle:
            report = {row["scene"]: row for row in csv.DictReader(handle)}
        self.assertEqual(report["scene_a"]["status"], "selected")
        self.assertIn("missing_metrics:128", report["scene_b"]["reason"])


class PreprocessPretrainCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.outputs = _output_args(self.root)
        preprocess.atomic_write_scene_list(
            self.outputs["valid_scene_list"], ["selected_scene"]
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _pretrain_args(self):
        return SimpleNamespace(
            dataset_root=self.root,
            resolutions=[512, 128],
            max_scenes=None,
            scene_list=None,
            direction="both",
            device="cuda",
            retry_errors=False,
            force_refit=False,
            matching_steps=2000,
            matching_images_per_step=32,
            matching_log_interval=20,
            matching_preview_interval=200,
            matching_l1_weight=1.0,
            matching_lpips_weight=1.0,
            amp=True,
            **self.outputs,
        )

    def test_pretrain_routes_both_directions_and_selected_scenes(self):
        calls = []

        def load_bundle(dataset_root, scene, resolution, device):
            return {
                "gs_params": {"means": torch.zeros((2, 3))},
                "scaler": object(),
                "cameras": {"camera_to_worlds": torch.zeros((1, 3, 4))},
                "image_paths": [Path("000.png")],
            }

        def fit_target(**kwargs):
            calls.append(
                (
                    kwargs["scene_name"],
                    kwargs["input_factor"],
                    kwargs["target_factor"],
                )
            )
            cache_path = (
                Path(kwargs["pre_matching_root"])
                / kwargs["scene_name"]
                / ("if%d_tf%d" % (kwargs["input_factor"], kwargs["target_factor"]))
                / "matching_target.pt"
            )
            return {}, {"status": "refit", "checkpoint_path": str(cache_path)}

        with mock.patch.object(preprocess, "require_cuda", return_value=torch.device("cpu")), \
            mock.patch.object(preprocess, "_bind_matching_optimizer"), \
            mock.patch.object(preprocess, "load_resolution_bundle", side_effect=load_bundle), \
            mock.patch.object(
                preprocess,
                "build_matching_source",
                return_value={"means": torch.zeros((2, 3))},
            ), \
            mock.patch.object(preprocess, "read_image", return_value=torch.zeros((4, 4, 3))), \
            mock.patch.object(preprocess, "get_or_fit_matching_target", side_effect=fit_target), \
            mock.patch.object(torch.cuda, "empty_cache"):
            preprocess.run_pretrain(self._pretrain_args())

        self.assertEqual(
            calls,
            [
                ("selected_scene", 128, 512),
                ("selected_scene", 512, 128),
            ],
        )
        latest = preprocess._read_latest_rows(
            self.outputs["pretrain_status"], ("scene", "direction")
        )
        self.assertEqual(latest[("selected_scene", "lr_to_hr")]["status"], "ok")
        self.assertEqual(latest[("selected_scene", "hr_to_lr")]["status"], "ok")

    def test_cli_requires_selection_thresholds_and_defaults_pretraining_off(self):
        parser = preprocess.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["select", "--dataset_root", str(self.root)])

        args = parser.parse_args(
            [
                "all",
                "--dataset_root",
                str(self.root),
                "--min_psnr",
                "20",
                "--min_gs_count",
                "100",
            ]
        )
        preprocess.validate_arguments(args, parser)
        self.assertEqual(args.direction, "none")
        self.assertEqual(args.resolutions, [512, 128])


class PreprocessStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.outputs = _output_args(self.root)
        preprocess.atomic_write_scene_list(
            self.outputs["valid_scene_list"], ["selected_scene"]
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _statistics_args(self):
        return SimpleNamespace(
            dataset_root=self.root,
            resolutions=[512, 128],
            expected_images=2,
            max_scenes=None,
            scene_list=None,
            retry_errors=False,
            force_statistics=False,
            **self.outputs,
        )

    def test_activation_and_equal_scene_weighting(self):
        first = _make_gs([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        second = _make_gs([[10.0, 10.0, 10.0]] * 4)
        first["scales"][0] = torch.log(torch.tensor([1.0, 2.0, 4.0]))

        activated = preprocess.activate_gaussian_params(first)
        self.assertTrue(
            torch.allclose(activated["scales"][0], torch.tensor([1.0, 2.0, 4.0]))
        )
        self.assertTrue(
            torch.allclose(activated["opacities"], torch.full((2, 1), 0.5))
        )
        self.assertTrue(
            torch.allclose(
                activated["quats"],
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            )
        )

        records = []
        for scene, gs_params in (("a", first), ("b", second)):
            records.append(
                {
                    "scene": scene,
                    "status": "ok",
                    "groups": [
                        preprocess._statistics_group(
                            "nerfstudio",
                            "checkpoint",
                            "raw",
                            gs_params,
                            resolution=128,
                        )
                    ],
                }
            )
        report = preprocess.aggregate_statistics_records(records, 2)
        parameter = report["spaces"]["checkpoint"]["raw"]["nerfstudio"]["128"][
            "parameters"
        ]["means"]
        self.assertEqual(parameter["scene_count"], 2)
        self.assertEqual(parameter["gaussian_count"], 6)
        self.assertTrue(
            torch.allclose(torch.tensor(parameter["mean"]), torch.full((3,), 5.5))
        )
        self.assertTrue(
            torch.allclose(torch.tensor(parameter["std"]), torch.full((3,), 0.5))
        )

    def test_cache_statistics_and_source_frame_paired_differences(self):
        args = self._statistics_args()
        source_scaler = _scaler(2.0, [1.0, 1.0, 1.0])
        target_scaler = _scaler(4.0, [-1.0, -1.0, -1.0])
        source_processed = _make_gs(
            [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]]
        )
        target_processed = _make_gs(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        bundles = {
            128: {
                "processed": source_processed,
                "checkpoint": preprocess.gaussian_checkpoint_space(
                    source_processed, source_scaler
                ),
                "scaler": source_scaler,
            },
            512: {
                "processed": target_processed,
                "checkpoint": preprocess.gaussian_checkpoint_space(
                    target_processed, target_scaler
                ),
                "scaler": target_scaler,
            },
        }
        expected_source = preprocess.convert_gaussian_frame(
            source_processed, source_scaler, target_scaler
        )
        fitted_target = {key: value.clone() for key, value in expected_source.items()}
        fitted_target["means"] += 0.4
        fitted_target["scales"] += torch.log(torch.tensor(2.0))
        fitted_target["opacities"] += 1.0

        cache_path = preprocess.statistics_cache_path(
            self.outputs["cache_root"], "selected_scene", 128, 512
        )
        cache_path.parent.mkdir(parents=True)
        torch.save(
            {
                "metadata": {
                    "version": 1,
                    "scene_name": "selected_scene",
                    "input_factor": 128,
                    "target_factor": 512,
                    "source_attributes": preprocess._source_attributes(expected_source),
                },
                "target_gs": fitted_target,
            },
            cache_path,
        )

        with mock.patch.object(
            preprocess,
            "load_statistics_resolution",
            side_effect=lambda unused_root, unused_scene, resolution: bundles[resolution],
        ):
            record = preprocess.process_statistics_scene(
                args, "selected_scene", {"fingerprint": 1}
            )

        self.assertEqual(record["status"], "ok")
        self.assertEqual(len(record["missing_caches"]), 1)
        groups = {
            (
                group["source_kind"],
                group["coordinate_space"],
                group["representation"],
                group.get("direction"),
            ): group
            for group in record["groups"]
        }
        pretrain = groups[("pretrain", "processed", "raw", "lr_to_hr")]
        self.assertEqual(pretrain["resolution"], 512)
        processed_delta = groups[
            ("paired_differences", "processed", "raw", "lr_to_hr")
        ]
        checkpoint_delta = groups[
            ("paired_differences", "checkpoint", "raw", "lr_to_hr")
        ]
        self.assertEqual(processed_delta["resolution"], 128)
        self.assertTrue(
            torch.allclose(
                torch.tensor(processed_delta["parameters"]["means"]["mean"]),
                torch.full((3,), 0.2),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(checkpoint_delta["parameters"]["means"]["mean"]),
                torch.full((3,), 0.1),
            )
        )
        self.assertAlmostEqual(
            processed_delta["parameters"]["scales"]["mean"][0],
            torch.log(torch.tensor(2.0)).item(),
            places=6,
        )
        activated_delta = groups[
            ("paired_differences", "processed", "post_activation", "lr_to_hr")
        ]
        self.assertAlmostEqual(
            activated_delta["parameters"]["opacities"]["mean"][0],
            torch.sigmoid(torch.tensor(1.0)).item() - 0.5,
            places=6,
        )

    def test_statistics_resume_force_and_all_stage_order(self):
        args = self._statistics_args()
        calls = []

        def process(unused_args, scene, fingerprint):
            calls.append(scene)
            return {
                "record_version": 1,
                "scene": scene,
                "status": "ok",
                "input_fingerprint": fingerprint,
                "groups": [],
                "missing_caches": [],
                "errors": [],
            }

        with mock.patch.object(
            preprocess, "statistics_input_fingerprint", return_value={"same": True}
        ), mock.patch.object(
            preprocess, "process_statistics_scene", side_effect=process
        ):
            preprocess.run_statistics(args)
            preprocess.run_statistics(args)
            self.assertEqual(calls, ["selected_scene"])
            args.force_statistics = True
            preprocess.run_statistics(args)
            self.assertEqual(calls, ["selected_scene", "selected_scene"])

        order = []
        stages = (
            "run_validate",
            "run_evaluate",
            "run_select",
            "run_pretrain",
            "run_statistics",
        )
        patchers = [
            mock.patch.object(
                preprocess, name, side_effect=lambda *unused, _name=name: order.append(_name)
            )
            for name in stages
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        preprocess.run_all(SimpleNamespace())
        self.assertEqual(order, list(stages))

    def test_statistics_cli_defaults_and_output_schema(self):
        parser = preprocess.build_parser()
        args = parser.parse_args(
            ["statistics", "--dataset_root", str(self.root)]
        )
        preprocess.validate_arguments(args, parser)
        self.assertFalse(args.force_statistics)
        self.assertFalse(args.retry_errors)
        self.assertEqual(args.resolutions, [512, 128])
        report = preprocess.aggregate_statistics_records([], 0)
        for space in ("checkpoint", "processed"):
            for representation in ("raw", "post_activation"):
                self.assertIn(representation, report["spaces"][space])


if __name__ == "__main__":
    unittest.main()

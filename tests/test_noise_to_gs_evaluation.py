"""Checks for comparable EMA and current-model evaluation outputs."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_noise_to_gs_evaluation", ROOT / "train-noise-to-gs.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class EvaluationTests(unittest.TestCase):
    def test_loads_all_reference_views_but_only_eight_preview_views(self):
        entry = {"resolution_paths": {128: {}}}
        dataset = mock.Mock()
        dataset.folders = [entry]
        dataset.tgt_resolution = 128
        dataset.scene_index.return_value = 0
        dataset.processor.selection_mask.return_value = torch.tensor([True])
        dataset.processor.normalize.return_value = ({}, None)
        dataset.build_background.return_value = torch.zeros(3)
        dataset.load_camera_meta.return_value = {
            "meta": {"camera_to_worlds": torch.arange(10)},
            "imgs_name": [f"{index}.png" for index in range(10)],
        }
        dataset.load_resolution_views.return_value = ([torch.zeros(2, 2, 3) for _ in range(10)], [], {})
        with mock.patch.object(SCRIPT.gs_io, "load_gsplat", return_value=({"means": torch.zeros(1, 3)}, None)):
            _, preview_indices, _, = SCRIPT.load_evaluation_cameras(dataset, "scene", "/gs", "cpu")
            cameras, indices, names, images = SCRIPT.load_evaluation_cameras(dataset, "scene", "/gs", "cpu", load_images=True)
        self.assertEqual(len(preview_indices), 8)
        self.assertEqual(indices, list(range(10)))
        self.assertEqual(cameras["camera_to_worlds"].tolist(), indices)
        self.assertEqual(len(names), len(images))
        self.assertEqual(dataset.load_resolution_views.call_args.kwargs["camera_ids"], indices)

    def test_single_scene_outputs_sample_metrics_and_baseline(self):
        from utils import metrics

        with tempfile.TemporaryDirectory() as temporary:
            flags = SimpleNamespace(output_dir=temporary, eval_scene_name="preview", gs_count=3, eval_sample_count=2)
            flow = {"gs_statistics_path": "stats.json", "flow_steps": 2, "eval_noise_seed": 17,
                    "flow_t_eps": 0.001, "coupling": "random", "model_output_type": "x"}
            model = torch.nn.Linear(1, 1)
            model.sh_degree = 0
            baseline = {"path": "val_step14999.json", "metrics": {"psnr": 25.0}}
            context = {"scene": "train", "cameras": {"camera_to_worlds": torch.arange(10)},
                       "images": torch.zeros(10, 2, 2, 3),
                       "names": [f"train_{i}.png" for i in range(10)], "baseline": baseline}
            computer = mock.Mock()
            computer.metrics = {"psnr": None, "ssim": None, "lpips": None}
            computer.finalize.return_value = {"psnr": 20.0, "ssim": 0.8, "lpips": 0.1}
            def render(_gaussians, cameras, **_kwargs):
                count = len(cameras.get("camera_to_worlds", [0]))
                return list(torch.zeros(count, 2, 2, 3).unbind(0)), None

            with mock.patch.object(SCRIPT, "FLAGS", flags), mock.patch.object(SCRIPT, "wandb", None), \
                    mock.patch.object(SCRIPT.torch.random, "fork_rng", return_value=nullcontext()), \
                    mock.patch.object(SCRIPT.torch.cuda, "device", return_value=nullcontext()), \
                    mock.patch.object(metrics, "MetricComputer", return_value=computer), \
                    mock.patch.object(SCRIPT.noise_to_gs, "sample_flow_model", return_value={}), \
                    mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer"), \
                    mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs", side_effect=render):
                SCRIPT.evaluate(model, None, {}, [0], ["preview.png"], flow, 12, 0, 1, metric_context=context)
            directory = Path(temporary) / "eval" / "00000012"
            report = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(report["reference_scene"], "train")
            self.assertEqual(report["prediction_mean"]["psnr"], 20.0)
            self.assertEqual(report["gsplat_baseline"], baseline)
            self.assertEqual(json.loads((directory / "sample_01" / "metrics.json").read_text())["prediction"]["ssim"], 0.8)
            self.assertEqual(computer.update.call_count, 4)
            self.assertEqual([call.args[0].shape[0] for call in computer.update.call_args_list], [8, 2, 8, 2])
            self.assertEqual(len(report["camera_names"]), 10)

    def test_ema_and_model_use_matching_seeds_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            flags = SimpleNamespace(output_dir=temporary, eval_scene_name="scene", gs_count=3, eval_sample_count=2)
            flow = {"gs_statistics_path": "stats.json", "flow_steps": 2, "eval_noise_seed": 17,
                    "flow_t_eps": 0.001, "coupling": "random", "model_output_type": "x"}
            ema, model = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
            ema.sh_degree = model.sh_degree = 0
            ema.eval()
            model.train()
            wandb = mock.Mock()
            wandb.run = True
            with mock.patch.object(SCRIPT, "FLAGS", flags), mock.patch.object(SCRIPT, "wandb", wandb), \
                    mock.patch.object(SCRIPT.torch.random, "fork_rng", return_value=nullcontext()), \
                    mock.patch.object(SCRIPT.noise_to_gs, "sample_flow_model", return_value={}) as sample, \
                    mock.patch.object(SCRIPT.gs_utils, "export_ply_forviewer"), \
                    mock.patch.object(SCRIPT.gs_utils, "rasterize_gaussians_to_multiimgs",
                                      return_value=(torch.zeros(1, 2, 2, 3), None)):
                for weights, variant in ((ema, "ema"), (model, "model")):
                    SCRIPT.evaluate(weights, None, {}, [0], ["view"], flow, 12, 0, 1, variant=variant)
            directory = Path(temporary) / "eval" / "00000012"
            ema_meta = json.loads((directory / "samples.json").read_text())
            model_meta = json.loads((directory / "model" / "samples.json").read_text())
            self.assertEqual(ema_meta["sample_seeds"], [17, 18])
            self.assertEqual(model_meta["sample_seeds"], ema_meta["sample_seeds"])
            self.assertEqual((ema_meta["weights"], model_meta["weights"]), ("ema", "model"))
            self.assertTrue((directory / "sample_00" / "view_00.png").exists())
            self.assertTrue((directory / "model" / "sample_00" / "view_00.png").exists())
            self.assertEqual([call.kwargs["noise_seed"] for call in sample.call_args_list], [17, 18, 17, 18])
            self.assertEqual([next(iter(call.args[0])) for call in wandb.log.call_args_list],
                             ["eval/samples", "eval/samples_model"])
            self.assertEqual([call.kwargs["commit"] for call in wandb.log.call_args_list], [False, True])
            self.assertFalse(ema.training)
            self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

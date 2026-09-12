"""Run with the training environment: python -m unittest discover -s tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import torch
from absl.testing import flagsaver

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_sr_gsfm", ROOT / "train-sr-gsfm.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if not m.FLAGS.is_parsed():
    m.FLAGS(["test"])
KEYS = m.SUPPORTED_GS_KEYS


def make_scene(i, images=True):
    generator = torch.Generator().manual_seed(100 + i)
    n = 64 + 16 * i
    source = {
        'means': torch.rand(n, 3, generator=generator) - 0.5,
        'features_dc': torch.rand(n, 3, generator=generator) * 0.1,
        'features_rest': torch.rand(n, 3, 3, generator=generator) * 0.01,
        'opacities': torch.zeros(n, 1), 'scales': torch.full((n, 3), -3.),
        'quats': torch.tensor([1., 0., 0., 0.]).repeat(n, 1),
    }
    source['means'][:, 2] -= 2
    target = {k: v + 0.02 * torch.rand(v.shape, generator=generator) for k, v in source.items()}
    target_data = {'gs_params': target}
    if images:
        target_data.update({
            'images': [torch.rand(16, 16, 3, generator=generator)], 'images_name': ['view.png'],
            'cameras': {'camera_to_worlds': torch.eye(4)[None], 'fx': torch.tensor(16.), 'fy': torch.tensor(16.), 'cx': torch.tensor(8.), 'cy': torch.tensor(8.),
                        'width': torch.tensor(16), 'height': torch.tensor(16), 'background_color': torch.zeros(3)},
        })
    return {'scene_idx': i, 'scene_name': f'scene{i}', 'data': {128: {'gs_params': source}, 512: target_data},
            'fit_lr_to_hr': {'tgt_gs': target}}


class Toy(torch.nn.Module):
    apply_feature_update = m.GSFlowPredictor.apply_feature_update
    quat_residual_mode = 'mul'

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.tensor(.1)) for key in KEYS})
        self.calls = []

    def forward(self, batch_flow_gs, batch_scene_idx, batch_reference_means, t):
        self.calls.append((list(batch_scene_idx), t.detach().clone()))
        return [{key: value * self.weights[key] + t[i] * self.weights[key] for key, value in gs.items()}
                for i, gs in enumerate(batch_flow_gs)]


class MicrobatchTests(unittest.TestCase):
    @flagsaver.flagsaver(alignment="fit_lr_to_hr")
    def test_gradients_and_statistics_match_unsplit_batch(self):
        dataset = SimpleNamespace(src_resolution=128, tgt_resolution=512, image_per_scene=1)
        config = {"enable_amp": False, "image_l1_loss_weight": 1., "lpips_loss_weight": 0.}
        variances = {key: torch.ones(()) for key in KEYS}
        for objective in ("velocity", "x1"):
            flow_config = {"flow_t_eps": 1e-4, "flow_noise_std": 0., "loss_type": objective}
            for schedule in ("fm-only", "linear", "free-range-gs"):
                for count, partitions in ((1, ([1],)), (4, ([4], [2, 2], [1] * 4)), (5, ([5], [3, 2], [1] * 5))):
                    scenes = [make_scene(i, schedule != "fm-only") for i in range(count)]
                    original_pairs = [{name: {key: value.clone() for key, value in params.items()} for name, params in (("source", scene["data"][128]["gs_params"]), ("target", scene["fit_lr_to_hr"]["tgt_gs"]))} for scene in scenes]
                    baseline = None
                    for sizes in partitions:
                        with self.subTest(objective=objective, schedule=schedule, sizes=sizes):
                            model = Toy()
                            torch.manual_seed(12)
                            np.random.seed(12)
                            offset = 0
                            stats = {}
                            # A differentiable renderer stand-in keeps this regression check on CPU.
                            with patch.object(m.gs_utils, "rasterize_gaussians_to_multiimgs", side_effect=lambda gs, cameras: ([gs["features_dc"].mean(0).expand(16, 16, 3)], None)) as render:
                                for size in sizes:
                                    loss, values = m.compute_microbatch_loss(
                                        model, dataset, scenes[offset:offset + size], "cpu", flow_config,
                                        {"schedule": schedule}, m.feature_mse_loss(), config, variances, None,
                                    )
                                    (loss / count).backward()
                                    for key, value in values.items():
                                        self.assertIsInstance(value, float)
                                        stats[key] = stats.get(key, 0.) + value / count
                                    offset += size
                                self.assertEqual(render.call_count, 0 if schedule == "fm-only" else count)
                            grads = torch.stack([p.grad for p in model.parameters()])
                            self.assertTrue(torch.isfinite(grads).all())
                            if baseline is None:
                                baseline = grads, stats
                            else:
                                torch.testing.assert_close(grads, baseline[0], atol=2e-6, rtol=2e-5)
                                for key in stats:
                                    self.assertTrue(np.isclose(stats[key], baseline[1][key], rtol=2e-5, atol=1e-6), key)
                            for scene, original in zip(scenes, original_pairs):
                                for name, params in (("source", scene["data"][128]["gs_params"]), ("target", scene["fit_lr_to_hr"]["tgt_gs"])):
                                    for key in KEYS:
                                        torch.testing.assert_close(params[key], original[name][key], rtol=0, atol=0)
                            self.assertEqual([len(ids) for ids, _ in model.calls], sizes)
                            self.assertEqual(torch.cat([t for _, t in model.calls]).unique().numel(), count)


class MemoryTests(unittest.TestCase):
    @flagsaver.flagsaver(alignment="fit_lr_to_hr")
    def test_fitted_pair_does_not_transfer_original_target(self):
        scene = make_scene(0)
        scene["fit_lr_to_hr"]["tgt_gs"] = {key: value.clone() for key, value in scene["data"][512]["gs_params"].items()}
        dataset = SimpleNamespace(src_resolution=128, tgt_resolution=512)
        with patch.object(m.gpu_utils, "move_to_device", side_effect=lambda data, device: data) as move:
            _, source, target = m.build_gaussian_pair(dataset, scene, "cpu")
        self.assertEqual(move.call_count, 2)
        self.assertIs(source, scene["data"][128]["gs_params"])
        self.assertIs(target, scene["fit_lr_to_hr"]["tgt_gs"])
        self.assertFalse(any(call.args[0] is scene["data"][512]["gs_params"] for call in move.call_args_list))

    @flagsaver.flagsaver(alignment="fit_lr_to_hr", compare_with_input=False, save_viewer=False)
    def test_dataset_passes_chunk_limit_and_reuses_source(self):
        scene = make_scene(0)
        scene["data"][512]["images"] *= 5
        dataset = SimpleNamespace(src_resolution=128, tgt_resolution=512, folders=[0], load_scene=lambda *args, **kwargs: scene)
        for limit, expected in ((2, 2), (None, 5), (0, 5), (-1, 5)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as output:
                dataset.image_per_scene = limit
                with patch.object(m, "evaluate_single_scene", return_value=({"score": 1.}, {})) as evaluate:
                    result = m.evaluate_dataset(Toy(), dataset, output)
                self.assertEqual(result, {10: {"score": 1.}})
                for call in evaluate.call_args_list:
                    self.assertEqual(call.kwargs["eval_chunk_size"], expected)
                    self.assertEqual(len(call.kwargs["eval_images"]), 5)
                    self.assertIs(call.kwargs["source_flow_gs"], call.kwargs["input_gs"])

    def test_chunked_evaluation_keeps_every_view(self):
        source = make_scene(0)["data"][128]["gs_params"]
        cameras = {"camera_to_worlds": torch.arange(5.).reshape(5, 1, 1)}
        images = [torch.full((16, 16, 3), i / 10.) for i in range(5)]
        names = [f"view{i}.png" for i in range(5)]
        outputs = []
        for chunk_size, expected_sizes in ((5, [5]), (2, [2, 2, 1])):
            metric = Mock()
            metric.finalize.return_value = {"score": 1.}
            with tempfile.TemporaryDirectory() as output, \
                 patch.object(m, "MetricComputer", return_value=metric), \
                 patch.object(m.flow, "sample_flow_model", return_value=source), \
                 patch.object(m.gs_utils, "rasterize_gaussians_to_multiimgs", side_effect=lambda gs, cams: ([torch.full((16, 16, 3), camera.item() / 10.) for camera in cams["camera_to_worlds"]], None)) as render, \
                 patch.object(m.cv2, "imwrite", return_value=True) as write:
                m.evaluate_single_scene(Toy(), source, source, 0, "scene0", images, cameras, names, output, 10, eval_chunk_size=chunk_size, save_viewer=False)
            self.assertEqual([len(call.args[1]["camera_to_worlds"]) for call in render.call_args_list], expected_sizes)
            saved_names = [Path(call.args[0]).name for call in write.call_args_list]
            self.assertEqual([name for name in saved_names if name.startswith("view")], names)
            outputs.append(torch.cat([call.args[0] for call in metric.update.call_args_list]))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

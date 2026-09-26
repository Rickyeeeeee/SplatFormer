"""Configured augmentation controls, artifacts, and optional CUDA render invariance."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from sr import data_augmentation as diagnostic
from utils.data_augmentation import rotate_camera_to_worlds, rotate_gaussians, quaternion_inverse
from utils.gs_normalization import normalize_gaussian_quaternions


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.gs = {
            "means": torch.tensor([[.3, .4, .5], [.7, .6, .5]]),
            "quats": torch.tensor([[2., .3, .1, .2], [-.4, .2, 1., .3]]),
            "scales": torch.tensor([[-3., -4., -3.5], [-3.5, -3., -4.]]),
            "opacities": torch.ones(2, 1), "features_dc": torch.ones(2, 3) * .2,
            "features_rest": torch.arange(18).reshape(2, 3, 3).float() / 100,
        }
        camera = torch.eye(4)[None]
        camera[0, :3, 3] = torch.tensor([.5, .5, 2.])
        self.cameras = {"camera_to_worlds": camera}

    def args(self, *extra):
        return diagnostic.parse_args(["--gin_file", "config.gin", "--mode", "configured", *extra])

    def test_controls_order_seed_and_nonmutation(self):
        normalized = normalize_gaussian_quaternions(self.gs)
        original = {k: v.clone() for k, v in self.gs.items()}
        for mode in ("full", "gravity_consistent"):
            for jitter in ("False", "True"):
                for rotate in ("False", "True"):
                    args = self.args("--random_jitter", jitter, "--random_rotate", rotate, "--rotation_mode", mode)
                    first = diagnostic.augment_trial(normalized, self.cameras, args, torch.Generator().manual_seed(42))
                    second = diagnostic.augment_trial(normalized, self.cameras, args, torch.Generator().manual_seed(42))
                    augmented, rotated, cameras, rotation, levels = first
                    for key in self.gs:
                        torch.testing.assert_close(augmented[key], second[0][key])
                        torch.testing.assert_close(self.gs[key], original[key])
                    generator = torch.Generator().manual_seed(42)
                    if rotate == "True":
                        sampler = diagnostic.sample_uniform_z_rotation_quaternion if mode == "gravity_consistent" else diagnostic.sample_uniform_rotation_quaternion
                        expected_rotation = sampler(torch.float32, "cpu", generator)
                        torch.testing.assert_close(rotation, expected_rotation)
                        torch.testing.assert_close(cameras["camera_to_worlds"], rotate_camera_to_worlds(self.cameras["camera_to_worlds"], rotation, args.rotation_pivot))
                        expected = rotate_gaussians(normalized, rotation, args.rotation_pivot)
                    else:
                        expected = normalized
                    if jitter == "True":
                        expected, expected_levels = diagnostic.jitter_gaussian_parameters(expected, args.jitter_max_levels, generator)
                        self.assertEqual(levels, expected_levels)
                    for key in self.gs:
                        torch.testing.assert_close(augmented[key], expected[key])
                    torch.testing.assert_close(rotated["quats"].norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(self.cameras["camera_to_worlds"][0, :3, :3], torch.eye(3))

    def test_zero_jitter_and_raw_float_errors(self):
        args = self.args("--random_jitter", "True", "--jitter_max_levels", "{'means': 0}")
        source = normalize_gaussian_quaternions(self.gs)
        augmented, _, _, _, _ = diagnostic.augment_trial(source, self.cameras, args, torch.Generator())
        for key in source:
            torch.testing.assert_close(augmented[key], source[key])
        image = torch.zeros(4, 4, 3)
        result = diagnostic.render_difference([image + .001], [image], args)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["mae"], .001)
        self.assertAlmostEqual(result["psnr"], 60., places=4)
        self.assertTrue(diagnostic.render_difference([image], [image], args)["passed"])
        self.assertFalse(diagnostic.render_difference([image * float('nan')], [image], args)["passed"])

    def test_checkpoint_reload_and_ply_export(self):
        exporter = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.pt"
            with mock.patch.dict(sys.modules, {"utils.gs_utils": SimpleNamespace(export_ply_forviewer=exporter)}):
                diagnostic.save_gaussian_artifact(path, self.gs, self.cameras, {"seed": 42})
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["artifact_type"], "gaussian_scene")
            self.assertEqual(checkpoint["metadata"]["seed"], 42)
            for key in self.gs:
                torch.testing.assert_close(checkpoint["gs_params"][key], self.gs[key])
            torch.testing.assert_close(checkpoint["cameras"]["camera_to_worlds"], self.cameras["camera_to_worlds"])
            self.assertEqual(exporter.call_args.args[1], path.with_suffix('.ply'))

    def test_failed_checks_still_write_trial_artifacts_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args('--device', 'cpu', '--preview_views', '0', '--output_dir', directory,
                             '--random_rotate', 'True', '--random_jitter', 'True')
            dataset = SimpleNamespace(src_resolution=32, tgt_resolution=128)
            scene = {'scene_name': 'test', 'scene_idx': 0, 'coordinate_frame': 'input_resolution',
                     'data': {128: {'images_name': ['view.png']}}}
            targets = [torch.zeros(4, 4, 3)]
            evaluator = mock.Mock()
            evaluator.evaluate.return_value = {'psnr': 20., 'ssim': .9, 'lpips': .1}
            # Deliberately fail normalization, then finish all three trials.
            rendered = [targets, [torch.ones(4, 4, 3)]] + [targets] * 9
            exporter = mock.Mock()
            fake_gpu = SimpleNamespace(move_to_device=lambda value, device: value)
            with mock.patch.dict(sys.modules, {'utils.gs_utils': SimpleNamespace(export_ply_forviewer=exporter)}), \
                 mock.patch('utils.gpu_utils', fake_gpu, create=True), \
                 mock.patch.object(diagnostic, '_load_experiment', return_value=(dataset, scene, self.gs, targets, self.cameras)), \
                 mock.patch.object(diagnostic, '_ImageMetricEvaluator', return_value=evaluator), \
                 mock.patch.object(diagnostic, '_render_views', side_effect=rendered):
                report = diagnostic.run(args)
            self.assertFalse(report['passed'])
            self.assertEqual(len(report['trials']), 3)
            saved = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertFalse(saved['passed'])
            for index in range(3):
                checkpoint = torch.load(Path(directory) / f'trial_{index:03d}.pt', weights_only=True)
                self.assertEqual(checkpoint['metadata']['trial'], index)
            with mock.patch.object(diagnostic, 'run', return_value=report):
                with self.assertRaises(SystemExit) as failure:
                    diagnostic.main(['--gin_file', 'config.gin', '--mode', 'configured'])
                self.assertEqual(failure.exception.code, 1)

    def test_launcher_controls_and_legacy_mode(self):
        self.assertEqual(diagnostic.parse_args(["--gin_file", "x"]).mode, "sweep")
        launcher = Path(__file__).resolve().parents[1] / 'scripts/test-data-augmentation.sh'
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        env = dict(os.environ, RANDOM_ROTATE="True", RANDOM_JITTER="True", ROTATION_MODE="gravity_consistent", PYTHON="python",
                   JITTER_MAX_LEVELS="{'means': 0.02}")
        result = subprocess.run(['bash', '-c', command, 'test', str(launcher)], env=env, text=True, capture_output=True, check=True)
        for text in ('--mode\nconfigured', '--random_rotate\nTrue', '--random_jitter\nTrue', '--rotation_mode\ngravity_consistent', "{'means': 0.02}"):
            self.assertIn(text, result.stdout)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('gsplat') is not None, 'CUDA and gsplat are required for render invariance')
    def test_cuda_render_invariance(self):
        from gsplat import rasterization
        source = {k: v.cuda() for k, v in self.gs.items()}
        cameras = self.cameras['camera_to_worlds'].cuda()
        normalized = normalize_gaussian_quaternions(source)
        args = self.args('--random_rotate', 'True')
        generator = torch.Generator(device='cuda').manual_seed(42)
        for mode in ('full', 'gravity_consistent'):
            args.rotation_mode = mode
            for _ in range(3):
                _, rotated, rotated_cameras, rotation, _ = diagnostic.augment_trial(normalized, {'camera_to_worlds': cameras}, args, generator)
                inverse = quaternion_inverse(rotation)
                restored = rotate_gaussians(rotated, inverse, args.rotation_pivot)
                restored_cameras = rotate_camera_to_worlds(rotated_cameras['camera_to_worlds'], inverse, args.rotation_pivot)
                renders = []
                for gs, poses in ((source, cameras), (normalized, cameras), (rotated, rotated_cameras['camera_to_worlds']), (restored, restored_cameras)):
                    # Convert OpenGL camera poses to gsplat's OpenCV view matrices.
                    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda'))
                    view = flip @ torch.linalg.inv(poses)
                    colors = torch.cat((gs['features_dc'][:, None], gs['features_rest']), dim=1)
                    image, _, _ = rasterization(means=gs['means'], quats=gs['quats'], scales=gs['scales'].exp(),
                        opacities=gs['opacities'].sigmoid().flatten(), colors=colors, viewmats=view,
                        Ks=torch.tensor([[[70., 0., 32.], [0., 70., 32.], [0., 0., 1.]]], device='cuda'),
                        width=64, height=64, sh_degree=1, packed=True)
                    renders.append(image[0].detach().cpu())
                self.assertGreater(renders[0].max().item(), .01)
                for image in renders[1:]:
                    result = diagnostic.render_difference([image], [renders[0]], args)
                    self.assertTrue(result['passed'], result)


if __name__ == '__main__':
    unittest.main()

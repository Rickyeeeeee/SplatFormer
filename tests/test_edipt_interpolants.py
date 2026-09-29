"""Manifold paths, launcher wiring, and EDiPT coordinate adapter coverage."""
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sr import edipt_interpolants as flow
from sr.flow import loss_mix_weights
from utils import data_augmentation, gpu_utils
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS
from utils.rotation_flow import (body_to_quaternion_derivative, integrate_rotation, quaternion_derivative_to_body,
                                 relative_rotation, rotation_exp, rotation_log)

ROOT = Path(__file__).resolve().parents[1]


class ConstantField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angular = torch.nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.last_geometry = None

    def forward(self, batch_flow_gs, batch_scene_idx, batch_geometry, batch_reference_means, t):
        self.last_geometry = batch_geometry
        result = []
        for state, geometry in zip(batch_flow_gs, batch_geometry):
            output = {key: torch.zeros_like(value) + self.angular[0] for key, value in state.items()}
            output['means'] = torch.ones_like(geometry['means']) * self.angular[0]
            output['quats'] = body_to_quaternion_derivative(geometry['quats'], self.angular.expand(len(geometry['quats']), 3))
            result.append(output)
        return result


class ManifoldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        shapes = {'means': (3,), 'quats': (4,), 'scales': (3,), 'opacities': (1,), 'features_dc': (3,), 'features_rest': (3, 3)}
        statistics = {stored: {'mean': torch.full(shapes[key], .2).tolist(), 'variance': torch.full(shapes[key], 4.).tolist()}
                      for key, stored in STATISTIC_KEYS.items()}
        path = Path(self.temp.name) / 'statistics.json'
        path.write_text(json.dumps({'aggregate': {'normalized': {'output': statistics}}}))
        self.standardizer = GaussianStandardizer(path, quaternion_representation='unit_unstandardized')
        source = {k: torch.rand((4,) + shape) for k, shape in shapes.items()}
        target = {k: torch.rand((4,) + shape) for k, shape in shapes.items()}
        self.source, self.target = self.standardizer.prepare_endpoints(source, target, align_target_sign=False)
        self.x0, self.x1 = self.standardizer.encode(self.source), self.standardizer.encode(self.target)
        self.noise = flow.seeded_noise_like(self.x1, 12)

    def test_exp_log_degeneracies_and_derivative(self):
        vectors = torch.tensor([[0., 0., 0.], [1e-8, 0., 0.], [.2, -.4, .1], [math.pi - 1e-5, 0., 0.]], requires_grad=True)
        q = rotation_exp(vectors)
        torch.testing.assert_close(rotation_log(q), vectors)
        torch.testing.assert_close(rotation_log(-q), vectors)
        pi = torch.tensor([[0., -1., 0., 0.]])
        torch.testing.assert_close(rotation_log(pi), torch.tensor([[math.pi, 0., 0.]]))
        omega = torch.randn(4, 3)
        torch.testing.assert_close(quaternion_derivative_to_body(q, body_to_quaternion_derivative(q, omega)), omega)
        rotation_log(q).square().sum().backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_paths_and_analytic_velocities(self):
        q0, q1 = self.source['quats'].double(), self.target['quats'].double()
        noise = self.noise['rotation'].double()
        for mode in flow.MODES:
            for time in (.15, .37, .5, .73):
                t = torch.tensor(time, dtype=torch.float64)
                q, omega = flow.rotation_path(q0, q1, t, mode, noise, .3)
                h = 1e-5
                before = flow.rotation_path(q0, q1, t-h, mode, noise, .3)[0]
                after = flow.rotation_path(q0, q1, t+h, mode, noise, .3)[0]
                numerical = quaternion_derivative_to_body(q, (after-before)/(2*h))
                torch.testing.assert_close(omega, numerical, atol=3e-4, rtol=1e-3)
                torch.testing.assert_close(q.norm(dim=-1), torch.ones(4, dtype=torch.float64))
            for time in (0., 1.):
                # Zero latent noise removes endpoint singularity for this endpoint check.
                sigma = 0. if mode == 'latent' else .3
                q, _ = flow.rotation_path(q0, q1, torch.tensor(time), mode, noise, sigma)
                expected = q1 if time else (rotation_exp(sigma * noise) if mode == 'one_sided' else q0)
                torch.testing.assert_close(relative_rotation(q, expected), torch.zeros(4, 3, dtype=q.dtype), atol=1e-6, rtol=0)
        left = flow.rotation_path(q0, q1, torch.tensor(.5-1e-7), 'encoding_decoding', noise, .3)[0]
        mid, omega = flow.rotation_path(q0, q1, torch.tensor(.5), 'encoding_decoding', noise, .3)
        right = flow.rotation_path(q0, q1, torch.tensor(.5+1e-7), 'encoding_decoding', noise, .3)[0]
        torch.testing.assert_close(left, mid, atol=1e-6, rtol=0)
        torch.testing.assert_close(right, mid, atol=1e-6, rtol=0)
        self.assertEqual(omega.count_nonzero(), 0)

    def test_euclidean_parity_noise_and_adapter(self):
        from sr import interpolants as original
        rng = torch.random.get_rng_state()
        repeated = flow.seeded_noise_like(self.x1, 12)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(repeated['rotation'], self.noise['rotation'])
        for mode in flow.MODES:
            path = flow.construct_path(self.x0, self.x1, .3, mode, self.source['means'], self.target['means'], noise=self.noise)
            old_noise = {**self.noise['euclidean'], 'quats': torch.zeros_like(self.x1['quats'])}
            baseline = original.construct_path(self.x0, self.x1, .3, mode, self.source['means'], self.target['means'], noise=old_noise)
            changed = flow.construct_path(self.x0, self.x1, .3, mode, self.source['means'], self.target['means'], noise_scale=.1,
                                          noise=self.noise, rotation_noise_std=.3)
            for key in self.noise['euclidean']:
                torch.testing.assert_close(path['query'][key], baseline['query'][key])
                torch.testing.assert_close(path['velocity'][key], baseline['velocity'][key])
            torch.testing.assert_close(changed['query']['quats'], path['query']['quats'])
        model = ConstantField()
        prediction = flow.predict_velocity(model, [self.x0], [0], [self.source['means']], .2, self.standardizer)[0]
        torch.testing.assert_close(model.last_geometry[0]['means'], self.source['means'])
        torch.testing.assert_close(prediction['means'], torch.full((4, 3), .05))
        torch.testing.assert_close(prediction['quats'], model.angular.expand(4, 3))

    def test_rollout_losses_and_evaluation(self):
        model = ConstantField()
        for mode in flow.MODES:
            steps = 4
            result = flow.rollout(model, self.x0, 0, steps, mode, self.source['means'], self.target['means'], self.noise,
                                  standardizer=self.standardizer)
            start = rotation_exp(.3*self.noise['rotation']) if mode == 'one_sided' else self.x0['quats']
            expected = integrate_rotation(start, model.angular, 1.)
            torch.testing.assert_close(result['quats'], expected)
            loss, _, _ = flow.attribute_mse(result, self.x1, 'x1', 2.)
            loss.backward()
            self.assertTrue(torch.isfinite(model.angular.grad).all())
            for count in flow.evaluation_steps(mode):
                flow.validate_settings(mode, count)
        flipped = {**self.x1, 'quats': -self.x1['quats']}
        torch.testing.assert_close(flow.attribute_mse(self.x0, self.x1, 'x1')[0], flow.attribute_mse(self.x0, flipped, 'x1')[0])
        _, raw, weighted = flow.attribute_mse(self.x0, self.x1, 'x1', 3.)
        torch.testing.assert_close(weighted['quats'], 3 * raw['quats'])

    def test_microbatch_all_modes_objectives_and_augmentation(self):
        tree = ast.parse((ROOT / 'overfit-sr-interpolants-edipt.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_microbatch_loss')
        namespace = {'torch': torch, 'np': np, 'gpu_utils': gpu_utils, 'interpolants': flow,
                     'flow': SimpleNamespace(loss_mix_weights=loss_mix_weights),
                     'gs_utils': SimpleNamespace(rasterize_gaussians_to_multiimgs=lambda gs, cameras: ([gs['means'].mean().expand(2, 2, 3)], None))}
        for name in ('sample_uniform_z_rotation_quaternion', 'sample_uniform_rotation_quaternion',
                     'rotate_gaussians', 'rotate_camera_to_worlds', 'jitter_gaussian_parameters'):
            namespace[name] = getattr(data_augmentation, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<edipt-microbatch>', 'exec'), namespace)
        compute = namespace['compute_microbatch_loss']
        scene = {'source_gs': self.source, 'target_gs': self.target, 'source_flow_gs': self.x0, 'target_flow_gs': self.x1,
                 'scene_idx': 0, 'fixed_train_noise': self.noise, 'target_images': [torch.zeros(2, 2, 3)],
                 'target_cameras': {'camera_to_worlds': torch.eye(4)[None]}, 'render_view_count': 1}
        for mode in flow.MODES:
            for objective in ('velocity', 'x1'):
                cfg = {'interpolant_type': mode, 'flow_t_eps': 1e-4, 'fixed_train_noise': True, 'flow_noise_std': .2,
                       'rotation_noise_std': .3, 'rotation_loss_weight': 1., 'loss_type': objective, 'loss_rollout_steps': 4}
                augmentation = {'random_jitter': mode != 'one_sided', 'random_rotate': True, 'rotation_mode': 'full',
                                'rotation_pivot': (0., 0., 0.), 'rotation_max_degrees': 10., 'jitter_max_levels': {'quats': .01}}
                model = ConstantField()
                loss, stats = compute(model, [scene, scene], 'cpu', cfg, {'schedule': 'linear'}, self.standardizer,
                                      1., 1., lambda x,y: (x-y).square().mean(), False, augmentation)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(model.angular.grad).all())
                self.assertIn('quats_weighted', stats)
                # Effective-batch averaging agrees with separate microbatch accumulation.
                a, b = ConstantField(), ConstantField()
                torch.manual_seed(99)
                compute(a, [scene, scene], 'cpu', cfg, {'schedule': 'fm-only'}, self.standardizer, 0., 0., None, False)[0].div(2).backward()
                torch.manual_seed(99)
                for _ in range(2):
                    compute(b, [scene], 'cpu', cfg, {'schedule': 'fm-only'}, self.standardizer, 0., 0., None, False)[0].div(2).backward()
                torch.testing.assert_close(a.angular.grad, b.angular.grad, atol=1e-5, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_edipt_training_objectives(self):
        import gin
        from models.equivariant_gaussian_dipt_predictor import EquivariantGaussianDiPTPredictor
        gin.clear_config()
        self.addCleanup(gin.clear_config)
        gin.parse_config("""
EquivariantGaussianDiPT.depth = 1
EquivariantGaussianDiPT.channels = 12
EquivariantGaussianDiPT.num_head = 3
EquivariantGaussianDiPT.patch_size = 3
EquivariantGaussianDiPT.frequency_embedding_size = 8
""")
        source = {k: v.cuda() for k, v in self.x0.items()}
        target = {k: v.cuda() for k, v in self.x1.items()}
        source_means, target_means = self.source['means'].cuda(), self.target['means'].cuda()
        model = EquivariantGaussianDiPTPredictor().cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for mode in flow.MODES:
            for objective in ('velocity', 'x1'):
                optimizer.zero_grad(set_to_none=True)
                path = flow.construct_path(source, target, .3, mode, source_means, target_means)
                with torch.autocast('cuda', dtype=torch.float16):
                    if objective == 'velocity':
                        prediction = flow.predict_velocity(model, [path['query']], [0], [path['reference_means']], .3, self.standardizer)[0]
                        loss = flow.attribute_mse(prediction, path['velocity'])[0]
                    else:
                        prediction = flow.rollout(model, source, 0, 4, mode, source_means, target_means, path['noise'], standardizer=self.standardizer)
                        loss = flow.attribute_mse(prediction, target, 'x1')[0]
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()

    def test_shell_arguments(self):
        executable = Path(self.temp.name) / 'python'
        executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o755)
        env = {**os.environ, 'PATH': self.temp.name + ':' + os.environ['PATH'], 'EDIPT_DEPTH': '2', 'GS_SH_DEGREE': '1',
               'ROTATION_NOISE_STD': '.4', 'ROTATION_LOSS_WEIGHT': '2', 'QUATERNION_REPRESENTATION': 'unit_unstandardized'}
        command = ['bash', 'scripts/overfit-sr-interpolants-edipt.sh', '2', '1', '1', '1']
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        self.assertIn('overfit-sr-interpolants-edipt.py', result.stdout)
        self.assertIn('EquivariantGaussianDiPT.depth=2', result.stdout)
        self.assertIn('flow_matching.rotation_noise_std=.4', result.stdout)
        self.assertNotIn('--predictor=', result.stdout)
        self.assertNotIn('DiffusionGaussianPredictor.', result.stdout)
        env['QUATERNION_REPRESENTATION'] = 'raw_standardized'
        self.assertNotEqual(subprocess.run(command, cwd=ROOT, env=env, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()

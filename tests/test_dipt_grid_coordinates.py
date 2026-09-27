"""CPU coverage of DiPT forward spatial inputs without CUDA backbone dependencies."""
import ast
from collections import OrderedDict
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import List
import unittest

import torch

from sr.interpolants import reference_means

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'models/diffusion_gaussian_predictor.py').read_text())
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef))
FORWARD = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
NAMESPACE = {'torch': torch, 'List': List, 'OrderedDict': OrderedDict, 'ALL_FEATURES': ['means']}
exec(compile(ast.Module(body=[FORWARD], type_ignores=[]), '<dipt-forward>', 'exec'), NAMESPACE)


class CapturePredictor(torch.nn.Module):
    forward = NAMESPACE['forward']

    def __init__(self, enabled):
        super().__init__()
        self.shift_negative_grid_coords = enabled
        self.grid_resolution = 1536
        self.input_features = ['means']
        self.output_features = []
        self.input_feat_to_mlp = False
        self.sh_degree = 0
        self.inputs = []

    def _input_feature_tensor(self, gs, key):
        return gs[key]

    def backbone(self, fields):
        self.inputs.append(fields)
        return SimpleNamespace(feat=fields['feat'])


class GridCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.references = [torch.tensor([[-5.96e-8, .2, -.1], [.3, .4, -.05], [.3, .4, -.05]]),
                           torch.tensor([[-2., .7, .1], [-1., .8, .2]])]
        self.states = [{'means': value + 10} for value in self.references]

    def test_forward_preserves_features_physical_positions_and_grid_differences(self):
        for enabled in (False, True):
            model = CapturePredictor(enabled)
            for training in (False, True):
                model.train(training)
                model(self.states, [0, 1], t=.2, batch_reference_means=self.references)
                fields = model.inputs[-1]
                torch.testing.assert_close(fields['coord'], torch.cat(self.references))
                torch.testing.assert_close(fields['feat'], torch.cat([gs['means'] for gs in self.states]))
                chunks = fields['grid_coord'].split([3, 2])
                for ref, actual in zip(self.references, chunks):
                    raw = torch.floor(ref * 1536).int()
                    expected = raw - raw.amin(0, keepdim=True).clamp(max=0) if enabled else raw
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual - actual[0], raw - raw[0])
                    torch.testing.assert_close(actual[:, 1], raw[:, 1])
                    if enabled:
                        self.assertTrue((actual >= 0).all())
                # The tiny negative X becomes zero after correcting its -1 grid index.
                self.assertEqual(chunks[0][0, 0].item(), 0 if enabled else -1)
                torch.testing.assert_close(chunks[0][1], chunks[0][2])
            model([self.states[0]], [0], t=.2, batch_reference_means=[self.references[0]])
            torch.testing.assert_close(model.inputs[-1]['grid_coord'], chunks[0])

    def test_modes_and_fixed_references_do_not_follow_flow_states(self):
        source, target = self.references[0], self.references[0] + torch.tensor([-.8, .1, .3])
        model = CapturePredictor(True)
        for mode in ('linear', 'latent', 'one_sided', 'encoding_decoding'):
            for time in (.1, .4, .5, .9):
                ref = reference_means(mode, time, source, target)
                expected_ref = target if mode == 'one_sided' or (mode == 'encoding_decoding' and time >= .5) else source
                self.assertIs(ref, expected_ref)
                captured = []
                for drift in (0., 100.):
                    model([{'means': source + drift}], [0], t=time, batch_reference_means=[ref])
                    captured.append(model.inputs[-1]['grid_coord'])
                torch.testing.assert_close(captured[0], captured[1])
                raw = torch.floor(expected_ref * 1536).int()
                torch.testing.assert_close(captured[0], raw - raw.amin(0, keepdim=True).clamp(max=0))

    def test_nonnegative_scenes_and_constructor_default(self):
        ref = torch.tensor([[.2, .3, .4], [.5, .6, .7]])
        model = CapturePredictor(True)
        model([{'means': ref}], [0], t=0.)
        torch.testing.assert_close(model.inputs[-1]['grid_coord'], torch.floor(ref * 1536).int())
        init = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        defaults = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults):], init.args.defaults))
        self.assertIs(ast.literal_eval(defaults['shift_negative_grid_coords']), False)

    def test_launcher_forwarding_validation_and_unchanged_names(self):
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        base = dict(os.environ, PREDICTOR='dipt', PYTHON='python')
        base.pop('GS_SHIFT_NEGATIVE_GRID_COORDS', None)
        outputs = []
        for value in (None, 'False', 'True'):
            env = dict(base)
            if value is not None:
                env['GS_SHIFT_NEGATIVE_GRID_COORDS'] = value
            result = subprocess.run(['bash', '-c', command, 'test', str(ROOT / 'scripts/overfit-sr-interpolants.sh')],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertIn('DiffusionGaussianPredictor.shift_negative_grid_coords=' + (value or 'False'), result.stdout)
            outputs.append(next(line for line in result.stdout.splitlines() if line.startswith('--output_dir=')))
        self.assertEqual(len(set(outputs)), 1)
        for predictor in ('dipt', 'ptv3'):
            result = subprocess.run(['bash', '-c', command, 'test', str(ROOT / 'scripts/overfit-sr-interpolants.sh')],
                                    env=dict(base, PREDICTOR=predictor, GS_SHIFT_NEGATIVE_GRID_COORDS='invalid'), capture_output=True, text=True)
            self.assertEqual(result.returncode, 2 if predictor == 'dipt' else 0)
            self.assertNotIn('--gin_param=GSFlowPredictor.shift_negative_grid_coords', result.stdout)
            if predictor == 'ptv3':
                self.assertNotIn('shift_negative_grid_coords', result.stdout)


if __name__ == '__main__':
    unittest.main()

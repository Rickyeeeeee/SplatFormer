"""EDiPT geometry, real Pointcept ordering, packing, and gradient checks."""
import copy
import unittest
from unittest import mock

import gin
import torch

from models.equivariant_gaussian_dipt import EquivariantGaussianDiPT, GeometricAttention
from models.equivariant_gaussian_dipt_predictor import EquivariantGaussianDiPTPredictor
from utils.data_augmentation import quaternion_multiply, quaternion_to_rotation_matrix


class EDiPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        gin.clear_config()
        gin.parse_config("""
EquivariantGaussianDiPT.depth = 2
EquivariantGaussianDiPT.channels = 12
EquivariantGaussianDiPT.num_head = 3
EquivariantGaussianDiPT.patch_size = (3, 2)
EquivariantGaussianDiPT.frequency_embedding_size = 8
""")

    def tearDown(self):
        gin.clear_config()

    def scenes(self, degree=1, counts=(5, 1)):
        states, geometry = [], []
        for n in counts:
            state = {key: torch.randn(n, width) for key, width in [('scales', 3), ('opacities', 1), ('features_dc', 3)]}
            if degree:
                state['features_rest'] = torch.randn(n, (degree + 1) ** 2 - 1, 3)
            states.append(state)
            geometry.append({'means': torch.randn(n, 3) * .05, 'quats': torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)})
        return states, geometry

    def packed(self, geometry):
        counts = [len(g['means']) for g in geometry]
        positions = torch.cat([g['means'] for g in geometry])
        return {'feat': torch.randn(sum(counts), 7), 'coord': positions,
                'rotations': quaternion_to_rotation_matrix(torch.cat([g['quats'] for g in geometry])),
                'grid_coord': torch.cat([(g['means'] * 100).floor().int() + 100 for g in geometry]),
                'offset': torch.tensor(counts).cumsum(0), 'timesteps': torch.arange(len(counts)).float() / 2}

    def test_shapes_zero_initialization_and_ordinary_sh(self):
        for degree in (0, 1, 3):
            states, geometry = self.scenes(degree)
            model = EquivariantGaussianDiPTPredictor(sh_degree=degree).eval()
            captured = []
            handle = model.backbone.register_forward_pre_hook(lambda module, args: captured.append(args[0]['feat'].detach()))
            outputs = model(states, t=.3, batch_geometry=geometry)
            handle.remove()
            expected = torch.cat([torch.cat([s[k].flatten(1) for k in model.input_features], dim=-1) for s in states])
            torch.testing.assert_close(captured[0], expected)
            for state, out in zip(states, outputs):
                n = len(state['scales'])
                self.assertEqual(out['means'].shape, (n, 3))
                self.assertEqual(out['quats'].shape, (n, 4))
                for key in state:
                    self.assertEqual(out[key].shape, state[key].shape)
                for value in out.values():
                    self.assertEqual(value.count_nonzero(), 0)
            if degree == 0:
                self.assertNotIn('features_rest', outputs[0])
            self.assertFalse(any('SubMConv' in type(m).__name__ for m in model.modules()))

    def test_excluded_sh_has_no_effect_or_gradients(self):
        states, geometry = self.scenes()
        model = EquivariantGaussianDiPTPredictor(use_features_rest=False, zeroinit=False).eval()
        self.assertEqual(model.gs_features_dim, 7)
        self.assertNotIn('features_rest', model.features_outputhead)
        for state in states:
            state['features_rest'].requires_grad_()
        baseline = model(states, t=.3, batch_geometry=geometry)
        sum(value.square().sum() for out in baseline for value in out.values()).backward()
        for state in states:
            self.assertIsNone(state['features_rest'].grad)
        changed = [{**state, 'features_rest': torch.full_like(state['features_rest'], float('nan'))} for state in states]
        result = model(changed, t=.3, batch_geometry=geometry)
        for first, second in zip(baseline, result):
            self.assertNotIn('features_rest', second)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

        flipped = [{**geom, 'quats': -geom['quats']} for geom in geometry]
        flipped_output = model(states, t=.3, batch_geometry=flipped)
        for geom, first, second in zip(geometry, baseline, flipped_output):
            torch.testing.assert_close((geom['quats'] * first['quats']).sum(-1), torch.zeros(len(geom['quats'])), atol=1e-6, rtol=0)
            for key in first:
                torch.testing.assert_close(second[key], -first[key] if key == 'quats' else first[key])

    def test_predictor_rigid_transform_and_quaternion_sign(self):
        states, geometry = self.scenes()
        model = EquivariantGaussianDiPTPredictor(zeroinit=False).eval()
        references = [g['means'] for g in geometry]
        baseline = model(states, t=[.2, .7], batch_geometry=geometry, batch_reference_means=references)
        global_q = torch.nn.functional.normalize(torch.tensor([.7, -.1, .4, .5]), dim=0)
        rotation = quaternion_to_rotation_matrix(global_q)
        transformed = [{'means': g['means'] @ rotation.T + torch.tensor([.1, -.2, .3]),
                        'quats': quaternion_multiply(global_q, g['quats'])} for g in geometry]
        result = model(states, t=[.2, .7], batch_geometry=transformed, batch_reference_means=references)
        for a, b in zip(baseline, result):
            torch.testing.assert_close(b['means'], a['means'] @ rotation.T, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(b['quats'], quaternion_multiply(global_q, a['quats']), atol=1e-5, rtol=1e-4)
            for key in states[0]:
                torch.testing.assert_close(a[key], b[key], atol=1e-5, rtol=1e-4)
        flipped = [{**g, 'quats': -g['quats']} for g in geometry]
        flipped_out = model(states, t=[.2, .7], batch_geometry=flipped, batch_reference_means=references)
        for g, a, b in zip(geometry, baseline, flipped_out):
            torch.testing.assert_close((g['quats'] * a['quats']).sum(-1), torch.zeros(len(g['quats'])), atol=1e-6, rtol=0)
            for key in a:
                torch.testing.assert_close(b[key], -a[key] if key == 'quats' else a[key])

    def test_ordering_fixed_permutations_and_scene_isolation(self):
        _, geometry = self.scenes()
        fields = self.packed(geometry)
        model = EquivariantGaussianDiPT(in_channels=7).eval()
        result = model(fields)
        for order, inverse in zip(result.serialized_order, result.serialized_inverse):
            torch.testing.assert_close(order[inverse], torch.arange(6))
        fixed = {**fields, 'serialized_order': result.serialized_order}
        torch.testing.assert_close(model(fixed).feat, result.feat)
        rotation = quaternion_to_rotation_matrix(torch.tensor([.5, .5, .5, .5]))
        transformed = {**fixed, 'coord': fields['coord'] @ rotation.T + 3,
                       'rotations': rotation @ fields['rotations']}
        torch.testing.assert_close(model(transformed).feat, result.feat, atol=1e-5, rtol=1e-4)
        perturbed = {**fields, 'feat': fields['feat'].clone()}
        perturbed['feat'][5] += 20
        torch.testing.assert_close(model(perturbed).feat[:5], result.feat[:5])
        # A singleton scene must agree alone and when packed after a larger scene.
        single = {key: fields[key][5:] for key in ('feat', 'coord', 'rotations', 'grid_coord')}
        single.update(offset=torch.tensor([1]), timesteps=fields['timesteps'][1:])
        torch.testing.assert_close(model(single).feat, result.feat[5:], atol=1e-6, rtol=1e-5)
        self.assertNotIn('sparse_conv_feat', result)

    def test_padding_and_known_geometry(self):
        attention = GeometricAttention(12, 3).eval()
        feat = torch.randn(1, 2, 12)
        pos = torch.tensor([[[0., 0., 0.], [2., 3., 4.]]])
        rotations = torch.eye(3).expand(1, 2, 3, 3)
        captured = []
        handle = attention.geometry_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        result = attention.attend(feat, pos, rotations, torch.ones(1, 2, dtype=torch.bool))
        handle.remove()
        torch.testing.assert_close(captured[0][0, 0, 1, :3], torch.tensor([2., 3., 4.]))
        torch.testing.assert_close(captured[0][0, 1, 0, :3], torch.tensor([-2., -3., -4.]))
        torch.testing.assert_close(captured[0][0, 0, 1, 3:], torch.eye(3).flatten())
        padded_feat = torch.cat([feat, torch.randn(1, 2, 12) * 100], dim=1)
        padded_pos = torch.cat([pos, torch.randn(1, 2, 3)], dim=1)
        padded_rot = torch.eye(3).expand(1, 4, 3, 3)
        padded = attention.attend(padded_feat, padded_pos, padded_rot, torch.tensor([[True, True, False, False]]))
        torch.testing.assert_close(padded[:, :2], result)

    def test_all_patches_match_independent_attention_and_gradients(self):
        attention = GeometricAttention(12, 3).train()
        reference = copy.deepcopy(attention)
        features = torch.randn(6, 12, requires_grad=True)
        positions = torch.randn(6, 3, requires_grad=True)
        rotations = quaternion_to_rotation_matrix(torch.randn(6, 4)).requires_grad_()
        inputs = (features, positions, rotations)
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        order = torch.tensor([4, 2, 0, 1, 3, 5])
        inverse = order.argsort()
        with mock.patch.object(attention, 'attend', wraps=attention.attend) as attend:
            actual = attention(*inputs, order, inverse, [5, 1], 3)
            actual.square().sum().backward()
            self.assertEqual(attend.call_count, 1)
            self.assertEqual(attend.call_args.args[0].shape, (3, 3, 12))
        # Independent unpadded patches are the mathematical reference.
        pieces = []
        for indices in (order[:3], order[3:5], order[5:]):
            patch_inputs = tuple(value[indices][None] for value in reference_inputs)
            pieces.append(reference.attend(*patch_inputs, torch.ones(1, len(indices), dtype=torch.bool))[0])
        expected = torch.cat(pieces)[inverse]
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        for value, reference_value in zip(inputs, reference_inputs):
            self.assertTrue(torch.isfinite(value.grad).all())
            torch.testing.assert_close(value.grad, reference_value.grad, atol=1e-5, rtol=1e-4)
        for (name, value), (_, reference_value) in zip(attention.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(value.grad, reference_value.grad, atol=1e-5, rtol=1e-4, msg=name)

    def test_validation(self):
        states, geometry = self.scenes()
        model = EquivariantGaussianDiPTPredictor()
        for kwargs in ({'t': None, 'batch_geometry': geometry}, {'t': .2, 'batch_geometry': geometry[:1]},
                       {'t': [.1, .2, .3], 'batch_geometry': geometry},
                       {'t': .2, 'batch_geometry': geometry, 'batch_reference_means': [torch.zeros(4, 3), geometry[1]['means']]}):
            with self.assertRaises(ValueError):
                model(states, **kwargs)
        bad = copy.deepcopy(geometry)
        bad[0]['quats'][0] = 0
        with self.assertRaises(ValueError):
            model(states, t=.2, batch_geometry=bad)
        with self.assertRaises(ValueError):
            model(states, t=.2, batch_geometry=geometry, batch_reference_means=[g['means'] + 100 for g in geometry])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_backward(self):
        states, geometry = self.scenes()
        states = [{k: v.cuda() for k, v in s.items()} for s in states]
        geometry = [{k: v.cuda().requires_grad_() for k, v in g.items()} for g in geometry]
        for use_rest in (True, False):
            model = EquivariantGaussianDiPTPredictor(zeroinit=False, use_features_rest=use_rest).cuda().train()
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(states, t=.5, batch_geometry=geometry)
                loss = sum(v.float().square().mean() for out in outputs for v in out.values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())



if __name__ == '__main__':
    unittest.main()

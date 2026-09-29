"""Measured fixed or augmented reference SE(3) checks. Run from the repository with PYTHONPATH=. ."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import gin
import torch

from models.equivariant_gaussian_dipt_predictor import EquivariantGaussianDiPTPredictor
from sr.edipt_interpolants import predict_velocity
from utils.data_augmentation import quaternion_multiply, quaternion_to_rotation_matrix
from utils.rotation_flow import quaternion_derivative_to_body, rotation_exp


def rotations():
    result = [torch.tensor([1., 0., 0., 0.])]
    for axis in torch.eye(3):
        result.extend(rotation_exp(axis * math.radians(angle)) for angle in (1, 45, 90, 180))
    axis = torch.tensor([1., 2., 3.]); axis = axis / axis.norm()
    result.extend(rotation_exp(axis * math.radians(angle)) for angle in (.01, 179.99, 180))
    generator = torch.Generator().manual_seed(2026)
    result.extend(torch.nn.functional.normalize(torch.randn(16, 4, generator=generator), dim=-1))
    return result


def evaluate(model, ordinary, positions, quats, references, adapter, std):
    model.zero_grad(set_to_none=True)
    p = positions.detach().clone().requires_grad_()
    q = quats.detach().clone().requires_grad_()
    attrs = {k: v.detach().clone().requires_grad_() for k, v in ordinary.items()}
    counts = (1, 7, 19)
    states = [{k: v.split(counts)[i] for k, v in attrs.items()} for i in range(3)]
    geometry = [{'means': a, 'quats': b} for a, b in zip(p.split(counts), q.split(counts))]
    hidden = []
    hook = model.backbone.register_forward_hook(lambda module, args, out: hidden.append(out.feat))
    if adapter:
        for state, geom in zip(states, geometry):
            state.update(means=(geom['means'] - std.means['means']) / std.stds['means'], quats=geom['quats'])
        outputs = predict_velocity(model, states, [0, 1, 2], references, [.2, .5, .8], std)
        physical = torch.cat([o['means'] for o in outputs]) * std.stds['means']
        angular = torch.cat([o['quats'] for o in outputs])
        # Adapter returns angular velocity; reconstruct its physical tangent derivative.
        normalized = torch.nn.functional.normalize(q, dim=-1)
        derivative = .5 * quaternion_multiply(normalized, torch.cat([torch.zeros_like(angular[:, :1]), angular], -1))
    else:
        outputs = model(states, t=[.2, .5, .8], batch_geometry=geometry, batch_reference_means=references)
        physical = torch.cat([o['means'] for o in outputs])
        derivative = torch.cat([o['quats'] for o in outputs])
        angular = quaternion_derivative_to_body(q, derivative)
    hook.remove()
    values = {'hidden': hidden[0], 'physical_velocity': physical, 'quaternion_derivative': derivative,
              'body_angular': angular, 'ordinary_outputs': torch.cat([torch.cat([o[k] for k in attrs], -1) for o in outputs])}
    # Invariant scalar objective exercises all output heads and geometry derivatives.
    loss = physical.square().sum(-1).mean() + angular.square().sum(-1).mean() + values['ordinary_outputs'].square().mean()
    loss.backward()
    values.update(loss=loss.reshape(1), parameter_gradients=torch.cat([v.grad.flatten() for v in model.parameters()]),
                  position_gradients=p.grad, quaternion_gradients=q.grad,
                  ordinary_gradients=torch.cat([v.grad.flatten() for v in attrs.values()]))
    return {k: v.detach() for k, v in values.items()}


def write_report(path):
    summary = json.loads((path / 'summary.json').read_text())
    records = [json.loads(line) for line in (path / 'cases.jsonl').read_text().splitlines()]
    reference_mode = summary.get('serialization_reference', 'unaugmented')
    lines = ['# EDiPT single-forward SE(3) validation', '',
             f'Serialization references: **{reference_mode}**.', '',
             '32 distinct rotations × 4 translations × 3 seeds = 384 cases per suite/device.',
             f'Two suites (predictor and standardized adapter), devices {summary["devices"]}, FP32: **{summary["total_transform_cases"]} forward/backward cases**.',
             f'Failed tensor comparisons: **{summary["failed_records"]} / {len(records)}**. Skipped checks: {summary["skipped"]}.',
             '', 'Tolerance: `atol=1e-5`, `rtol=1e-4`. Normalized error is `abs(error)/(atol+rtol*abs(expected))`; acceptance ≤ 1.',
             '', '| Tensor | Largest absolute error | Largest normalized error | Absolute worst case (device/suite/seed/rotation/translation) |',
             '|---|---:|---:|---|']
    for key in sorted({r['tensor'] for r in records}):
        rows = [r for r in records if r['tensor'] == key]
        worst = max(rows, key=lambda r: r['max_absolute'])
        normalized = max(r['max_tolerance_normalized'] for r in rows)
        case = f'{worst["device"]}/{worst["suite"]}/{worst["seed"]}/R{worst["rotation_index"]}/{worst["translation"]}'
        lines.append(f'| {key} | {worst["max_absolute"]:.9g} | {normalized:.9g} | {case} |')
    lines += ['', 'Absolute and normalized maxima can come from different cases. `cases.jsonl` contains every metric, including RMS errors and exact quaternions; `summary.json` identifies normalized worst cases per suite/device.',
              '', '## Rotation definitions', '', 'R0 identity; R1–R4 X axis at 1°, 45°, 90°, 180°; R5–R8 Y axis; R9–R12 Z axis;',
              'R13–R15 normalized (1,2,3) axis at 0.01°, 179.99°, 180°; R16–R31 random quaternions from generator seed 2026.',
              'Input/model seeds: 11, 23, 47. Scenes: 1, 7, 19 points. Patch sizes: 4, 8. Nonzero heads, evaluation mode.',
              '', '## Regression logs', '']
    for name in ('model_regression.log', 'regression.log', 'interpolant_regression.log', 'augmentation_regression.log'):
        log = path / name
        if log.exists():
            text = log.read_text()
            counts = [line for line in text.splitlines() if line.startswith('Ran ')]
            lines.append(f'- [{name}]({name}): {counts[-1] if counts else "running"}; {"OK" if text.rstrip().endswith("OK") else "see log"}.')
    lines += ['', 'Regression logs, when present, describe separate regression runs; this matrix measures FP32 transformation consistency.',
              '', '## Scope and use', '', f'The matrix uses {reference_mode} serialization references and excludes higher-order SH. Geometry is physically transformed, and the adapter test uses coordinate mean (0.2,-0.3,0.7) and std (0.3,1.7,2.4).',
              'Backward checks use an invariant physical scalar loss, not the production standardized loss. Parameter gradients cover every trainable parameter; geometry gradients use physical leaf positions and scalar-first quaternions.',
              'Ground-truth SH restoration happens after sampling; evaluation rendering metrics therefore use oracle higher-order SH. No full dataset training/rendering run was performed for this change.',
              '', '```bash', 'SERIALIZATION_REFERENCE=unaugmented GS_USE_FEATURES_REST=False MIX_SCHEDULE=fm-only \\', '  bash experiments/edipt-rotation-range.sh', '```', '',
              f'Reproduce the matrix with `PYTHONPATH=. python tests/validate_edipt_se3.py --serialization-reference {reference_mode}`. The guarantee does not cover recomputing world-axis neighborhoods after transformation or stochastic training trajectories.', '']
    (path / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--serialization-reference', choices=('unaugmented', 'augmented'), default='unaugmented')
    args = parser.parse_args()
    path = Path(args.output or ('validation/edipt_se3_augmented' if args.serialization_reference == 'augmented' else 'validation/edipt_se3')); path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    gin.parse_config('''
EquivariantGaussianDiPT.depth = 2
EquivariantGaussianDiPT.channels = 12
EquivariantGaussianDiPT.num_head = 3
EquivariantGaussianDiPT.patch_size = (4, 8)
EquivariantGaussianDiPT.frequency_embedding_size = 8
EquivariantGaussianDiPT.shuffle_orders = False
''')
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    translations = [(0, 0, 0), (.1, -.2, .3), (1, -2, 3), (-3, 2, -1)]
    summary = {'serialization_reference': args.serialization_reference, 'devices': devices, 'dtype': 'float32', 'seeds': [11, 23, 47], 'rotations': 32,
               'translations': translations, 'cases_per_suite_device': 384, 'atol': 1e-5, 'rtol': 1e-4,
               'skipped': [] if 'cuda' in devices else ['CUDA unavailable'], 'worst': {}, 'failed_records': 0}
    with (path / 'cases.jsonl').open('w') as records, (path / 'results.log').open('w') as log:
        for device in devices:
            for adapter in (False, True):
                suite = 'adapter' if adapter else 'predictor'
                for seed in summary['seeds']:
                    torch.manual_seed(seed)
                    model = EquivariantGaussianDiPTPredictor(zeroinit=False, use_features_rest=False).to(device).eval()
                    positions = torch.randn(27, 3, device=device) * .3
                    quats = torch.nn.functional.normalize(torch.randn(27, 4, device=device), dim=-1)
                    ordinary = {k: torch.randn(27, n, device=device) for k, n in [('scales', 3), ('opacities', 1), ('features_dc', 3)]}
                    references = [p.detach().clone() for p in positions.split((1, 7, 19))]
                    std = SimpleNamespace(quaternion_representation='unit_unstandardized',
                                          means={'means': torch.tensor([.2, -.3, .7], device=device)},
                                          stds={'means': torch.tensor([.3, 1.7, 2.4], device=device)})
                    baseline = evaluate(model, ordinary, positions, quats, references, adapter, std)
                    failures_before = summary['failed_records']
                    for rotation_index, quaternion in enumerate(rotations()):
                        quaternion = quaternion.to(device)
                        rotation = quaternion_to_rotation_matrix(quaternion)
                        expected = dict(baseline)
                        for key in ('physical_velocity', 'position_gradients'):
                            expected[key] = baseline[key] @ rotation.T
                        for key in ('quaternion_derivative', 'quaternion_gradients'):
                            expected[key] = quaternion_multiply(quaternion, baseline[key])
                        for translation in translations:
                            transformed_positions = positions @ rotation.T + positions.new_tensor(translation)
                            transformed_references = references
                            if args.serialization_reference == 'augmented':
                                transformed_references = [p.detach().clone() for p in transformed_positions.split((1, 7, 19))]
                            actual = evaluate(model, ordinary, transformed_positions,
                                              quaternion_multiply(quaternion, quats), transformed_references, adapter, std)
                            for key, value in actual.items():
                                error = (value - expected[key]).abs()
                                normalized = error / (1e-5 + 1e-4 * expected[key].abs())
                                record = dict(device=device, dtype='float32', suite=suite, seed=seed,
                                              rotation_index=rotation_index, quaternion=quaternion.tolist(), translation=translation,
                                              tensor=key, max_absolute=error.max().item(), rms_absolute=error.square().mean().sqrt().item(),
                                              max_tolerance_normalized=normalized.max().item(), passed=bool((normalized <= 1).all()))
                                records.write(json.dumps(record) + '\n')
                                summary['failed_records'] += not record['passed']
                                category = f'{device}/{suite}/{key}'
                                previous = summary['worst'].get(category)
                                if previous is None or record['max_tolerance_normalized'] > previous['max_tolerance_normalized']:
                                    summary['worst'][category] = record
                    line = f'{device} {suite} seed={seed}: 32 rotations x 4 translations = 128 forward/backward cases; failed records={summary["failed_records"]-failures_before}'
                    print(line, flush=True); log.write(line + '\n'); log.flush()
        summary['total_transform_cases'] = 384 * 2 * len(devices)
        log.write(json.dumps(summary, indent=2) + '\n')
    (path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(path)
    print(f'Total cases={summary["total_transform_cases"]}, failed records={summary["failed_records"]}', flush=True)
    return bool(summary['failed_records'])


if __name__ == '__main__':
    raise SystemExit(main())

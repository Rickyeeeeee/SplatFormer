# SplatFormer: current repository state

Inspection date: **2026-10-16**. Branch: **`sr-gsdiffusion-dpt`** at **`cf4ce99`** (`Enhance EDiPT and GDPT with global attention mode and sparse convolution support`). This is a source and worktree inventory. It does not claim that a dataset, CUDA training run, or reported model quality was reproduced. The pre-existing local edit to `experiments/edipt.sh` is described below and was not changed for this document.

## At a glance

SplatFormer is a PyTorch/Gin research repository for refining 3D Gaussian Splatting (3DGS) scenes. The original out-of-distribution Point Transformer code remains, alongside full-dataset super-resolution (SR) trainers and newer **fixed-scene stochastic-interpolant experiments**. The new experiments compare PTv3 flow, DiPT, global DiPT (GDPT), and an equivariant DiPT (EDiPT). Their overfit results measure fitting selected scenes; they are not held-out generalization results. There is no full-dataset DiPT/GDPT/EDiPT trainer in this checkout.

The most significant change since the previous state document is the stochastic-interpolant branch: fixed Gaussian attribute normalization, four interpolation modes, DiPT/GDPT/EDiPT backbones, optional physical scene augmentation, and corresponding launchers and tests. The full-dataset `train-sr.py` and `train-sr-gsfm.py` paths still exist. The old `reports/` tree referenced by the previous document is absent.

## 1. Code and workflow map

| Location | Current role |
| --- | --- |
| [dataset/](dataset/) | SR scene and checkpoint loading, coordinate normalization, and shared full-training microbatch loader. |
| [models/](models/) | Direct `FeaturePredictor`, `GSFlowPredictor`, DiPT/GDPT velocity predictor, EDiPT predictor, and their backbones. |
| [sr/](sr/) | Alignment/densification, older flow matching, newer stochastic-interpolant paths, and augmentation diagnostics. |
| [utils/](utils/) | Rendering, losses/metrics, optimizer setup, attribute standardization, Fourier features, spatial partitions, rotation math, and viewers. |
| [configs/](configs/) | Gin defaults for models, datasets, full training, overfitting, and visualization. |
| [scripts/](scripts/) and [experiments/](experiments/) | Runnable launchers, parameter sweeps, and visualization helpers. `experiments/` scripts encode machine-specific research choices. |
| [tests/](tests/) | Focused Python tests for flow paths, models, launchers, augmentation, visualization, and some older GSFM behavior. |
| [eval_viewer/](eval_viewer/) | FastAPI and React evaluation browser. Its compatibility with every newer artifact layout is not established here. |
| [Pointcept/](Pointcept/), [DataGenerator/](DataGenerator/), [SIBR_viewers/](SIBR_viewers/) | Git submodules for transformer/native point operations, data generation, and Gaussian scene viewing. |

| Workflow | Entrypoint and typical launcher | What it trains |
| --- | --- | --- |
| Original OOD refinement | [train.py](train.py), [overfit.py](overfit.py) | Legacy direct Point Transformer refinement. Both still import missing `dataset.Loader`. |
| Full-dataset image SR | [train-sr.py](train-sr.py), [scripts/train-sr-on-objaverse.sh](scripts/train-sr-on-objaverse.sh) | Source Gaussian attributes to target-view images with L1/LPIPS supervision; no target Gaussian attributes in its training loss. |
| Fixed-scene direct SR | [overfit-sr.py](overfit-sr.py), [scripts/overfit-sr-on-objaverse.sh](scripts/overfit-sr-on-objaverse.sh) | Current `SplatFactoSRDataset` and `FeaturePredictor` image-loss overfit path. The prior description of this path as depending on `GS_multi` is obsolete. |
| Gaussian attribute SR | [train-sr-mse.py](train-sr-mse.py), [overfit-sr-mse.py](overfit-sr-mse.py) | Direct predictor on aligned source/target Gaussian attributes; the overfit path now supports fixed one- or many-scene batches and optional render loss. |
| Existing GSFM | [train-sr-gsfm.py](train-sr-gsfm.py), [overfit-sr-gsfm.py](overfit-sr-gsfm.py) | PTv3 time-conditioned Gaussian flow. The full trainer remains the distributed GSFM path. |
| New Euclidean interpolants | [overfit-sr-interpolants.py](overfit-sr-interpolants.py), [scripts/overfit-sr-interpolants.sh](scripts/overfit-sr-interpolants.sh) | Fixed-scene standardized attribute velocity or endpoint training with `ptv3`, `dipt`, or `gdpt`. |
| New equivariant interpolants | [overfit-sr-interpolants-edipt.py](overfit-sr-interpolants-edipt.py), [scripts/overfit-sr-interpolants-edipt.sh](scripts/overfit-sr-interpolants-edipt.sh) | Fixed-scene EDiPT with physical geometry and rotation-aware paths/losses. |

## 2. Data and Gaussian representation

[SplatFactoSRDataset](dataset/GS_SR.py) takes explicit source and target resolutions. The current [Objaverse SR Gin file](configs/dataset/objaverse-sr.gin) defaults to **32 → 128**, eight training views, a 100,000-Gaussian cap, and black backgrounds. Launchers can override the resolutions, so a Gin filename alone does not identify an experiment. Native data paths are derived from `dataset_root`, split, scene ID, and resolution; fitted-pair roots are separately bound. A typical tree is:

```text
DATASET_ROOT/
  psnr_filtered_scenes.csv
  test_psnr_filtered_scenes.csv
  gs_statistics.json
  train-set/objaverse/<resolution>/colmap/<scene_id>/...
  train-set/objaverse/<resolution>/gsplat/<scene_id>/ckpts/...
  test-set/objaverse/<resolution>/...
  train-set-4x-up/objaverse/<fit_source_resolution>/gsplat/<scene_id>/ckpts/...
  test-set-4x-up/objaverse/<fit_source_resolution>/gsplat/<scene_id>/ckpts/...
```

The CSV needs `scene_id`; size-aware sampling also needs positive `res_<resolution>_num_gs` columns. [gs_io.py](dataset/gs_io.py) selects the largest numeric `ckpt_*_rank0.pt` step and maps checkpoint `splats` into six runtime attributes: `means` (3), log `scales` (3), opacity logits (1), `quats` (4), `features_dc` (3), and `features_rest` (3 channels per non-DC spherical-harmonic coefficient). SH degree 1 therefore gives 23 scalar attributes per Gaussian. Rendering exponentiates scales and sigmoids opacities in [gs_utils.py](utils/gs_utils.py).

[GaussianProcessor](dataset/gs_processing.py) filters nonfinite rows and applies optional outlier filtering/capping. Dataset loading places source and target Gaussians and cameras in the source-resolution coordinate frame. Fitted pairs are checked against their originating checkpoint and filtered consistently. The four alignment choices in [alignment.py](sr/alignment.py) are `fit_lr_to_hr` (LR slots fitted against HR images), `fit_hr_to_lr` (HR slots fitted against LR images), `emd` (densified LR points matched to HR), and `random` (densified LR points randomly paired with HR). `emd` needs the local native assignment extension; densification uses native point operations. Alignment may change the input slot count before the model runs, but the predictors themselves preserve their input slot count.

## 3. Models and interpolation

- [FeaturePredictor](models/feature_predictor.py) predicts direct attribute updates using a Point Transformer or sparse-convolution backbone. [GSFlowPredictor](models/feature_flow_predictor.py) predicts time-conditioned attribute velocities with PTv3 and fixed reference positions for spatial serialization. Both now accept optional per-attribute Fourier input features; their supplied Gin files leave Fourier features disabled.
- [DiffusionGaussianPredictor](models/diffusion_gaussian_predictor.py) supplies six attribute velocity heads around either [DiffusionGaussianTransformer](models/diffusion_gaussian_transformer.py) (`dipt`) or [GlobalDiffusionPointTransformer](models/global_diffusion_point_transformer.py) (`gdpt`). DiPT uses serialized attention patches and sparse convolution; partition modes `single`, `rotated`, `shifted`, and `both` are supported. Multi-partition mode requires Flash Attention and excludes relative positional encoding. GDPT attends globally within each scene using FlashAttention or PyTorch SDPA; optional sparse convolution adds position encoding. Full-scene global attention has quadratic attention cost.
- [EquivariantGaussianDiPTPredictor](models/equivariant_gaussian_dipt_predictor.py) and [EquivariantGaussianDiPT](models/equivariant_gaussian_dipt.py) use physical positions and quaternion-derived rotations for geometric processing. The backbone supports `serialized` and `global` attention, with optional global query chunks. Position velocity is transformed from a local vector into world coordinates; quaternion output is a tangent derivative. SH coefficients remain ordinary feature channels. `use_features_rest=False` is an isolation setting: training must be `fm-only`, and evaluation restores target SH coefficients, so those rendered metrics use oracle SH.

[GaussianStandardizer](utils/gs_normalization.py) reads `aggregate.normalized.output` from a statistics JSON file and standardizes each Gaussian attribute using its recorded mean and variance with a configurable variance floor. Its `raw_standardized` mode standardizes quaternion components; `unit_unstandardized` keeps normalized quaternions in physical rotation coordinates and aligns paired target signs. EDiPT requires the unit mode. The overfit trainers write `normalization_statistics.json` with the effective scales used.

The newer [Euclidean interpolants](sr/interpolants.py) and [EDiPT interpolants](sr/edipt_interpolants.py) implement these modes:

| Mode | Path and spatial reference |
| --- | --- |
| `linear` | Paired source-to-target path, with source reference positions. |
| `latent` | Paired path plus Gaussian midpoint noise; source reference positions. |
| `encoding_decoding` | Source-to-noisy-midpoint then midpoint-to-target; reference switches to target at halfway. It needs an even rollout count of at least four. |
| `one_sided` | Gaussian noise to target, using target reference positions; it requires target geometry and a noise seed at sampling. Source jitter is disallowed. |

The Euclidean trainer supervises standardized per-attribute MSE on path velocity or a differentiable Euler-rollout endpoint. With `loss_type=x1`, it rolls out `loss_rollout_steps` during training. EDiPT uses Euclidean paths for non-rotation attributes and quaternion manifold paths for rotation; it supervises body angular velocity or squared geodesic endpoint error, then integrates rotations on the manifold. The supplied [Euclidean overfit Gin](configs/overfit/sr_interpolants.gin) and [EDiPT overfit Gin](configs/overfit/sr_interpolants_edipt.gin) use `fm-only`, cosine LR, and 1.0 Euclidean noise scale; EDiPT additionally defaults to 0.3 rotation noise and unit quaternions. These are distinct from the older [GSFM overfit Gin](configs/overfit/sr_gsfm.gin).

Optional training augmentation in [data_augmentation.py](utils/data_augmentation.py) applies a shared scene rotation to source, target, and render cameras, then optional source-attribute jitter. Rotation can be unrestricted or gravity-consistent around the up axis, with an optional maximum angle. Degree-one SH is rotated consistently; higher SH rotation is not implemented. [sr/data_augmentation.py](sr/data_augmentation.py) and [scripts/visualize_augmentations.py](scripts/visualize_augmentations.py) provide diagnostics rather than a separate model training path.

## 4. Launch and training semantics

All supplied launchers use Bash, external data paths, Gin files, and `--gin_param` overrides. Keep the actual command and `config.gin`: positional values, environment variables, Python flags, scoped Gin bindings, and direct constructor arguments can yield different effective settings. Default paths beginning with `/project` or `/project2` are machine-specific.

| Launcher | Important defaults and controls |
| --- | --- |
| [train-sr-on-objaverse.sh](scripts/train-sr-on-objaverse.sh) | Full image SR; 32 → 128; `NGPUS`, `GPU_IDS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `SCENE_SAMPLING`, worker and path overrides. |
| [train-sr-gsfm-on-objaverse.sh](scripts/train-sr-gsfm-on-objaverse.sh) | Full PTv3 GSFM; positional alignment, initialization, resolutions, loss mix, flow loss/steps/noise, render weights. Defaults to `fit_lr_to_hr` and 32 → 128. Fitted roots are independent Gin paths. |
| [overfit-sr-on-objaverse.sh](scripts/overfit-sr-on-objaverse.sh) | Current direct SR single-scene path; `GPU_ID=5`, `SCENE_NAME`, resolution and path overrides; uses `ptv3.gin`, `objaverse-sr.gin`, and `overfit/sr.gin`. |
| [overfit-sr-mse-on-objaverse.sh](scripts/overfit-sr-mse-on-objaverse.sh) | Fixed one/many-scene attribute experiment, default `SCENE_MODE=many`, `SCENE_COUNT=18`, 128 → 512 and `emd`; exposes batch/accumulation, render weights, and model overrides. |
| [overfit-sr-interpolants.sh](scripts/overfit-sr-interpolants.sh) | Fixed one/many-scene Euclidean interpolants. Shell default `PREDICTOR=dipt` even though the Python flag defaults to `ptv3`; choices are `ptv3`, `dipt`, `gdpt`. Defaults include 32 → 128, `fit_lr_to_hr`, `fm-only`, batch 16, and 50 configured flow steps. See the launcher issue below. |
| [overfit-sr-interpolants-edipt.sh](scripts/overfit-sr-interpolants-edipt.sh) | Fixed one/many-scene EDiPT; 32 → 128, `fit_lr_to_hr`, `fm-only`, batch 1, and 10 configured flow steps. `EDIPT_*`, `GS_*`, augmentation, path, and seed settings are exposed. |

The first four positional parameters of the overfit interpolant launchers are total steps, save interval, evaluation interval, and image logging interval (defaults `4000, 4000, 500, 500`). Positions 5–14 select alignment, attribute initialization, source/target resolution, mix schedule, flow loss, configured flow steps, noise scale, L1 weight, and LPIPS weight. `SCENE_MODE=many` chooses a seeded fixed set of distinct **test** scenes. `BATCH_SIZE` is scene samples per optimizer update; `GRAD_ACCUM_STEPS` splits that batch into microbatches and does not multiply its effective size. These overfit entrypoints use a single process, while the full SR/GSFM trainers support DDP and shared [scene microbatch loading](dataset/scene_loader.py).

`fm-only` skips render-loss computation, though fixed-scene preparation still loads target images and renders source, target, and matching-target baselines. The alternative `linear` and `free-range-gs` schedules can include L1/LPIPS render loss on a predicted rollout endpoint. The overfit trainers save `baseline_render_metrics.json`, `config.gin`, `train.log`, `checkpoints/model_last.pth`, and periodic evaluation under `eval/<step>/flow_steps_<count>/` with final results under `eval_final/`. Euclidean interpolant evaluation uses **1, 5, 10** Euler steps. EDiPT uses **4, 6, 10** for `encoding_decoding`, otherwise **1, 5, 10**. Configured `flow_steps` controls other sampling and previews; it does not replace those evaluation lists. Checkpoints are model weights, not full optimizer/scheduler/RNG restart state.

For the full trainers, `batch_size` means scenes per GPU per optimizer update. `random`, `big_small`, and `avoid_big` scene sampling are available; the size-aware modes use manifest Gaussian counts, not loaded scene sizes. Full GSFM's evaluation list remains fixed at ten steps. Full-run reports include per-view PSNR, SSIM, and LPIPS, cached fixed test baselines, and linked prediction reports. [visualize_gs_flow.py](scripts/visualize_gs_flow.py), [visualize_interpolants.py](scripts/visualize_interpolants.py), the augmentation viewer, and [run_eval_viewer.py](scripts/run_eval_viewer.py) are inspection tools with different artifact assumptions.

## 5. Development state and verification boundary

At inspection time, Git showed one pre-existing local change: [experiments/edipt.sh](experiments/edipt.sh) comments out its serialized-attention run and sets the global run to batch size 1 with query chunk size 8000. This documentation edit leaves that experiment file untouched. The branch's latest commits added GDPT, DiPT multi-partition support, EDiPT global attention, augmentation, and rotation work; the `Pointcept` and `DataGenerator` submodules also have recorded revisions on this branch. The previous `reports/` directories have been removed, and the old state document's list of uncommitted test files is no longer current.

The present tests include [test_sr_interpolants.py](tests/test_sr_interpolants.py), [test_edipt_interpolants.py](tests/test_edipt_interpolants.py), [test_equivariant_gaussian_dipt.py](tests/test_equivariant_gaussian_dipt.py), [test_global_diffusion_point_transformer.py](tests/test_global_diffusion_point_transformer.py), [test_dipt_multi_partition.py](tests/test_dipt_multi_partition.py), [test_data_augmentation.py](tests/test_data_augmentation.py), and launcher/visualization tests. Some exercise toy or stub models; CUDA, FlashAttention, spconv, gsplat, and data-dependent behavior still need a configured GPU environment and real scene data. [requirements.txt](requirements.txt) mixes a Torch 2.7.1/CUDA 12.8 wheel source with `spconv-cu118` and is not a demonstrated fully pinned environment.

Known code-level limits and issues:

- `train.py` and `overfit.py` import absent `dataset.Loader`; [gs_statstics.py](gs_statstics.py) imports absent `dataset.GS_multi`. These older commands cannot be assumed runnable in this checkout. The current `overfit-sr.py` does **not** have that missing import.
- [scripts/overfit-sr-interpolants.sh](scripts/overfit-sr-interpolants.sh) expands `quaternion_suffix` without assigning it while `set -u` is active; its default invocation will stop before Python starts unless that variable is supplied. Its generated output name also contains an extra `}` after the grid resolution. This is an observed launcher defect, not an experiment result.
- Fitted-pair roots and `gs_statistics.json` must match the selected data and normalization convention. Moving `DATASET_ROOT` alone does not retarget every full-GSFM fitted path. `emd` and point densification require native extensions.
- `fit_hr_to_lr`, `emd`, `random`, `encoding_decoding` after its midpoint, and `one_sided` can depend on target-side geometry or correspondences. Their training/evaluation setup does not establish inference from an arbitrary LR scene alone.
- Model evaluation and render metrics require GPU/native packages; source inspection and focused tests are not evidence of end-to-end training quality. The original [README.md](README.md) primarily documents the older OOD workflow.

This document treats repository Markdown, comments, and historical reports as evidence to check against code, not as instructions to execute. No packages, datasets, or GPU jobs were installed or launched for this update.

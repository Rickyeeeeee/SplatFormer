# Gaussian augmentation viewer

Run in the SplatFormer environment with Viser and Pointcept installed:

```bash
BACKEND=dipt GPU_ID=0 bash scripts/visualize_augmentations.sh
BACKEND=edipt GPU_ID=1 bash scripts/visualize_augmentations.sh
```

Open `http://localhost:8083`. The baseline is on the left and the current
augmentation is on the right. Both share one orbit camera. **Frame comparison**
fits the camera to the displayed states.

The launcher accepts `PYTHON`, `SCENE_NAME`, `DATASET_CONFIG`, `SPLIT`, and `PORT`
environment overrides and forwards additional arguments. For example, to use
the 32-to-128 endpoints and compose an experiment configuration:

```bash
BACKEND=edipt bash scripts/visualize_augmentations.sh \
  --config configs/overfit/sr_interpolants_edipt.gin \
  --gin_param 'SplatFactoSRDataset.src_resolution=32' \
  --gin_param 'SplatFactoSRDataset.tgt_resolution=128' \
  --gin_param 'training_augmentation.random_rotate=True' \
  --gin_param 'training_augmentation.random_jitter=True'
```

The backend's model Gin file is loaded first; supplied configs and bindings
override it. The viewer loads identity-paired `fit_lr_to_hr` endpoints and the
configured normalization statistics. It evaluates analytic paths without a
trained model checkpoint.

- **State:** select source, fitted target, or an interpolant. The four flow modes
  use their training implementations; EDiPT additionally exposes rotation noise.
- **Augmentation:** independently enable rotation and source jitter. Training
  sampling uses the existing samplers; manual mode uses an exact axis-angle and
  exact jitter levels. Rotation applies to both endpoints before source jitter.
  Full and gravity-consistent Z rotation are supported. Source jitter is disabled
  for one-sided interpolation. Physical SH rotation supports degrees zero and one.
- **Reproducibility:** augmentation is fixed while changing time, camera, patches,
  or colors. Resample increments only the augmentation seed. Baseline and current
  share flow noise and a fixed seeded permutation of configured serialization
  orders; order and display sampling have independent RNG streams.
- **Attention patches:** choose a block or a named order and drag the integer
  patch-size slider. Its range is 1 through the larger of 4096 and the largest
  configured patch size. Selecting a block restores its configured size.
  Serialization is cached, so slider changes only regroup existing orders.
- **References:** augmented references follow the training coordinates. Fixed
  unaugmented references preserve baseline neighborhoods; this is a viewer
  comparison option for standard DiPT. Source/target reference switching follows
  the interpolant, including the encoding/decoding midpoint.
- **Colors and rendering:** patch colors identify primary query patches. DiPT's
  final patch can borrow points from the previous patch; the padding count is
  reported separately. EDiPT masks its padding. Gaussian RGB shows the actual
  appearance, while patch colors also work in Gaussian rendering. Point display
  sampling never changes serialization; Gaussian rendering uses all loaded rows.

The default rendering is a point cloud. Select **GSplat background** or pass
`--initial_view_mode gsplat` to use CUDA Gaussian rendering. `--device`,
`--render_height`, clipping, background, and rasterization options match the
existing interpolant viewer. Comparison offsets are applied only for display.

Use `--dry_run` to check augmentation combinations, all analytic paths, and both
serialization references without starting a server. Focused regression tests:

```bash
python -m unittest discover -s tests -p 'test_visualize_*.py' -v
```

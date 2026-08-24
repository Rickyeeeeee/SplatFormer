# SR refactor isolation suite

This directory is self-contained experiment glue. It imports existing training and
dataset code read-only; it does not modify the current or pinned implementations.

The default output root is
`/project2/ricky/experiments/0822-refactor-isolation`. Override paths and the GPU
with environment variables documented by `./run_matrix.sh`.

```bash
# Inspect commands and overrides.
experiments/sr_refactor_isolation/run_matrix.sh

# Build the pinned export, train the missing native Nerfstudio pair, and verify fixtures.
GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh prepare

# Run E0-E3 with a 20-step smoke test before every full run; E4 is conditional.
GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh full

# Run every smoke and full experiment, including E4, without early stopping.
GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh all

# Rebuild JSON and Markdown conclusions from existing artifacts.
experiments/sr_refactor_isolation/run_matrix.sh summarize

# Repeat the guarded matrix on the confirmation scene.
GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh confirm
```

The launcher deliberately uses two Conda environments:

- `splatformer` for `ns-train` and the native Nerfstudio producer pair.
- `3dgs-sr` for gsplat 1.5.3, fixture conversion, E0-E4, tests, and summaries.

They can be overridden with `NERFSTUDIO_CONDA_ENV` and `SR_CONDA_ENV`.
`CONDA_BIN`, `SR_PYTHON_BIN`, and `NS_TRAIN_BIN` are also configurable.

Run compatibility tests with:

```bash
conda run -n 3dgs-sr python experiments/sr_refactor_isolation/test_compat_fixture.py
```

Every run has its own output and matching-cache directory. Existing completed runs
are reused, while incomplete/nonempty directories and mismatched fixtures are never
overwritten by default. After confirming that no process is still writing an
interrupted run, restart it with:

```bash
RESTART_INCOMPLETE=1 GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh full
```

The launcher moves both the incomplete run and its matching cache under
`$EXPERIMENT_ROOT/interrupted/` before starting cleanly, so the partial artifacts
remain available for inspection.

The full matrix normally stops as soon as E1 or E2 identifies the regression
source. To run the remaining E2/E3 diagnostics anyway, use:

```bash
CONTINUE_AFTER_FAILURE=1 GPU_ID=0 experiments/sr_refactor_isolation/run_matrix.sh full
```

This does not override a failed E0 control, and E4 remains conditional on the
E2-pass/E3-fail decision branch.

Alternatively, the `all` command runs E0-E4 unconditionally and ignores all
intermediate decision classifications. Existing completed runs are still reused,
and incomplete runs retain the `RESTART_INCOMPLETE=1` safety guard.

# Refactor plan

## Goal and scope

Focus flow matching on noise-to-GS generation, preparing for LR-GS-conditioned
noise-to-HR-GS generation. Preserve the existing `train-sr.py` and
`overfit-sr.py` workflows and all of their dependencies. Preserve current
noise-to-GS training, overfitting, visualization, and supported model options
unless a later step explicitly changes them.

Execute and review one stage at a time. Before deleting a shared file or
configuration, check its remaining imports, Gin registrations, launchers,
tests, and documentation. Each stage should leave the retained workflows usable.

## Stage 1: Remove obsolete SR workflows

**Status: implementation complete; validation blocked.** Syntax checks,
retained launcher Gin-file path checks, and the stale-reference search passed.
The default Python environment lacks PyTorch and Gin. Test runs in the existing
`srsplatformer` environment stalled in filesystem I/O before producing output;
rerun the focused tests and validate Gin parsing in a working project environment
before marking this stage done.

Remove the dedicated LR-GS-to-HR-GS flow-matching, SR interpolant, and separate
SR MSE workflows, together with their dedicated supporting files:

- Entry points: `train-sr-gsfm.py`, `overfit-sr-gsfm.py`,
  `overfit-sr-interpolants.py`, `overfit-sr-interpolants-edipt.py`,
  `train-sr-mse.py`, and `overfit-sr-mse.py`.
- Their dedicated files under `configs/`, `experiments/`, `scripts/`, and
  `tests/`, including old SR flow and interpolant visualizers.
- Old experiment scripts that only invoke the removed workflows. Keep
  `experiments/noise_to_gs.sh` and `experiments/gs_generation.sh`.

Keep `train-sr.py`, `overfit-sr.py`, and their related files. Keep shared
dataset, model, utility, and configuration files until their consumers have been
checked. Review mixed tests and diagnostics individually and retain coverage
that applies to supported workflows or models.

Validation: search for references to removed entry points and launchers; run
focused syntax and configuration checks for the retained SR and noise-to-GS
paths, plus applicable tests available in the environment.

## Stage 2: Clean up orphaned internals

Recheck references after Stage 1, then remove code used only by the removed
workflows, potentially including parts of `sr/flow.py`, `sr/alignment.py`,
`sr/matching.py`, `sr/densification.py`, `sr/edipt_interpolants.py`, the old
flow viewer, and `emd_assignment/`. Verify each dependency before removal.

Extract `attribute_mse` from `sr/interpolants.py` before removing its obsolete
interpolation paths: both noise-to-GS trainers use it. Preserve any other
functionality needed by retained SR, noise-to-GS, or supported diagnostics.

Validation: search for stale imports and Gin references; run focused tests for
the retained SR and noise-to-GS paths and any moved shared functionality.

## Stage 3: Refactor shared dataset and configuration

Refactor `dataset/GS_SR.py` and `configs/dataset/objaverse-sr.gin` around the
needs of both retained SR and noise-to-GS. Remove fitted-alignment options only
after confirming that neither retained workflow uses them. Preserve a clear
route for paired LR/HR loading for future conditioning; do not implement the
conditioned model as part of this cleanup.

Update affected launchers, Gin bindings, and tests together. Preserve existing
experiment behavior unless the stage explicitly requires a change.

Validation: check dataset construction and configuration for both retained SR
and noise-to-GS, and run focused dataset and training tests.

## Stage 4: Review optional backbones and diagnostics

Review optional noise-to-GS backbones and shared diagnostics separately from
the removal of obsolete SR workflows. The main noise-to-GS trainer uses Gaussian
DiT, while overfitting and checkpoint visualization also support PTV3, DiPT,
and GDPT. Keep those options unless a deliberate later decision narrows support.
Likewise, keep or adapt augmentation, viewer, and statistics tools according to
their actual use by the retained workflows.

Validation: test each retained model and diagnostic affected by changes, and
verify that its configs and launchers still resolve.

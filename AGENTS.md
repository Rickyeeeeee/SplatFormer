# SplatFormer Repository Guide

SplatFormer is a Python/PyTorch research codebase for Point Transformer-based
3D Gaussian Splatting refinement and super-resolution. It uses Gin
configuration for experiments and contains dataset loaders, model code,
utilities, training/evaluation scripts, and generated-data tooling.

## Working Principles

- Preserve existing experiment and configuration behavior unless the task
  explicitly requires changing it.
- Keep changes scoped to the task and do not modify unrelated worktree changes.
- Run focused syntax checks or tests appropriate to the component changed.

## Code Guidelines

- Add concise single-line comments for important code sections.
- Keep function calls on one line when the complete call fits naturally; do
  not expand a simple `function(arg)` call across three lines unnecessarily.
- Do not introduce local helper functions unless they are necessary. Put
  genuinely reusable functionality in the appropriate module under `utils/`.
- This is research code: prefer direct, readable implementations and minimize
  defensive `if` branches and exception handling unless they protect a
  meaningful failure mode.

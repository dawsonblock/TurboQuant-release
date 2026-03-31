# Packaging and hardening pass

This pass layers packaging and operational honesty on top of the blocker-fixed tree.

## Added

- `pyproject.toml` with setuptools build metadata and optional extras
- `MANIFEST.in` for source distribution contents
- `.github/workflows/static-ci.yml` for public packaging and syntax checks
- `scripts/validate_apple_silicon.sh` for real local MLX validation on Apple Silicon
- `docs/validation-local.md` documenting the runtime validation boundary

## Changed

- Bumped `turboquant.__version__` to `0.2.1`
- `mlx_lm.generate.maybe_turboquant_k_cache(...)` now emits deprecation warnings when callers try to control the production path via adapter-only knobs:
  - `turboquant_return_mode`
  - `turboquant_resid_scale_bits`
- README and integration docs now state that public CI does not certify MLX runtime behavior
- Added targeted tests for config-mismatch restore failure and deprecated dead-knob warnings

## Validation completed here

- `pyproject.toml` parsed successfully
- `python -m compileall turboquant mlx_lm tests` succeeded

## Not completed here

- No MLX runtime tests were run in this environment
- No Apple Silicon model smoke tests were run in this environment

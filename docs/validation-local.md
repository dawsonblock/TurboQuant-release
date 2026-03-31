# Local validation on Apple Silicon

Public CI in this repository checks packaging and static validation. It does not certify the MLX runtime path, because generic hosted runners are not Apple Silicon and do not provide a usable `mlx` environment.

## Two-track testing model

| Track | What it tests | Where it runs |
|---|---|---|
| **Static** (`make test-static`) | Import smoke, version consistency, source-checkout preflight, schema-level checks | Any platform |
| **MLX structural** (`make test-mlx`, `make test-structural`, `make test-path-proof`) | KVCompressor, pipeline, calibration, streaming attention, path proof | Apple Silicon only |
| **Runtime certification** (`make certify-apple-runtime`) | Artifact-producing release validation, smoke runs, benchmarks, and metric aggregation | Apple Silicon only |

## Quick start

```bash
# Static tests (safe everywhere)
make test-static

# Apple Silicon structural validation
make test-mlx
make test-structural
make test-path-proof

# Full runtime certification
make certify-apple-runtime
```

## Validation scripts

`./scripts/validate_apple_silicon.sh`

- local developer validation
- creates a fresh virtualenv
- installs the package in editable mode with Apple extras
- runs strict preflight
- runs the canonical MLX test surface

`./scripts/certify_apple_runtime.sh`

- release certification
- writes timestamped artifacts under `artifacts/runtime-cert/`
- runs strict preflight, structural tests, optional model smoke tests, benchmarks, and metric aggregation

## Legacy integration tests

`tests/integration/` is transitional coverage only if retained. It is not part of the canonical release-certification path. The release surface is `tests/unit/` plus `tests/integration_mlx/`.

## Manual smoke testing

For manual model smoke tests, run dense generation first, then the TurboQuant upgrade path on the same prompt and compare stability, memory use, and throughput.

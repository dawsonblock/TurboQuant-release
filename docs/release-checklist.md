# Release checklist

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. Custom Metal kernels are experimental and not part of the default supported runtime.

This is the minimum bar for calling a tagged snapshot technically credible. It is a release gate, not a wish list.

## Static gate

Run from a fresh checkout.

- `python scripts/preflight.py` passes
- `python -m compileall turboquant mlx_lm tests` passes
- `python -m build` produces both sdist and wheel
- `python -m pytest tests/unit_static -q` passes
- `README.md`, `docs/supported-surface.md`, and `docs/validation-local.md` agree on the supported slice

## Apple Silicon structural gate

Run on an Apple Silicon Mac with MLX installed.

- `python -m pytest tests/unit/ -q` passes
- `python -m pytest tests/integration_mlx -k "not llama and not gemma" -q` passes
- `make test-path-proof` passes

## Apple Silicon runtime gate

Use the certification script as the authoritative release runtime gate.

- `./scripts/certify_apple_runtime.sh` passes
- At least one Llama-family smoke run succeeds when `TQ_TEST_LLAMA_MODEL` is set
- At least one Gemma-family smoke run succeeds when `TQ_TEST_GEMMA_MODEL` is set
- Dense vs TurboQuant artifact outputs are saved under `artifacts/runtime-cert/<timestamp>/`
- Preflight JSON and JUnit outputs are present in the certification artifact directory

## Regression gate

- Non-power-of-two rotation remains orthogonal
- `residual_topk` survives the legacy adapter path
- state save/restore rejects config drift
- deprecated legacy knobs still warn instead of silently changing runtime behavior
- `python scripts/preflight.py` continues to work from a plain checkout

## Documentation gate

- No benchmark claim is labeled production unless it is backed by release data
- No CI badge implies MLX runtime certification on generic runners
- Supported models are named explicitly
- Validation commands in docs match the actual Makefile and Nox sessions

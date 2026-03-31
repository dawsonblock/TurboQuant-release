# Runtime Certification

## Purpose

This document describes the **narrow** Apple-Silicon MLX runtime
certification surface for TurboQuant v0.2.2.

Passing this certification means **only**:

> The TurboQuant compressed KV-cache path works for the supported
> Apple-MLX runtime on selected Llama-family and Gemma-family models,
> with reproducible artifacts, bounded quality loss, and measurable
> memory benefit.

It does **not** certify production readiness, cross-platform support,
all model families, custom Metal kernels, or distributed inference.

---

## Supported certification surface

| Dimension         | Value                                    |
| ----------------- | ---------------------------------------- |
| Hardware          | Apple Silicon Mac (arm64)                |
| OS                | macOS (version recorded per run)         |
| Python            | 3.11 (recommended; 3.9–3.11 supported)  |
| MLX               | ≥ 0.30.0 (exact version recorded)       |
| TurboQuant        | 0.2.2 (commit hash recorded)            |
| Llama model       | set via `TQ_TEST_LLAMA_MODEL` env var    |
| Gemma model       | set via `TQ_TEST_GEMMA_MODEL` env var    |
| Modes             | dense baseline, TurboQuant enabled       |
| Prompt classes    | short (5), medium (5), long (5)          |

## Unsupported surface

- Linux / Windows
- CUDA / ROCm
- General-purpose mlx_lm compatibility
- All model families
- Custom Metal kernel runtime
- Production readiness
- Distributed inference
- Training / fine-tuning

---

## Required environment

```text
macOS on Apple Silicon (M1/M2/M3/M4)
Python 3.11 (recommended)
A clean virtual environment
pip install -e '.[apple,test]'
```

## Environment variables

```bash
export TQ_TEST_LLAMA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
export TQ_TEST_GEMMA_MODEL="mlx-community/gemma-2-2b-it-4bit"
```

Use small quantized models to keep certification runs fast.

---

## Exact command

```bash
./scripts/certify_apple_runtime.sh
```

This single command runs the full certification pipeline. Artifacts are
written to `artifacts/runtime-cert/<timestamp>/`.

---

## Certification stages

| # | Stage                          | Tool                                                      |
| - | ------------------------------ | --------------------------------------------------------- |
| 1 | Strict preflight               | `python scripts/preflight.py --strict --json`             |
| 2 | Cache upgrade roundtrip        | `pytest tests/integration_mlx/test_cache_upgrade_roundtrip.py` |
| 3 | Streaming attention equivalence | `pytest tests/integration_mlx/test_streaming_attention_equivalence.py` |
| 4 | Llama smoke test               | `pytest tests/integration_mlx/test_llama_runtime_smoke.py` |
| 5 | Gemma smoke test               | `pytest tests/integration_mlx/test_gemma_runtime_smoke.py` |
| 6 | Long-context stability         | `pytest tests/integration_mlx/test_long_context_stability.py` |
| 7 | Dense vs TQ benchmarks         | `run_dense_vs_tq.py` × 3 prompt classes × 2 models       |
| 8 | Metric aggregation             | `collect_metrics.py`                                      |

---

## Artifacts produced

After a full run, `artifacts/runtime-cert/<timestamp>/` contains:

| File                             | Description                                |
| -------------------------------- | ------------------------------------------ |
| `preflight.json`                 | Machine-readable preflight result          |
| `junit_cache_roundtrip.xml`      | Cache roundtrip test results               |
| `junit_attention_equiv.xml`      | Attention equivalence test results         |
| `junit_llama_smoke.xml`          | Llama smoke test results                   |
| `junit_gemma_smoke.xml`          | Gemma smoke test results                   |
| `junit_long_context.xml`         | Long-context stability test results        |
| `*_dense.json`                   | Raw per-run dense benchmark results        |
| `*_turboquant.json`              | Raw per-run TurboQuant benchmark results   |
| `aggregate_runs.csv`             | All runs in tabular form                   |
| `certification_summary.json`     | Pass/fail rollup with memory/speed deltas  |

---

## Thresholds and pass/fail rules

A certification run **passes** only if all of the following are true:

### Structural

- Zero cache upgrade failures
- Zero state restore failures
- Zero sequence-offset mismatches

### Numerical (attention equivalence)

| Metric                | Threshold            |
| --------------------- | -------------------- |
| Cosine similarity     | ≥ 0.960              |
| Mean absolute error   | ≤ 0.06               |
| Max absolute error    | ≤ 0.25               |

Thresholds frozen after pilot run on Apple Silicon (M-series).
Observed cosine ~0.97 for 3-bit K + 4-bit V with Hadamard rotation.

### Runtime

- Zero crashes on supported models
- Zero empty generations in smoke tests
- No silent dense fallback when TurboQuant is requested

### Quality (bounded degradation)

- 100% of prompts complete without crash
- No catastrophic degeneration on any prompt
- Output length within tolerance of target
- Perplexity delta (TQ - dense) ≤ 0.5
- Mean KL divergence ≤ 0.1

### Performance

- Measurable memory reduction in TurboQuant long-context mode (≥ 25.0%)
- No evidence of catastrophic decode slowdown (degradation ≤ -25.0%)

---

## Threshold freeze process

1. Run `./scripts/certify_apple_runtime.sh` once (pilot run)
2. Inspect `certification_summary.json` and test output
3. Adjust thresholds in `test_streaming_attention_equivalence.py` to
   match reality (not fantasy)
4. Commit the frozen thresholds
5. Run certification again — this is the official pass/fail result

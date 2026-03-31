<div align="center">

# ⚡ TurboQuant

**3-bit KV-cache compression for Apple Silicon MLX LLMs**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.30.0%2B-orange?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple&logoColor=white)](https://apple.com/mac)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*3-bit keys · 4-bit values · deterministic rotation · top-k sparse residual · no numpy in the hot path*

</div>

---

## What is TurboQuant?

TurboQuant compresses the KV cache of transformer models running on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). It targets memory reduction first — a 3-bit key / 4-bit value configuration yields roughly **3.5–4× smaller KV cache** than float16 at 1 024 tokens.

> **Current status — Serious prototype.**
> Supported runtime paths are local Apple-Silicon validation for **Llama-family** and **Gemma-family** models.
> Custom Metal kernels are experimental and not part of the default supported runtime.
> Supported surface is documented in [`docs/supported-surface.md`](docs/supported-surface.md).
> Release gating is documented in [`docs/release-checklist.md`](docs/release-checklist.md).

### Illustrative compression numbers

These are local examples showing the shape of the compression trade-off — not release-certified benchmarks unless matched by saved runtime-certification artifacts for the exact commit and hardware used.

| Configuration | Tokens | Total MB | Bytes / Token | Ratio vs Dense |
|:---|:---:|:---:|:---:|:---:|
| **Dense** `float16` | 1 024 | 2.10 MB | 2 048 | 1.0× |
| TurboQuant k=4-bit, g=64 | 1 024 | 0.61 MB | 592 | **3.5× smaller** |
| TurboQuant k=3-bit, g=64 | 1 024 | 0.57 MB | 560 | **3.7× smaller** |
| TurboQuant k=2-bit, g=64 | 1 024 | 0.48 MB | 464 | **4.4× smaller** |

Artifact-backed release measurements belong under `artifacts/runtime-cert/<timestamp>/` and are authoritative over any README example.

---

## How it works

```
                       K  path
┌──────────┐    ┌─────────────────┐    ┌──────────────────────┐    ┌──────────┐
│ raw keys │───▶│ FixedRotation   │───▶│ GroupScalarQuantizer │───▶│  packed  │
│ [B,H,T,D]│    │ Hadamard/QR/id  │    │ N-bit, per-group     │    │  codes   │
└──────────┘    └─────────────────┘    └──────────────────────┘    └──────────┘
                                                  │ residual
                                                  ▼
                                       ┌──────────────────────┐
                                       │ encode_topk_residual │
                                       │ top-k values+indices │
                                       └──────────────────────┘

                       V  path
┌────────────┐    ┌──────────────────────┐    ┌──────────┐
│ raw values │───▶│ GroupScalarQuantizer │───▶│  packed  │
│ [B,H,T,D]  │    │ M-bit, per-group     │    │  codes   │
└────────────┘    └──────────────────────┘    └──────────┘

Decode K (streaming attention):
  packed_codes ──▶ dequant ──▶ + topk_residual ──▶ crop ──▶ [B,H,T,D]
  (queries are rotated with the same FixedRotation before the matmul)
```

**Key design choices:**

| Choice | Rationale |
|:---|:---|
| Deterministic Hadamard rotation | Isometry — preserves inner products, improves quantizer alignment |
| Per-group scalar quantization | Adaptive scale per 32–128 elements; no learned codebooks |
| Top-k sparse residual | Recovers the highest-error elements cheaply; configurable k |
| `return_mode="view"` default | Returns `TurboQuantKeysView` — no eagerly materialized dense tensor in the hot path |
| Versioned state schema | `schema_version: 2` enforced by `validate_state()`; forward-migration safe |
| Gate 2 model allowlist | `turboquant.runtime.support.SUPPORTED_FAMILIES` rejects unsupported architectures before any cache mutation |

---

## Repo layout

This repository is the **release surface** of TurboQuant — containing the patched source files, governance tooling, and test suite.

```
turboquant/
└── runtime/
    └── support.py          Central model-family allowlist (Gate 2)

mlx_lm/
└── models/
    └── cache.py            Patched KVCache: to_turboquant() default view mode

integrations/
└── mlx/
    └── upgrade.py          upgrade_cache_list() — Gate 2 wired, canonical upgrade API

tests/
├── helpers/
│   └── mlx_env.py          Shared MLX_SKIP_MARKER, HAS_MLX, IS_APPLE_SILICON
├── unit_static/            Structural governance tests (no MLX needed — run anywhere)
│   └── test_governance.py  5 static contracts (support gate, noxfile, doc truth)
├── compatibility/          KVCache contract tests (MLX required)
├── integration/            End-to-end integration tests (MLX required)
└── performance/            Memory/throughput regression tests (MLX required)

integrations_mlx/           All tests that import mlx directly (Apple Silicon only)

scripts/
└── write_cert_manifest.py  Writes structured cert_manifest.json for CI artifacts

tools/
└── audit_vendored_surface.py  Audits mlx_lm patch surface vs VENDORED_MLX_LM.md

docs/
├── architecture.md         Component map, data-flow, memory model
├── cache-format.md         State dict schema v2, uint32 packing layout
├── integration.md          Step-by-step wiring guide for new model families
├── evaluation.md           Metrics reference, benchmark workflow, thresholds
├── supported-surface.md    Supported model families and runtime paths
└── release-checklist.md    Release gating criteria
```

---

## Quick start

### Installation

```bash
# Clone and install in editable mode
git clone https://github.com/dawsonblock/TurboQuant-release.git
cd TurboQuant-release

pip install uv nox
uv pip install -e ".[dev]"
```

### Run the static governance tests (no Apple Silicon required)

```bash
nox -s tests_static
# or
pytest tests/unit_static/ -v
```

### Run all MLX tests (Apple Silicon required)

```bash
nox -s tests_mlx
# or
pytest tests/integration_mlx/ tests/integration/ tests/compatibility/ tests/performance/ -v
```

### Compress a KV cache

```python
from turboquant.runtime.support import assert_supported_model_family
from integrations.mlx.upgrade import upgrade_cache_list
from turboquant.config import TurboQuantConfig

# Gate 2: rejects unsupported architectures before any cache mutation
assert_supported_model_family("llama")

config = TurboQuantConfig(
    main_bits=3,
    group_size=64,
    v_enabled=True,   # V quantisation is enabled by default
    residual_k=2,
)

events = upgrade_cache_list(
    prompt_cache=prompt_cache,
    k_start=64,
    config=config,
    model_family="llama",
)
```

---

## Test strategy

TurboQuant enforces a hard boundary between tests that require MLX (Apple Silicon) and those that do not:

| Suite | Location | MLX required? | Purpose |
|:---|:---|:---:|:---|
| `unit_static` | `tests/unit_static/` | ❌ | Governance contracts — run in any CI |
| `integration_mlx` | `tests/integration_mlx/` | ✅ | Core MLX unit tests (quantizer, rotation, pipeline) |
| `integration` | `tests/integration/` | ✅ | End-to-end adapter + attention tests |
| `compatibility` | `tests/compatibility/` | ✅ | KVCache API contract tests |
| `performance` | `tests/performance/` | ✅ | Memory ratio regression tests |

All MLX-requiring test files carry `pytestmark = MLX_SKIP_MARKER` at module scope, sourced from `tests/helpers/mlx_env.py`. This ensures graceful skip on non-Apple-Silicon CI without per-test guards.

### Governance contracts (`test_governance.py`)

Five static contracts enforced by `tests/unit_static/test_governance.py`:

1. **`no_mlx_import_in_unit_static`** — static tests must be portable; no `mlx` import allowed.
2. **`noxfile_excludes_unit_from_mlx_session`** — `tests/unit/` must not appear in the `tests_mlx` nox session.
3. **`support_module_has_expected_families`** — `SUPPORTED_FAMILIES` must be exactly `{"llama", "gemma"}`.
4. **`unsupported_family_raises_unsupported_model_error`** — Gate 2 must reject unlisted families.
5. **`v_enabled_default_matches_architecture_doc`** — `architecture.md` must say "enabled by default" (regression guard against Phase 4 fix).

---

## Model support gate (Gate 2)

`turboquant/runtime/support.py` is the **single source of truth** for which model families have TurboQuant attention wiring and runtime-certification coverage.

```python
from turboquant.runtime.support import (
    SUPPORTED_FAMILIES,          # frozenset[str] — {"llama", "gemma"}
    is_supported_model_family,   # bool check, normalisation applied
    assert_supported_model_family,  # raises UnsupportedModelError if not supported
)
```

Rules for adding a new family:

1. Wire its attention module to dispatch through `turboquant_streaming_attention`.
2. Add runtime-certification coverage in `scripts/certify_apple_runtime.sh`.
3. Add the normalised family name to `SUPPORTED_FAMILIES`.
4. Update `docs/support_matrix.md`.
5. Update the `test_support_module_has_expected_families` governance test.

**Do not add a family before completing steps 1–3.**

---

## Vendored surface governance

```bash
# Audit the mlx_lm patch surface against VENDORED_MLX_LM.md
python tools/audit_vendored_surface.py

# Machine-readable output
python tools/audit_vendored_surface.py --json
```

Exit 0 = no violations. Exit 1 = undocumented modifications, missing markers, or missing files.

---

## Certification artifacts

```bash
# Write a structured cert_manifest.json after a certification run
python scripts/write_cert_manifest.py \
    --artifact-dir artifacts/runtime-cert/$(date +%Y%m%d_%H%M%S) \
    --passed 7 --failed 0 --total 7 \
    --turboquant-version 0.2.2
```

Produces:
```json
{
  "schema_version": "1",
  "turboquant_version": "0.2.2",
  "timestamp_utc": "2026-03-31T12:00:00Z",
  "platform": "darwin-arm64",
  "stages": {"passed": 7, "failed": 0, "total": 7},
  "result": "PASS",
  "files": ["preflight.json", "junit_cache_roundtrip.xml"]
}
```

---

## Requirements

| | |
|:---|:---|
| **Platform** | macOS · Apple Silicon (M1 / M2 / M3 / M4) |
| **Python** | ≥ 3.9 |
| **MLX** | ≥ 0.30.0, < 1.0.0 |
| **mlx-lm** | vendored v0.29.1 — see [`VENDORED_MLX_LM.md`](docs/VENDORED_MLX_LM.md) |

Static tests (`tests/unit_static/`) run on any platform without MLX installed.

---

## Component status

| Component | Status |
|:---|:---:|
| `turboquant.runtime.support` — Gate 2 allowlist | ✅ |
| `SUPPORTED_FAMILIES` — `{"llama", "gemma"}` | ✅ |
| `to_turboquant()` — `return_mode="view"` default | ✅ |
| `upgrade_cache_list()` — Gate 2 wired, `model_family` param | ✅ |
| Governance tests (`test_governance.py`) | ✅ 5 / 5 |
| Vendored surface audit (`audit_vendored_surface.py`) | ✅ |
| Cert manifest writer (`write_cert_manifest.py`) | ✅ |
| MLX skip markers — all non-static test files | ✅ |
| `noxfile.py` — `tests/unit/` excluded from `tests_mlx` | ✅ |
| `docs/architecture.md` — `v_enabled` claim accurate | ✅ |
| Gemma streaming attention | ✅ wired |
| Llama streaming attention | ✅ wired |
| Other architectures (Mistral, Phi, …) | ⬜ needs per-arch patch |

---

## Documentation

| Doc | Contents |
|:---|:---|
| [`docs/architecture.md`](docs/architecture.md) | Component map, data-flow, memory model |
| [`docs/cache-format.md`](docs/cache-format.md) | State dict schema v2, uint32 packing layout |
| [`docs/integration.md`](docs/integration.md) | Step-by-step wiring guide for new model families |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics reference, benchmark workflow, thresholds |
| [`docs/supported-surface.md`](docs/supported-surface.md) | Supported models and runtime paths |
| [`docs/release-checklist.md`](docs/release-checklist.md) | Release gating criteria |

---

## Limitations

- **Quality gated but not yet measured at scale** — `run_quality_eval.py` enforces Δppl ≤ 0.5 and mean_kl ≤ 0.1 gates. Run with model weights to validate.
- **Apple Silicon only for MLX paths** — all MLX-dependent tests and benchmarks require an M-series Mac.
- **Hadamard is O(d²)** — for very large head-dims, `rotation="identity"` is faster with marginally worse compression.
- **Metal kernels are opt-in** — set `TQ_USE_METAL=1` to enable native Metal shaders. Default paths use `mx.compile`.

---

<div align="center">

Made for Apple Silicon · Built with [MLX](https://github.com/ml-explore/mlx)

</div>

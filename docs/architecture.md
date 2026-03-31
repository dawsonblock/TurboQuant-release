# TurboQuant Architecture

> **Status**: research-grade  
> **Last updated**: 2025

---

## 1. Overview

TurboQuant compresses the KV (key-value) cache of transformer language models
by quantising both K and V heads to low bit-widths (typically 3–4 bits) and
decompressing on-the-fly during attention.  The goal is to cut memory
bandwidth at decode time on Apple Silicon (MLX backend) while preserving
model quality.

```text
Input token
     │
     ▼
┌────────────────────────────────────────────────┐
│                   Model Layer                  │
│                                                │
│  Linear projections  →  Q, K, V               │
│       │                   │                    │
│       │           ┌────────────────────┐       │
│       │           │    KVCompressor    │       │
│       │           │                    │       │
│       │           │                   │       │
│       │           │  encode_k()  encode_v()│       │
│       │           │  k_packed   v_packed │       │
│       │           └──────────┬─────────┘       │
│       │                      │ iter_blocks()    │
│       ▼                      ▼                 │
│  rotate_q() ──► streaming softmax attention    │
│                      │                         │
│                      ▼                         │
│                   Output                       │
└────────────────────────────────────────────────┘
```text
---

## 2. Package structure

```text
turboquant/
├── config.py               # TurboQuantConfig dataclass (production schema)
├── core/
│   ├── pipeline.py         # TurboQuantPipeline — per-layer quantise/dequantise
│   ├── quantizer.py        # group quantisation + bit-packing primitives
│   ├── rotation.py         # Hadamard / identity / random orthogonal rotation
│   └── residual.py         # top-k sparse residual encoder
├── runtime/
│   ├── kv_interface.py     # KVCompressor — the canonical compression class
│   ├── attention.py        # turboquant_streaming_attention (shared adapter)
│   └── state.py            # STATE_SCHEMA_VERSION + validate_state()
├── eval/
│   ├── __init__.py
│   ├── perplexity.py
│   ├── generation_drift.py
│   └── memory.py
└── tests/                  # compat stub (canonical tests live in tests/unit/)
```text
```text
mlx_lm/
├── models/
│   ├── cache.py            # KVCache, TurboQuantKCache (adapter), helpers
│   └── gemma.py            # Gemma attention wired to streaming attention
├── generate.py             # generate_step
└── upgrade.py              # (in integrations/mlx) canonical upgrade entry point

```text
---

## 3. Key components

### 3.1 TurboQuantConfig (`turboquant/config.py`)

Production configuration dataclass.  Fields:

| field | default | description |
|---|---|---|
| `k_bits` | 3 | bits per key element |
| `k_group_size` | 64 | keys quantised in groups of this size |
| `v_bits` | 4 | bits per value element |
| `v_group_size` | 64 | values quantised in groups of this size |
| `v_enabled` | True | whether to quantise V (K is always quantised) |
| `rotation` | `"hadamard"` | pre-rotation: `"identity"`, `"hadamard"`, `"random_orthogonal"` |
| `residual_topk` | 2 | top-k residual elements stored per group (0 = disabled) |
| `block_tokens` | 256 | streaming attention block size |
| `allocation_step` | 512 | how many tokens to pre-allocate at a time |
| `eps` | 1e-6 | numerical stability floor |
| `scale_dtype` | `"float16"` | dtype for scale factors |
| `v_scale_dtype` | `"float16"` | dtype for V scale factors |

> **Legacy note**: `mlx_lm.models.cache.TurboQuantConfig` uses old field names
> (`main_bits`, `group_size`, `return_mode`, …).  It is a shim that maps to the
> production dataclass — see [integration.md](integration.md).

### 3.2 KVCompressor (`turboquant/runtime/kv_interface.py`)

The single implementation of KV quantisation.  Lifecycle:

1. `__init__(config)` — creates a `TurboQuantPipeline` (lazy; no MLX arrays yet)
2. `update_and_fetch(k, v)` → `(TurboQuantKeysView, v_or_none)` — appends new
   tokens, quantises them, returns a view object for streaming attention
3. `iter_rotated_kv_blocks(view, block_tokens)` — yields `(s, e, k_rot, v_blk)`
   blocks; callers accumulate online-softmax accumulators
4. `rotate_queries_for_attention(q)` — applies the same rotation as K so Q and
   K are in the same space
5. `state()` / `from_state(state, config)` — serialise/restore (schema v2)
6. `memory_breakdown()` — returns per-buffer byte counts + total
7. `trim(n)` — decrease `offset` by n (for prompt trimming)

### 3.3 Streaming attention (`turboquant/runtime/attention.py`)

`turboquant_streaming_attention(queries, keys_view, *, scale)`:

- Rotates queries via `cache.rotate_queries_for_attention(q)`
- Iterates over K/V blocks with `iter_rotated_kv_blocks`
- Accumulates attention scores with the **online softmax** (2-accumulator)
  algorithm: tracks running max `m` and log-sum-exp `lse` without materialising
  the full attention matrix

`maybe_turboquant_attention(q, k, v, mask, scale, fallback, cache)`:
- Dispatches: if `isinstance(k, TurboQuantKeysView)` → streaming path;
  else → `fallback(q, k, v, mask, scale)`

### 3.4 Rotation (`turboquant/core/rotation.py`)

Three modes:

| mode | description | cost |
|---|---|---|
| `"identity"` | no rotation (fastest, least entropy spreading) | O(1) |
| `"hadamard"` | dense Hadamard matrix via NumPy → `mx.array` | O(d²) |
| `"random_orthogonal"` | random orthogonal via SVD at init | O(d²) |

> **Note**: the Hadamard implementation uses a dense matrix multiply, not the
> fast Walsh-Hadamard transform butterfly (O(d log d)).

### 3.5 TurboQuantPipeline (`turboquant/core/pipeline.py`)

Wraps the per-layer encode/decode primitives. Now features an explicit `.build()` phase that pre-allocates caches, quantizers, and fixed rotations ahead-of-time to avoid any branch-heavy lazy initialization during hot-path execution.

### 3.6 State schema (`turboquant/runtime/state.py`)

State dicts carry `schema_version: 2`.  `validate_state(state, config)` checks:

- `schema_version` present and equal to `STATE_SCHEMA_VERSION`
- Required scalar keys: `offset`, `d_head`, `d_pad`, `v_dim`, `v_pad`
- Token dimension of `k_packed` ≥ `offset`
- Group count consistent with config

---

## 4. Data flow: one decode step

```text
q [B, H_q, 1, d]    k [B, H_kv, 1, d]    v [B, H_kv, 1, d]
        │                    │                    │
        │            KVCompressor.update_and_fetch(k, v)
        │                    │
        │            encode_k() → k_packed  [B, H, T, n_words] uint32
        │            encode_v() → v_packed  [B, H, T, n_words] uint32
        │                    │
        │            TurboQuantKeysView (lazy proxy)
        │                    │
        └──── rotate_queries_for_attention(q) ──── R·q
                             │
              iter_rotated_kv_blocks(view)
                   ┌─────────┴──────────┐
                   │   for each block   │
                   │   decode_k() → k_blk  (rotated)
                   │   decode_v() → v_blk
                   │   scores = q_rot @ k_blk.T / scale
                   │   online-softmax update
                   └───────────────────┘
                             │
                          output [B, H_q, 1, d]
```text
---

## 5. Memory model

For a sequence of T tokens, N KV heads, head dimension d, at b bits/group of g:

$$\text{bytes}_{K} \approx \frac{b \cdot N \cdot T \cdot d}{8} + \frac{2 \cdot N \cdot T \cdot d}{g \cdot 8} \cdot \text{sizeof}(\text{scale\_dtype})$$

The V component is analogous.  At 3-bit K + 4-bit V with group=64 and float16 scales, TurboQuant uses roughly **4–5×** less memory than float16 dense KV for sequences > 512 tokens (see `benchmarks/bench_memory_footprint.py`).

---

## 6. Limitations

- **Hadamard** is O(d²) — not the butterfly O(d log d).  For large d (≥ 128) this adds noticeable encode overhead.
- **Residual** is top-k sparse; the legacy sign-sketch residual is not supported in the production path.
- **V quantisation** is enabled by default (`v_enabled=True`).  Family-specific
  integrations may disable it explicitly when V quantisation degrades quality on
  a given model.  There is no family-level override in the core config; callers
  must pass `v_enabled=False` explicitly.
- **Llama** wiring is complete (see [integration.md](integration.md)).  Other model families require a one-line `maybe_turboquant_attention` dispatch.


## 7. Validation boundary

This repository now includes packaging metadata and public static CI, but that CI does not certify MLX runtime behavior. Real runtime validation still requires an Apple Silicon Mac with `mlx` installed. Use `scripts/validate_apple_silicon.sh` for the supported local validation path.

## 8. Experimental & Native Acceleration
TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. Custom Metal kernels are experimental and not part of the default supported runtime.

- **MLX JIT Core Compilation (Supported Default)**: Fallback stream topologies (`dequantize_groups`, `decode_k_fallback`) aggressively compile inline Python iterators directly into static C++ execution trees (`mx.compile(fn, shapeless=False)`). Configuration constants (e.g., bit-widths, padding sizes) are aggressively injected at the compilation cache layer `_DEQUANT_CACHE[key]` effectively stripping dispatch latency.
- **Metal Shader Injection (Experimental)**: Raw logic is ported directly to Apple Silicon via explicit `mx.fast.metal_kernel` C++ templates (`decode_k.metal`). Heavy inner accumulation loops (`resid_idx` matches) are natively untrolled utilizing statically mapped compilation signatures `[("BITS", config.k_bits)]`.


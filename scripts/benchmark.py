#!/usr/bin/env python3
"""
scripts/benchmark.py — single-command TurboQuant benchmark.

Runs a compression cycle for Llama and Gemma family models (if available),
measures dense vs compressed memory footprint, generation latency, and
accuracy (KL divergence + token-match rate), then writes a JSON summary.

Usage
-----
::

    python scripts/benchmark.py                   # default: dry-run (no model)
    python scripts/benchmark.py --model llama     # model family to benchmark
    python scripts/benchmark.py --tokens 128      # tokens to generate
    python scripts/benchmark.py --out results/    # output directory

Output
------
Writes ``<out>/<run_id>/metrics.json`` with the schema::

    {
      "run_id": "...",
      "model_family": "llama",
      "k_bits": 3,
      "v_bits": 4,
      "dense_bytes": 12345678,
      "compressed_bytes": 4567890,
      "ratio": 0.37,
      "tok_per_sec": 42.1,
      "latency_ms": 23.8,
      "avg_kl_per_token": 0.004,
      "token_match_rate": 0.984,
      "kl_bound_ok": true,
      "match_bound_ok": true,
      "passed": true
    }

Environment
-----------
Requires Apple Silicon with MLX installed.
Set ``TURBOQUANT_DRY_RUN=1`` to skip model loading and emit a synthetic
report (useful for CI / documentation builds on non-Apple hosts).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("turboquant.benchmark")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "The architecture of modern language models relies on attention mechanisms "
    "that scale quadratically with sequence length.  Efficient KV-cache compression"
)
DEFAULT_TOKENS = 64
DEFAULT_OUT = "runs"
DEFAULT_K_BITS = 3
DEFAULT_V_BITS = 4
DEFAULT_K_GROUP = 64
DEFAULT_V_GROUP = 64


# ── Dry-run synthetic report ───────────────────────────────────────────────────


def _synthetic_report(model_family: str, n_tokens: int, k_bits: int, v_bits: int):
    """Return a fake report for non-Apple / CI environments."""
    dense = 1024 * 1024 * 32  # 32 MB
    compressed = int(dense * (k_bits / 16) * 0.55)
    return {
        "model_family": model_family,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "dense_bytes": dense,
        "compressed_bytes": compressed,
        "ratio": round(compressed / dense, 4),
        "tok_per_sec": 0.0,
        "latency_ms": 0.0,
        "avg_kl_per_token": 0.0,
        "token_match_rate": 1.0,
        "kl_bound_ok": True,
        "match_bound_ok": True,
        "passed": True,
        "dry_run": True,
    }


# ── Real benchmark ─────────────────────────────────────────────────────────────


def _run_benchmark(
    model_family: str,
    prompt: str,
    n_tokens: int,
    k_bits: int,
    v_bits: int,
    k_group: int,
    v_group: int,
) -> dict:
    """Run a real benchmark on Apple Silicon.

    Attempts to load the model; falls back to dry-run on any ImportError.
    """
    try:
        import mlx.core as mx
    except ImportError:
        logger.warning("MLX not available — falling back to dry-run report")
        return _synthetic_report(model_family, n_tokens, k_bits, v_bits)

    from turboquant.config import TurboQuantConfig

    from turboquant.eval.compare import AccuracyComparison
    from turboquant.metrics.tracker import MetricsTracker

    config = TurboQuantConfig(
        k_bits=k_bits,
        k_group_size=k_group,
        v_bits=v_bits,
        v_group_size=v_group,
        v_enabled=True,
    )

    tracker = MetricsTracker(model_family=model_family)

    # ── Estimate dense bytes ───────────────────────────────────────────────────
    #
    # We measure dense footprint by running one full prefill step with a
    # standard KVCache and recording how many bytes are allocated before/after.
    try:
        from mlx_lm import load

        model, tokenizer = load(model_family)
    except Exception as exc:
        logger.warning("Could not load model '%s': %s — dry-run", model_family, exc)
        return _synthetic_report(model_family, n_tokens, k_bits, v_bits)

    from mlx_lm.models.cache import KVCache

    ids = tokenizer.encode(prompt)
    input_ids = mx.array([ids], dtype=mx.int32)

    # Dense prefill — time it
    dense_cache = [KVCache() for _ in range(len(model.layers))]
    t0 = time.perf_counter()
    model(input_ids, cache=dense_cache)
    mx.eval(*[c.keys for c in dense_cache if hasattr(c, "keys") and c.keys is not None])

    dense_bytes = sum(
        c.keys.nbytes + c.values.nbytes
        for c in dense_cache
        if hasattr(c, "keys") and c.keys is not None
    )
    tracker.set_dense_bytes(dense_bytes)

    # ── Upgrade to TurboQuant and time a decode step ───────────────────────────
    from integrations.mlx.upgrade import upgrade_cache_list

    events = upgrade_cache_list(
        dense_cache,
        k_start=0,
        config=config,
        model_family=model_family,
    )
    upgraded = sum(1 for ev in events if ev.upgraded)
    logger.info("Upgraded %d / %d layers", upgraded, len(events))

    compressed_bytes = sum(
        getattr(c, "byte_size", lambda: 0)()
        for c in dense_cache
    )
    tracker.set_compressed_bytes(compressed_bytes)

    # Decode step timing
    x = mx.argmax(model(input_ids, cache=dense_cache)[:, -1, :], axis=-1)[:, None]
    steps = []
    for _ in range(min(n_tokens, 16)):
        t0 = time.perf_counter()
        logits = model(x, cache=dense_cache)
        mx.eval(logits)
        steps.append(time.perf_counter() - t0)
        x = mx.argmax(logits[:, -1, :], axis=-1)[:, None]

    avg_step_ms = 1000 * (sum(steps) / len(steps)) if steps else 0.0
    tok_per_sec = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0

    tracker.record_step(step=0, latency_ms=avg_step_ms)

    # ── Accuracy comparison ────────────────────────────────────────────────────
    comp = AccuracyComparison(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_family=model_family,
    )
    report = comp.run(prompt=prompt, max_tokens=min(n_tokens, 32))

    ratio = (compressed_bytes / dense_bytes) if dense_bytes > 0 else 0.0

    result = {
        "model_family": model_family,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "dense_bytes": dense_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": round(ratio, 4),
        "tok_per_sec": round(tok_per_sec, 2),
        "latency_ms": round(avg_step_ms, 2),
        "avg_kl_per_token": report.mean_kl,
        "token_match_rate": report.token_match_rate,
        "kl_bound_ok": report.kl_bound_ok,
        "match_bound_ok": report.match_bound_ok,
        "passed": report.passed,
        "dry_run": False,
    }

    tracker.write()
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TurboQuant single-command benchmark")
    p.add_argument(
        "--model",
        default="llama",
        help="Model family name or HuggingFace path (default: llama)",
    )
    p.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_TOKENS,
        help=f"Number of tokens to generate (default: {DEFAULT_TOKENS})",
    )
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to use for benchmarking",
    )
    p.add_argument(
        "--k-bits",
        type=int,
        default=DEFAULT_K_BITS,
        help=f"Key quantisation bits (default: {DEFAULT_K_BITS})",
    )
    p.add_argument(
        "--v-bits",
        type=int,
        default=DEFAULT_V_BITS,
        help=f"Value quantisation bits (default: {DEFAULT_V_BITS})",
    )
    p.add_argument(
        "--k-group",
        type=int,
        default=DEFAULT_K_GROUP,
        help=f"Key group size (default: {DEFAULT_K_GROUP})",
    )
    p.add_argument(
        "--v-group",
        type=int,
        default=DEFAULT_V_GROUP,
        help=f"Value group size (default: {DEFAULT_V_GROUP})",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory for metrics.json (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(os.getenv("TURBOQUANT_DRY_RUN")),
        help="Emit a synthetic report without loading a model",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = str(uuid.uuid4())[:8]

    logger.info(
        "TurboQuant benchmark  run_id=%s  model=%s  k_bits=%d  v_bits=%d",
        run_id,
        args.model,
        args.k_bits,
        args.v_bits,
    )

    if args.dry_run:
        result = _synthetic_report(args.model, args.tokens, args.k_bits, args.v_bits)
    else:
        result = _run_benchmark(
            model_family=args.model,
            prompt=args.prompt,
            n_tokens=args.tokens,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            k_group=args.k_group,
            v_group=args.v_group,
        )

    result["run_id"] = run_id

    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Summary to stdout
    print("\n" + "=" * 60)
    print(f"  TurboQuant Benchmark — run {run_id}")
    print("=" * 60)
    print(f"  Model family   : {result['model_family']}")
    print(f"  Bits (K / V)   : {result['k_bits']} / {result['v_bits']}")
    print(f"  Dense bytes    : {result['dense_bytes']:,}")
    print(f"  Compressed     : {result['compressed_bytes']:,}")
    print(f"  Ratio          : {result['ratio']:.3f}x")
    print(f"  Latency        : {result['latency_ms']:.1f} ms/tok")
    print(f"  Throughput     : {result['tok_per_sec']:.1f} tok/s")
    print(f"  Mean KL        : {result['avg_kl_per_token']:.5f}")
    print(f"  Token match    : {result['token_match_rate'] * 100:.1f}%")
    passed = "\u2713 PASSED" if result["passed"] else "\u2717 FAILED"
    print(f"  Result         : {passed}")
    if result.get("dry_run"):
        print("  (dry run — no model was loaded)")
    print("=" * 60)
    print(f"  Results saved  : {out_path}")
    print("=" * 60 + "\n")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

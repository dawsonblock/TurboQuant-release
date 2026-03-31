#!/usr/bin/env python3
"""
run_quality_eval — perplexity and KL-divergence quality gate.

Runs the existing turboquant.eval.perplexity and turboquant.eval.generation_drift
modules across a prompt set and produces a structured JSON artifact with
pass/fail verdicts against configurable quality thresholds.

Usage::

    python benchmarks/runtime_cert/run_quality_eval.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --prompt-file benchmarks/runtime_cert/prompts/short.jsonl \
        --prompt-class short \
        --output-dir artifacts/runtime-cert/20260329_120000 \
        --max-delta-ppl 0.5 \
        --max-mean-kl 0.1

Exit code 0 if all quality gates pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.runtime_cert.utils import (
    collect_environment_metadata,
    ensure_artifact_dir,
    load_prompts,
    now_utc_iso,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TurboQuant quality evaluation gate")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--prompt-file", required=True, help="Path to a .jsonl prompt file")
    p.add_argument("--prompt-class", required=True, choices=["short", "medium", "long"])
    p.add_argument("--output-dir", required=True, help="Directory for artifacts")
    p.add_argument(
        "--max-delta-ppl",
        type=float,
        default=0.5,
        help="Maximum allowed Δperplexity (TQ − dense). Default: 0.5",
    )
    p.add_argument(
        "--max-mean-kl",
        type=float,
        default=0.1,
        help="Maximum allowed mean KL divergence. Default: 0.1",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_model(model_id: str):
    """Load model and tokenizer via mlx_lm."""
    from mlx_lm import load

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)
    return model, tokenizer


def _tokenize_for_eval(tokenizer, prompt_text: str):
    """Tokenize a prompt string into [1, T] input_ids."""
    import mlx.core as mx

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt_text
    else:
        text = prompt_text
    ids = tokenizer.encode(text)
    return mx.array(ids).reshape(1, -1), len(ids)


def main() -> int:
    args = _parse_args()
    out_dir = ensure_artifact_dir(args.output_dir)
    prompts = load_prompts(args.prompt_file)

    import mlx.core as mx

    mx.random.seed(args.seed)

    model, tokenizer = _load_model(args.model)

    from turboquant.config import TurboQuantConfig
    from turboquant.eval.generation_drift import drift_report
    from turboquant.eval.perplexity import perplexity_report

    tq_config = TurboQuantConfig(
        k_bits=3,
        k_group_size=64,
        rotation="hadamard",
        residual_topk=2,
        v_bits=4,
        v_group_size=64,
        v_enabled=True,
        block_tokens=256,
    )

    env = collect_environment_metadata(
        model=args.model,
        mode="quality_eval",
    )

    results: list[dict] = []
    all_pass = True

    for prompt in prompts:
        pid = prompt["id"]
        text = prompt["text"]
        print(f"  [{pid}] tokenizing ...", end=" ", flush=True)

        input_ids, n_tokens = _tokenize_for_eval(tokenizer, text)

        if n_tokens < 4:
            print(f"SKIP (only {n_tokens} tokens)")
            continue

        # Perplexity comparison
        t0 = time.perf_counter()
        ppl = perplexity_report(
            model=model,
            input_ids=input_ids,
            turboquant_config=tq_config,
            k_start=0,
        )
        ppl_sec = time.perf_counter() - t0

        # KL divergence
        t1 = time.perf_counter()
        drift = drift_report(
            model=model,
            input_ids=input_ids,
            turboquant_config=tq_config,
            k_start=0,
        )
        drift_sec = time.perf_counter() - t1

        # Apply quality gates
        ppl_pass = True
        kl_pass = True

        delta_ppl = ppl.get("delta_ppl")
        if delta_ppl is not None and delta_ppl > args.max_delta_ppl:
            ppl_pass = False

        mean_kl = drift.get("mean_kl", 0.0)
        if mean_kl > args.max_mean_kl:
            kl_pass = False

        row_pass = ppl_pass and kl_pass
        if not row_pass:
            all_pass = False

        status = "PASS" if row_pass else "FAIL"
        print(
            f"{status} | ppl_dense={ppl['dense_ppl']:.3f} "
            f"ppl_tq={ppl.get('tq_ppl', 'N/A')} "
            f"Δppl={delta_ppl} "
            f"mean_kl={mean_kl:.6f} "
            f"({ppl_sec + drift_sec:.1f}s)"
        )

        results.append(
            {
                "prompt_id": pid,
                "prompt_class": args.prompt_class,
                "n_tokens": n_tokens,
                "dense_ppl": ppl["dense_ppl"],
                "tq_ppl": ppl.get("tq_ppl"),
                "delta_ppl": delta_ppl,
                "mean_kl": mean_kl,
                "max_kl": drift.get("max_kl", 0.0),
                "ppl_pass": ppl_pass,
                "kl_pass": kl_pass,
                "pass": row_pass,
                "ppl_seconds": round(ppl_sec, 2),
                "drift_seconds": round(drift_sec, 2),
            }
        )

    # Write artifact
    summary = {
        "timestamp": now_utc_iso(),
        "environment": env,
        "model": args.model,
        "prompt_class": args.prompt_class,
        "thresholds": {
            "max_delta_ppl": args.max_delta_ppl,
            "max_mean_kl": args.max_mean_kl,
        },
        "seed": args.seed,
        "results": results,
        "all_pass": all_pass,
        "n_prompts": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "n_fail": sum(1 for r in results if not r["pass"]),
    }

    fname = f"quality_eval_{args.prompt_class}.json"
    write_json(out_dir / fname, summary)
    print(f"\nQuality eval written to {out_dir / fname}")

    if all_pass:
        print(f"✓ ALL {len(results)} prompts passed quality gates")
        return 0
    else:
        n_fail = summary["n_fail"]
        print(f"✗ {n_fail}/{len(results)} prompt(s) FAILED quality gates")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

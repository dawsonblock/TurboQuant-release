#!/usr/bin/env python3
"""
run_dense_vs_tq — paired dense / TurboQuant benchmark runner.

Executes paired generation runs for one model across short, medium, and
long prompt sets, writing one raw JSON artifact per run to disk.

Usage::

    python benchmarks/runtime_cert/run_dense_vs_tq.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --prompt-file benchmarks/runtime_cert/prompts/short.jsonl \
        --prompt-class short \
        --output-dir artifacts/runtime-cert/20260329_120000 \
        --max-new-tokens 64 \
        --seed 42 \
        --mode both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.runtime_cert.utils import (
    build_run_result,
    collect_environment_metadata,
    ensure_artifact_dir,
    load_prompts,
    make_run_id,
    measure_peak_memory_bytes,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense vs TurboQuant paired benchmark")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--prompt-file", required=True, help="Path to a .jsonl prompt file")
    p.add_argument("--prompt-class", required=True, choices=["short", "medium", "long"])
    p.add_argument(
        "--output-dir", required=True, help="Directory for raw JSON artifacts"
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=["dense", "turboquant", "both"],
        default="both",
        help="Run mode(s)",
    )
    p.add_argument(
        "--turboquant-config-json", default=None, help="Path to TQ config JSON override"
    )
    return p.parse_args()


def _load_model(model_id: str):
    """Load model and tokenizer via mlx_lm."""
    from mlx_lm import load

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)
    return model, tokenizer


def _tokenize(tokenizer, prompt_text: str):
    """Tokenize a prompt string."""
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
    return mx.array(ids), len(ids)


def run_single_generation(
    model,
    tokenizer,
    prompt_text: str,
    *,
    model_id: str,
    prompt_id: str,
    prompt_class: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    mode: str,  # "dense" or "turboquant"
    tq_config_dict: dict | None = None,
) -> dict:
    """Run a single generation and return a structured result dict."""
    import mlx.core as mx

    from mlx_lm.generate import generate_step

    # Deterministic seeding — must precede every run for reproducibility
    mx.random.seed(seed)

    prompt_tokens, prompt_len = _tokenize(tokenizer, prompt_text)

    gen_kwargs: dict = dict(max_tokens=max_tokens)

    tq_active = mode == "turboquant"
    if tq_active:
        gen_kwargs["turboquant_k_start"] = 0
        gen_kwargs["turboquant_main_bits"] = 3
        gen_kwargs["turboquant_group_size"] = 64
        gen_kwargs["turboquant_rotation"] = "hadamard"
        gen_kwargs["turboquant_residual_topk"] = 2
        gen_kwargs["turboquant_v_bits"] = 4
        gen_kwargs["turboquant_v_group_size"] = 64
        gen_kwargs["turboquant_v_enabled"] = True
        gen_kwargs["turboquant_block_tokens"] = 256

    env = collect_environment_metadata(
        model=model_id,
        mode=mode,
        turboquant_config=tq_config_dict,
    )
    run_id = make_run_id(model_id, prompt_id, mode)

    error_text = None
    status = "ok"
    tokens_out: list[int] = []
    output_text = ""
    prefill_sec = 0.0
    decode_sec = 0.0

    try:
        t0 = time.perf_counter()

        first_token_time = None
        sampler = (
            (lambda x: mx.random.categorical(x * (1.0 / temperature)))
            if temperature > 0
            else (lambda x: mx.argmax(x, axis=-1))
        )
        for token, _ in generate_step(
            prompt_tokens, model, sampler=sampler, **gen_kwargs
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                prefill_sec = first_token_time - t0
            tokens_out.append(int(token))
            if len(tokens_out) >= max_tokens:
                break

        total_sec = time.perf_counter() - t0
        decode_sec = total_sec - prefill_sec
        output_text = tokenizer.decode(tokens_out)

    except Exception as exc:
        import traceback

        traceback.print_exc()

        status = "error"
        error_text = str(exc)
        total_sec = time.perf_counter() - t0

    gen_count = len(tokens_out)
    tps = gen_count / decode_sec if decode_sec > 0 else 0.0
    peak_mem = measure_peak_memory_bytes()

    return build_run_result(
        run_id=run_id,
        environment=env,
        model=model_id,
        mode=mode,
        prompt_id=prompt_id,
        prompt_class=prompt_class,
        prompt_length=prompt_len,
        generated_tokens=gen_count,
        prefill_seconds=prefill_sec,
        decode_seconds=decode_sec,
        total_seconds=total_sec,
        tokens_per_second=tps,
        peak_memory_bytes=peak_mem,
        turboquant_active=tq_active,
        turboquant_config=tq_config_dict,
        status=status,
        error=error_text,
        output_preview=output_text,
        seed=seed,
        temperature=temperature,
    )


def main() -> int:
    args = _parse_args()
    out_dir = ensure_artifact_dir(args.output_dir)
    prompts = load_prompts(args.prompt_file)

    model, tokenizer = _load_model(args.model)

    modes = []
    if args.mode in ("dense", "both"):
        modes.append("dense")
    if args.mode in ("turboquant", "both"):
        modes.append("turboquant")

    tq_cfg = None  # could load from --turboquant-config-json if provided

    results = []
    for prompt in prompts:
        pid = prompt["id"]
        text = prompt["text"]
        for mode in modes:
            print(f"  [{mode}] prompt={pid} ...", end=" ", flush=True)
            result = run_single_generation(
                model,
                tokenizer,
                text,
                model_id=args.model,
                prompt_id=pid,
                prompt_class=args.prompt_class,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                mode=mode,
                tq_config_dict=tq_cfg,
            )
            fname = f"{result['run_id']}.json"
            write_json(out_dir / fname, result)
            results.append(result)
            tok = result["generated_tokens"]
            secs = result["total_seconds"]
            status = result["status"]
            print(f"{status} | {tok} tokens | {secs:.2f}s")

    print(f"\n{len(results)} runs written to {out_dir}")

    # Return non-zero if any run failed
    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        print(f"WARNING: {len(failures)} run(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

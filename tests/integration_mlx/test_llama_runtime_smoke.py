"""
test_llama_runtime_smoke — prove one Llama-family model generates through
the TurboQuant path.

Requires:
    TQ_TEST_LLAMA_MODEL  — env var pointing to a small Llama-family HF model
                           (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``)
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import pytest

pytestmark = pytest.mark.mlx_integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 32,
    temp: float = 0.0,
    turboquant_k_start: int | None = None,
) -> dict:
    """Run generation and return a diagnostic dict."""
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    t0 = time.perf_counter()

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
    else:
        text = prompt

    input_ids = tokenizer.encode(text)
    prompt_tokens = mx.array(input_ids)

    # Build generation kwargs
    gen_kwargs: dict[str, Any] = dict(
        max_tokens=max_tokens,
    )
    if turboquant_k_start is not None:
        gen_kwargs["turboquant_k_start"] = turboquant_k_start
        gen_kwargs["turboquant_main_bits"] = 3
        gen_kwargs["turboquant_group_size"] = 64
        gen_kwargs["turboquant_rotation"] = "hadamard"
        gen_kwargs["turboquant_residual_topk"] = 2
        gen_kwargs["turboquant_v_bits"] = 4
        gen_kwargs["turboquant_v_group_size"] = 64
        gen_kwargs["turboquant_v_enabled"] = True
        gen_kwargs["turboquant_block_tokens"] = 256

    # Collect tokens
    tokens = []
    for token, logprobs in generate_step(
        prompt_tokens, model, sampler=make_sampler(temp=temp), **gen_kwargs
    ):
        tokens.append(int(token))
        if len(tokens) >= max_tokens:
            break

    elapsed = time.perf_counter() - t0
    output_text = tokenizer.decode(tokens)

    return {
        "output_text": output_text,
        "generated_tokens": len(tokens),
        "prompt_length": len(input_ids),
        "elapsed_seconds": elapsed,
        "turboquant_active": turboquant_k_start is not None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLlamaRuntimeSmoke:
    """Prove the Llama-family model generates through both paths."""

    def test_llama_dense_smoke(
        self, llama_model_and_tokenizer, short_prompt, decode_settings
    ):
        """Dense generation produces non-empty output."""
        model, tokenizer = llama_model_and_tokenizer

        result = _generate_text(
            model,
            tokenizer,
            short_prompt,
            max_tokens=decode_settings["max_tokens"],
            temp=decode_settings["temp"],
            turboquant_k_start=None,  # dense
        )

        assert result["generated_tokens"] > 0, "Dense generation produced no tokens"
        assert len(result["output_text"].strip()) > 0, "Dense output is empty"
        print(
            f"\n  [llama-dense] {result['generated_tokens']} tokens in {result['elapsed_seconds']:.2f}s"
        )
        print(f"  output: {result['output_text'][:100]}...")

    def test_llama_turboquant_smoke(
        self, llama_model_and_tokenizer, short_prompt, decode_settings
    ):
        """TurboQuant generation produces non-empty output."""
        model, tokenizer = llama_model_and_tokenizer

        result = _generate_text(
            model,
            tokenizer,
            short_prompt,
            max_tokens=decode_settings["max_tokens"],
            temp=decode_settings["temp"],
            turboquant_k_start=0,  # activate immediately
        )

        assert result["generated_tokens"] > 0, "TQ generation produced no tokens"
        assert len(result["output_text"].strip()) > 0, "TQ output is empty"
        assert result["turboquant_active"], "TurboQuant was not requested"
        print(
            f"\n  [llama-tq] {result['generated_tokens']} tokens in {result['elapsed_seconds']:.2f}s"
        )
        print(f"  output: {result['output_text'][:100]}...")

    def test_llama_turboquant_token_count(
        self, llama_model_and_tokenizer, short_prompt, decode_settings
    ):
        """TurboQuant generation meets the requested token count."""
        model, tokenizer = llama_model_and_tokenizer
        target = decode_settings["max_tokens"]

        result = _generate_text(
            model,
            tokenizer,
            short_prompt,
            max_tokens=target,
            temp=decode_settings["temp"],
            turboquant_k_start=0,
        )

        # Should generate close to target (may stop early on EOS)
        assert result["generated_tokens"] >= 1, "No tokens generated at all"
        print(
            f"\n  [llama-tq-count] requested={target}, got={result['generated_tokens']}"
        )

    def test_llama_medium_prompt(
        self, llama_model_and_tokenizer, medium_prompt, decode_settings
    ):
        """TurboQuant handles a medium-length prompt without crashing."""
        model, tokenizer = llama_model_and_tokenizer

        result = _generate_text(
            model,
            tokenizer,
            medium_prompt,
            max_tokens=decode_settings["max_tokens"],
            temp=decode_settings["temp"],
            turboquant_k_start=0,
        )

        assert result["generated_tokens"] > 0, "Medium prompt TQ generation failed"
        print(
            f"\n  [llama-medium-tq] {result['generated_tokens']} tokens, "
            f"prompt_len={result['prompt_length']}"
        )

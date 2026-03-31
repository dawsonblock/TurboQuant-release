"""
test_long_context_stability — first meaningful cache-pressure test.

The project exists to compress cache pressure.  This test proves TurboQuant
survives a context length that makes the project meaningful.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor

pytestmark = pytest.mark.mlx_integration

# ---------------------------------------------------------------------------
# Synthetic long-context helpers  (no model download required)
# ---------------------------------------------------------------------------

B, H, D = 1, 4, 128


def _generate_long_kv(
    T: int,
    *,
    seed: int = 42,
) -> tuple[mx.array, mx.array]:
    """Generate deterministic long-context KV tensors."""
    mx.random.seed(seed)
    keys = mx.random.normal((B, H, T, D))
    values = mx.random.normal((B, H, T, D))
    mx.eval(keys, values)
    return keys, values


# ---------------------------------------------------------------------------
# Structural long-context tests (no model required)
# ---------------------------------------------------------------------------


class TestLongContextStructural:
    """Prove KVCompressor survives meaningful cache pressure."""

    def test_long_prefill_dense_no_crash(self):
        """A 512-token dense prefill succeeds without error."""
        cfg = TurboQuantConfig()
        cache = KVCompressor(cfg)
        keys, values = _generate_long_kv(512)

        cache.update_and_fetch(keys, values)
        assert cache.offset == 512

    def test_long_prefill_tq_no_crash(self, default_tq_config):
        """A 512-token TurboQuant prefill succeeds."""
        cache = KVCompressor(default_tq_config)
        keys, values = _generate_long_kv(512)

        view, _ = cache.update_and_fetch(keys, values)
        assert cache.offset == 512
        assert view.end == 512

    def test_incremental_long_context(self, default_tq_config):
        """Multiple incremental updates to 1024 tokens succeed."""
        cache = KVCompressor(default_tq_config)

        for chunk_idx in range(8):
            keys, values = _generate_long_kv(128, seed=chunk_idx)
            cache.update_and_fetch(keys, values)

        assert cache.offset == 1024

    def test_offsets_remain_consistent_after_long_context(self, default_tq_config):
        """Offsets and block iteration are consistent at 1024 tokens."""
        cache = KVCompressor(default_tq_config)

        for chunk_idx in range(8):
            keys, values = _generate_long_kv(128, seed=chunk_idx)
            cache.update_and_fetch(keys, values)

        assert cache.offset == 1024

        # Verify block iteration covers all tokens
        total = 0
        for s, e, k_blk, v_blk in cache.iter_blocks(block_tokens=256):
            assert e > s
            assert e - s == k_blk.shape[2]
            total += e - s

        assert total == 1024

    def test_state_roundtrip_after_long_context(self, default_tq_config):
        """Serialize and restore after 512 tokens."""
        cache = KVCompressor(default_tq_config)
        keys, values = _generate_long_kv(512)
        cache.update_and_fetch(keys, values)

        state = cache.state()
        restored = KVCompressor.from_state(state, default_tq_config)

        assert restored.offset == 512

    def test_decode_after_long_prefill(self, default_tq_config):
        """A single decode step after 512-token prefill succeeds."""
        cache = KVCompressor(default_tq_config)

        # Prefill
        keys, values = _generate_long_kv(512)
        cache.update_and_fetch(keys, values)

        # Decode step (1 token)
        mx.random.seed(100)
        k_step = mx.random.normal((B, H, 1, D))
        v_step = mx.random.normal((B, H, 1, D))
        mx.eval(k_step, v_step)

        view, _ = cache.update_and_fetch(k_step, v_step)
        assert cache.offset == 513
        assert view.end == 513

    def test_memory_reduction_at_long_context(self, default_tq_config):
        """At 1024 tokens, TQ compressed size is smaller than uncompressed estimate."""
        cache = KVCompressor(default_tq_config)

        for chunk_idx in range(8):
            keys, values = _generate_long_kv(128, seed=chunk_idx)
            cache.update_and_fetch(keys, values)

        bd = cache.memory_breakdown()
        compressed_bytes = bd["total"]

        # Dense baseline: 1024 tokens × B × H × D × 4 bytes (float32) × 2 (K+V)
        dense_bytes = 1024 * B * H * D * 4 * 2

        print(f"\n  compressed = {compressed_bytes:,} bytes")
        print(f"  dense est  = {dense_bytes:,} bytes")
        print(f"  ratio      = {compressed_bytes / dense_bytes:.2%}")

        # TQ must use less memory (the entire point of the project)
        assert compressed_bytes < dense_bytes, (
            f"TurboQuant used {compressed_bytes:,} bytes vs "
            f"dense estimate {dense_bytes:,} bytes — no compression benefit"
        )

    def test_no_nan_in_long_context_output(self, default_tq_config):
        """Decoded K values contain no NaN or Inf after long context."""
        cache = KVCompressor(default_tq_config)
        keys, values = _generate_long_kv(512)
        cache.update_and_fetch(keys, values)

        k_full = cache.decode_k_full()
        k_np = np.array(k_full)

        assert np.all(np.isfinite(k_np)), "Decoded K contains NaN/Inf"


# ---------------------------------------------------------------------------
# Model-based long-context smoke (requires env var)
# ---------------------------------------------------------------------------


class TestLongContextModelSmoke:
    """Prove a real model survives long-context TurboQuant generation."""

    def test_long_context_llama_smoke(
        self, llama_model_and_tokenizer, long_prompt, decode_settings
    ):
        """Llama generation with a long prompt and TurboQuant."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = llama_model_and_tokenizer

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": long_prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = long_prompt
        else:
            text = long_prompt

        input_ids = tokenizer.encode(text)
        prompt_tokens = mx.array(input_ids)

        t0 = time.perf_counter()
        tokens = []
        for token, _ in generate_step(
            prompt_tokens,
            model,
            sampler=make_sampler(temp=0.0),
            max_tokens=decode_settings["max_tokens"],
            turboquant_k_start=0,
            turboquant_main_bits=3,
            turboquant_group_size=64,
            turboquant_rotation="hadamard",
            turboquant_residual_topk=2,
            turboquant_v_bits=4,
            turboquant_v_group_size=64,
            turboquant_v_enabled=True,
            turboquant_block_tokens=256,
        ):
            tokens.append(int(token))
            if len(tokens) >= decode_settings["max_tokens"]:
                break

        elapsed = time.perf_counter() - t0
        output = tokenizer.decode(tokens)

        assert len(tokens) > 0, "Long-context TQ generation produced no tokens"
        print(
            f"\n  [long-llama-tq] {len(tokens)} tokens in {elapsed:.2f}s, "
            f"prompt_len={len(input_ids)}"
        )
        print(f"  output: {output[:120]}...")

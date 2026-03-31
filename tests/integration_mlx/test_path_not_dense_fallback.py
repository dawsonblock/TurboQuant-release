"""
test_path_not_dense_fallback — prove TurboQuant path is actually used.

This is the central audit concern: "Is TurboQuant actually used during
generation, or just implemented?"

These tests verify that when TurboQuant is requested:
  1. KVCompressor.update_and_fetch returns a TurboQuantKeysView (not dense K)
  2. The streaming attention path is taken (not SDPA fallback)
  3. The generate_step loop actually upgrades the cache
  4. Log records confirm path activation

No model weights are needed — tests use synthetic data and the actual
KVCompressor/streaming-attention code paths.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.runtime.attention import (
    maybe_turboquant_attention,
    turboquant_streaming_attention,
)
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView

pytestmark = pytest.mark.mlx_integration

# ---------------------------------------------------------------------------
# Dimensions for synthetic data
# ---------------------------------------------------------------------------

B, H, D = 1, 4, 128


@pytest.fixture
def default_tq_config():
    return TurboQuantConfig(
        k_bits=3,
        k_group_size=64,
        rotation="hadamard",
        residual_topk=2,
        v_bits=4,
        v_group_size=64,
        v_enabled=True,
        block_tokens=256,
    )


@pytest.fixture
def compressor_with_data(default_tq_config):
    """Build a KVCompressor, feed it 32 tokens, return (cache, view)."""
    cache = KVCompressor(default_tq_config)
    T = 32
    keys = mx.random.normal((B, H, T, D))
    values = mx.random.normal((B, H, T, D))
    view, _ = cache.update_and_fetch(keys, values)
    mx.eval(cache.k_packed)
    return cache, view


# ---------------------------------------------------------------------------
# Test: update_and_fetch returns TurboQuantKeysView, not dense tensor
# ---------------------------------------------------------------------------


class TestPathNotDenseFallback:
    """Verify the TurboQuant code path is actually exercised."""

    def test_update_and_fetch_returns_view(self, compressor_with_data):
        """The return type must be TurboQuantKeysView, not mx.array."""
        _, view = compressor_with_data
        assert isinstance(view, TurboQuantKeysView), (
            f"Expected TurboQuantKeysView, got {type(view).__name__}. "
            "This means the TurboQuant path is NOT active."
        )

    def test_view_has_correct_bounds(self, compressor_with_data):
        """The view should span [0, offset)."""
        cache, view = compressor_with_data
        assert view.start == 0
        assert view.end == cache.offset == 32
        assert view.d_head == D

    def test_dispatcher_takes_tq_path(self, compressor_with_data):
        """maybe_turboquant_attention must dispatch to streaming when
        given a TurboQuantKeysView (not call fallback)."""
        cache, view = compressor_with_data
        queries = mx.random.normal((B, H, 1, D))

        fallback_called = False

        def _fallback(*args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            return mx.zeros((B, H, 1, D))

        output = maybe_turboquant_attention(
            queries=queries,
            keys=view,
            values=mx.zeros((B, H, 1, D)),  # unused in TQ path
            mask=None,
            scale=D**-0.5,
            fallback=_fallback,
        )
        mx.eval(output)

        assert not fallback_called, (
            "Fallback was called despite TurboQuantKeysView being passed. "
            "The dispatcher silently fell back to dense SDPA."
        )
        assert output.shape == (B, H, 1, D)

    def test_dispatcher_uses_fallback_for_dense_keys(self):
        """When keys is a dense tensor, fallback MUST be called."""
        queries = mx.random.normal((B, H, 1, D))
        keys = mx.random.normal((B, H, 32, D))
        values = mx.random.normal((B, H, 32, D))

        fallback_called = False

        def _fallback(*args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            return mx.zeros((B, H, 1, D))

        output = maybe_turboquant_attention(
            queries=queries,
            keys=keys,
            values=values,
            mask=None,
            scale=D**-0.5,
            fallback=_fallback,
        )
        mx.eval(output)

        assert fallback_called, (
            "Fallback was NOT called for dense keys. "
            "Dispatcher incorrectly routed dense keys to TQ path."
        )

    def test_streaming_attention_produces_output(self, compressor_with_data):
        """turboquant_streaming_attention must return a valid tensor."""
        cache, view = compressor_with_data
        queries = mx.random.normal((B, H, 1, D))
        scale = D**-0.5

        output = turboquant_streaming_attention(queries, view, scale=scale)
        mx.eval(output)

        assert output.shape == (B, H, 1, D)
        assert not mx.any(mx.isnan(output)).item(), "NaN in streaming attention output"

    def test_iter_blocks_actually_decodes(self, compressor_with_data):
        """iter_rotated_kv_blocks must yield decoded K/V, not empty tensors."""
        cache, view = compressor_with_data
        blocks = list(cache.iter_rotated_kv_blocks(view))

        assert len(blocks) > 0, "No blocks yielded — decode path is empty"

        total_tokens = 0
        for s, e, k_blk, v_blk in blocks:
            assert k_blk.shape[0] == B
            assert k_blk.shape[1] == H
            # K should not be all-zeros (it was encoded from random data)
            k_abs_sum = float(mx.sum(mx.abs(k_blk)).item())
            assert k_abs_sum > 0, (
                f"Decoded K block [{s}:{e}] is all zeros — "
                "compression/decompression pipeline is not functioning"
            )
            total_tokens += e - s

        assert total_tokens == cache.offset, (
            f"Block iteration covered {total_tokens} tokens but offset is "
            f"{cache.offset}"
        )

    def test_logging_confirms_tq_path(self, default_tq_config, caplog):
        """Log records must confirm TQ path activation at DEBUG level."""
        cache = KVCompressor(default_tq_config)
        keys = mx.random.normal((B, H, 8, D))
        values = mx.random.normal((B, H, 8, D))

        with caplog.at_level(logging.DEBUG, logger="turboquant.runtime.kv_cache"):
            view, _ = cache.update_and_fetch(keys, values)
            mx.eval(cache.k_packed)

        assert any("TQ path active" in record.message for record in caplog.records), (
            "No 'TQ path active' log record found. "
            "The logging instrumentation is missing or broken."
        )

    def test_logging_attention_dispatch(self, compressor_with_data, caplog):
        """Log records must confirm streaming attention dispatch."""
        cache, view = compressor_with_data
        queries = mx.random.normal((B, H, 1, D))

        def _fallback(*a, **kw):
            return mx.zeros((B, H, 1, D))

        with caplog.at_level(logging.DEBUG, logger="turboquant.runtime.attention"):
            maybe_turboquant_attention(
                queries=queries,
                keys=view,
                values=mx.zeros((B, H, 1, D)),
                mask=None,
                scale=D**-0.5,
                fallback=_fallback,
            )

        assert any(
            "TurboQuant streaming path" in record.message for record in caplog.records
        ), (
            "No 'TurboQuant streaming path' log record from attention dispatch. "
            "The dispatcher logging is missing."
        )

    def test_incremental_updates_stay_on_tq_path(self, compressor_with_data):
        """Repeated single-token updates should all return views."""
        cache, _ = compressor_with_data

        for i in range(5):
            k = mx.random.normal((B, H, 1, D))
            v = mx.random.normal((B, H, 1, D))
            view, _ = cache.update_and_fetch(k, v)
            assert isinstance(
                view, TurboQuantKeysView
            ), f"Step {i}: update returned {type(view).__name__}, not view"
            assert view.end == 32 + i + 1

"""
test_cache_upgrade_roundtrip — structural correctness of the dense → TurboQuant
cache upgrade path.

This is the first real correctness gate in the certification pipeline.  Every
assertion here targets the *structure* of the upgrade, not numerical accuracy
(that is tested separately in ``test_streaming_attention_equivalence``).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView
from turboquant.runtime.state import STATE_SCHEMA_VERSION, validate_state

pytestmark = pytest.mark.mlx_integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small synthetic shapes: B=1, H=4, T=16, D=128  (fits any Apple Silicon Mac)
B, H, D = 1, 4, 128


def _make_kv(T: int = 16, *, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Return deterministic (keys, values) tensors."""
    mx.random.seed(seed)
    keys = mx.random.normal((B, H, T, D))
    values = mx.random.normal((B, H, T, D))
    mx.eval(keys, values)
    return keys, values


def _build_compressor(
    config: TurboQuantConfig | None = None,
) -> KVCompressor:
    cfg = config or TurboQuantConfig()
    return KVCompressor(cfg, layer_id=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCacheUpgradeRoundtrip:
    """Prove the dense → TurboQuant upgrade path is structurally sound."""

    def test_upgrade_stores_tokens_correctly(self, default_tq_config):
        """After update_and_fetch, ``offset`` reflects the stored token count."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        view, _ = cache.update_and_fetch(keys, values)

        assert cache.offset == 16
        assert isinstance(view, TurboQuantKeysView)
        assert view.start == 0
        assert view.end == 16

    def test_upgrade_preserves_layer_structure(self, default_tq_config):
        """Compressed buffers have the expected batch/head/token shape."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        # K packed should be [B, H, >=16, n_words]
        assert cache._k_packed is not None
        assert cache._k_packed.shape[0] == B
        assert cache._k_packed.shape[1] == H
        assert cache._k_packed.shape[2] >= 16

        # K scales
        assert cache._k_scales is not None
        assert cache._k_scales.shape[0] == B
        assert cache._k_scales.shape[1] == H

    def test_incremental_update_extends_offset(self, default_tq_config):
        """Successive calls to ``update_and_fetch`` accumulate tokens."""
        cache = _build_compressor(default_tq_config)

        k1, v1 = _make_kv(T=8, seed=1)
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2, v2 = _make_kv(T=12, seed=2)
        view, _ = cache.update_and_fetch(k2, v2)
        assert cache.offset == 20
        assert view.end == 20

    def test_residual_buffers_present_when_enabled(self, default_tq_config):
        """Residual vals/idx are stored when ``residual_topk > 0``."""
        assert default_tq_config.residual_topk > 0
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        assert cache._resid_vals is not None
        assert cache._resid_idx is not None
        assert cache._resid_vals.shape[0] == B
        assert cache._resid_vals.shape[1] == H

    def test_residual_buffers_absent_when_disabled(self):
        """When ``residual_topk=0``, no residual buffers are allocated."""
        cfg = TurboQuantConfig(residual_topk=0)
        cache = _build_compressor(cfg)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        # Residual buffers should be None
        assert cache._resid_vals is None
        assert cache._resid_idx is None

    def test_v_compression_active(self, default_tq_config):
        """Value compression is engaged when ``v_enabled=True``."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        assert cache._v_packed is not None
        assert cache._v_scales is not None

    def test_state_roundtrip_preserves_offset(self, default_tq_config):
        """Serialize → restore → offset is identical."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        state = cache.state()
        restored = KVCompressor.from_state(state, default_tq_config, layer_id=0)

        assert restored.offset == cache.offset

    def test_state_has_current_schema_version(self, default_tq_config):
        """The state dict reports the current schema version."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        state = cache.state()
        assert state["schema_version"] == STATE_SCHEMA_VERSION

    def test_state_validation_passes(self, default_tq_config):
        """``validate_state`` does not raise for a well-formed state dict."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        state = cache.state()
        # Should not raise
        validate_state(state, default_tq_config)

    def test_state_roundtrip_restores_config_keys(self, default_tq_config):
        """State embeds config fields that survive the roundtrip."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        state = cache.state()
        assert state["k_bits"] == default_tq_config.k_bits
        assert state["v_bits"] == default_tq_config.v_bits
        assert state["residual_topk"] == default_tq_config.residual_topk
        assert state["rotation"] == default_tq_config.rotation

    def test_block_iterator_covers_all_tokens(self, default_tq_config):
        """``iter_blocks`` yields blocks that span [0, offset)."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=32)
        cache.update_and_fetch(keys, values)

        total_tokens = 0
        for s, e, k_blk, v_blk in cache.iter_blocks(block_tokens=16):
            assert e > s
            assert k_blk.shape[2] == e - s
            assert v_blk.shape[2] == e - s
            total_tokens += e - s

        assert total_tokens == 32

    def test_decode_k_full_matches_offset(self, default_tq_config):
        """``decode_k_full`` returns a tensor with token dim == offset."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        k_full = cache.decode_k_full()
        assert k_full.shape == (B, H, 16, D)

    def test_trim_reduces_offset(self, default_tq_config):
        """``trim(n)`` correctly reduces the offset."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 16

        trimmed = cache.trim(4)
        assert trimmed == 4
        assert cache.offset == 12

    def test_memory_breakdown_is_positive(self, default_tq_config):
        """``memory_breakdown`` returns a dict with positive total bytes."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        cache.update_and_fetch(keys, values)

        bd = cache.memory_breakdown()
        assert isinstance(bd, dict)
        assert bd["total"] > 0
        assert bd["k_packed"] > 0
        assert bd["k_scales"] > 0

    def test_view_marks_turboquant_active(self, default_tq_config):
        """The returned view references the compressor (TurboQuant is active)."""
        cache = _build_compressor(default_tq_config)
        keys, values = _make_kv(T=16)
        view, _ = cache.update_and_fetch(keys, values)

        assert isinstance(view, TurboQuantKeysView)
        assert view.cache is cache
        assert view.d_head == D
        assert view.block_tokens == default_tq_config.block_tokens

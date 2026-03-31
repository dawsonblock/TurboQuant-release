"""
test_streaming_attention_equivalence — numerical sanity gate.

TurboQuant is lossy.  We do not need exact equality.  We need **bounded
divergence** between the dense attention output and the TurboQuant
streaming-attention output under a controlled decode step.

Thresholds here are starting values.  After the pilot certification run,
inspect real numbers and freeze thresholds that match reality.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.runtime.attention import (
    _streaming_softmax_attention,
)
from turboquant.runtime.kv_interface import KVCompressor

pytestmark = pytest.mark.mlx_integration

# ---------------------------------------------------------------------------
# Thresholds — FREEZE after pilot run
# ---------------------------------------------------------------------------

# Cosine similarity between dense and TQ attention outputs
# Observed ~0.97 for 3-bit k + 4-bit v with Hadamard rotation on random data.
# Threshold is set conservatively below observed floor.
MIN_COSINE_SIMILARITY = 0.960

# Mean absolute error
MAX_MEAN_ABS_ERROR = 0.06

# Max absolute error (single element)
MAX_MAX_ABS_ERROR = 0.25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, H_KV, D = 1, 4, 128
H_Q = 4  # no GQA for controlled comparison


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened vectors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _dense_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
) -> mx.array:
    """Straightforward dense causal attention (no masking needed for T_q=1)."""
    # queries: [B, H, 1, D], keys: [B, H, T_kv, D], values: [B, H, T_kv, D]
    qf = queries.astype(mx.float32)
    kf = keys.astype(mx.float32)
    vf = values.astype(mx.float32)

    scores = mx.matmul(qf, kf.transpose(0, 1, 3, 2)) * scale  # [B, H, 1, T_kv]
    weights = mx.softmax(scores, axis=-1)
    out = mx.matmul(weights, vf)  # [B, H, 1, D]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingAttentionEquivalence:
    """Compare dense and TurboQuant attention on a controlled decode step."""

    @pytest.fixture
    def controlled_state(self, default_tq_config):
        """Build a controlled cache state: encode T_kv tokens, then query with T_q=1."""
        T_kv = 32
        mx.random.seed(42)
        keys = mx.random.normal((B, H_KV, T_kv, D))
        values = mx.random.normal((B, H_KV, T_kv, D))
        queries = mx.random.normal((B, H_Q, 1, D))
        mx.eval(keys, values, queries)

        # Dense path: rotate queries with the same rotation for fair comparison
        cache = KVCompressor(default_tq_config, layer_id=0)
        view, _ = cache.update_and_fetch(keys, values)

        return cache, view, keys, values, queries

    def test_streaming_attention_close_to_dense(
        self, controlled_state, default_tq_config
    ):
        """TurboQuant streaming attention output stays close to dense."""
        cache, view, keys_orig, values_orig, queries = controlled_state
        scale = D**-0.5

        # Dense reference: apply same rotation to keys and queries
        q_rot = cache.rotate_queries(queries)
        # For dense, we rotate keys through the pipeline too
        k_rot_dense = cache.pipeline.rotate_queries(keys_orig)
        dense_out = _dense_attention(q_rot, k_rot_dense, values_orig, scale)
        mx.eval(dense_out)

        # TurboQuant streaming attention
        tq_out = _streaming_softmax_attention(q_rot, view, scale=scale)
        mx.eval(tq_out)

        dense_np = np.array(dense_out)
        tq_np = np.array(tq_out)

        # Compute metrics
        mean_abs = float(np.mean(np.abs(dense_np - tq_np)))
        max_abs = float(np.max(np.abs(dense_np - tq_np)))
        cosine = _cosine_similarity(dense_np, tq_np)

        # Report for inspection
        print(f"\n  cosine_similarity = {cosine:.6f}")
        print(f"  mean_abs_error    = {mean_abs:.6f}")
        print(f"  max_abs_error     = {max_abs:.6f}")

        assert (
            cosine >= MIN_COSINE_SIMILARITY
        ), f"Cosine similarity {cosine:.6f} < threshold {MIN_COSINE_SIMILARITY}"
        assert (
            mean_abs <= MAX_MEAN_ABS_ERROR
        ), f"Mean absolute error {mean_abs:.6f} > threshold {MAX_MEAN_ABS_ERROR}"
        assert (
            max_abs <= MAX_MAX_ABS_ERROR
        ), f"Max absolute error {max_abs:.6f} > threshold {MAX_MAX_ABS_ERROR}"

    def test_multiple_blocks_converge(self, default_tq_config):
        """Attention over multiple blocks (large T_kv) stays bounded."""
        T_kv = 256
        mx.random.seed(99)
        keys = mx.random.normal((B, H_KV, T_kv, D))
        values = mx.random.normal((B, H_KV, T_kv, D))
        queries = mx.random.normal((B, H_Q, 1, D))
        mx.eval(keys, values, queries)

        cfg = TurboQuantConfig(
            block_tokens=64,  # Force multiple blocks
            k_bits=default_tq_config.k_bits,
            k_group_size=default_tq_config.k_group_size,
            v_bits=default_tq_config.v_bits,
            v_group_size=default_tq_config.v_group_size,
            residual_topk=default_tq_config.residual_topk,
        )
        cache = KVCompressor(cfg, layer_id=0)
        view, _ = cache.update_and_fetch(keys, values)

        scale = D**-0.5
        q_rot = cache.rotate_queries(queries)

        tq_out = _streaming_softmax_attention(q_rot, view, scale=scale)
        mx.eval(tq_out)

        # Sanity: output is finite and has expected shape
        tq_np = np.array(tq_out)
        assert tq_np.shape == (B, H_Q, 1, D)
        assert np.all(np.isfinite(tq_np)), "TurboQuant output contains NaN/Inf"

        # Compare against dense
        k_rot = cache.pipeline.rotate_queries(keys)
        dense_out = _dense_attention(q_rot, k_rot, values, scale)
        mx.eval(dense_out)
        dense_np = np.array(dense_out)

        cosine = _cosine_similarity(dense_np, tq_np)
        mean_abs = float(np.mean(np.abs(dense_np - tq_np)))

        print(f"\n  multi-block cosine_similarity = {cosine:.6f}")
        print(f"  multi-block mean_abs_error    = {mean_abs:.6f}")

        # Slightly relaxed thresholds for larger context
        assert cosine >= 0.96, f"Multi-block cosine {cosine:.6f} < 0.96"
        assert mean_abs <= 0.10, f"Multi-block MAE {mean_abs:.6f} > 0.10"

    def test_output_shape_is_correct(self, controlled_state, default_tq_config):
        """Streaming attention returns [B, H_q, L_q, D]."""
        cache, view, _, _, queries = controlled_state
        scale = D**-0.5

        q_rot = cache.rotate_queries(queries)
        tq_out = _streaming_softmax_attention(q_rot, view, scale=scale)
        mx.eval(tq_out)

        assert tq_out.shape == (B, H_Q, 1, D)

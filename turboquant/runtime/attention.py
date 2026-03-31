"""
turboquant.runtime.attention — model-agnostic streaming causal attention.

This module provides the shared attention helpers used by any model that has
been wired for TurboQuant.  Moving the logic here lets multiple architecture
files (Gemma, Llama, …) share a single implementation instead of copying it.

Public API
----------
    maybe_turboquant_attention(queries, keys, values, mask, scale, fallback)
        High-level dispatcher: uses streaming attention when *keys* is a
        :class:`TurboQuantKeysView`, otherwise delegates to *fallback*.

    turboquant_streaming_attention(queries, keys_view, *, scale)
        Direct entry point for models that have already confirmed that
        *keys* is a view.

    _expand_kv_heads(x, target_heads)
        Broadcast KV heads to match query heads for GQA.

    _streaming_softmax_attention(q_rot, keys_view, *, scale)
        Numerically stable online-softmax over compressed K/V blocks.

Internal helpers are prefixed with ``_`` and should not be imported
directly by model files.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import mlx.core as mx

from turboquant.runtime.kv_interface import TurboQuantKeysView

logger = logging.getLogger("turboquant.runtime.attention")

# ── GQA broadcast ─────────────────────────────────────────────────────────────


def attention_kernel(q, k, v, scale=1.0):
    """
    Placeholder attention kernel boundary.
    Can be replaced with fused/flash/tiled attention.
    """
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    return mx.matmul(qf, kf.transpose(0, 1, 3, 2)) * scale


def _expand_kv_heads(x: mx.array, target_heads: int) -> mx.array:
    """Broadcast KV heads to match query heads for grouped-query attention.

    Parameters
    ----------
    x:
        [B, H_kv, T, D]
    target_heads:
        Number of query heads (must be a multiple of H_kv).

    Returns
    -------
    [B, target_heads, T, D]
    """
    h = x.shape[1]
    if h == target_heads:
        return x
    if target_heads % h != 0:
        raise ValueError(
            f"Cannot expand {h} KV heads to {target_heads} query heads: "
            f"{target_heads} is not divisible by {h}."
        )
    repeats = target_heads // h
    return mx.repeat(x, repeats, axis=1)


# ── Core online-softmax loop ──────────────────────────────────────────────────


def _streaming_softmax_attention(
    q_rot: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    """Numerically stable online-softmax causal attention over TurboQuant blocks.

    Implements the two-accumulator (m, lse, acc) update from "Flash Attention"
    without requiring the full K matrix to be materialised.

    Parameters
    ----------
    q_rot:
        [B, H_q, L_q, D] — queries already rotated into K's coordinate frame.
        Rotation is the caller's responsibility (call
        ``cache.rotate_queries(q)`` or ``cache.rotate_queries_for_attention(q)``
        before passing here).
    keys_view:
        A :class:`TurboQuantKeysView` handle over the compressed K/V history.
    scale:
        Attention scale factor (typically ``head_dim ** -0.5``).

    Returns
    -------
    [B, H_q, L_q, Dv]  in ``float32``.  Cast to model dtype in the caller.
    """
    cache = keys_view.cache
    B, H_q, L_q, _ = q_rot.shape

    q_end = keys_view.end
    q_start = q_end - L_q
    # q_pos: [1, 1, L_q, 1] — broadcast-friendly causal position tensor
    q_pos = mx.arange(q_start, q_end, dtype=mx.int32).reshape(1, 1, L_q, 1)

    # Online-softmax state
    m = mx.full((B, H_q, L_q, 1), -1e30, dtype=mx.float32)
    lse = mx.zeros((B, H_q, L_q, 1), dtype=mx.float32)
    acc: mx.array | None = None

    for s, e, k_rot_blk, v_blk in cache.iter_rotated_kv_blocks(keys_view):
        k_rot_blk = _expand_kv_heads(k_rot_blk, H_q)
        v_blk = _expand_kv_heads(v_blk, H_q)

        q_rot.astype(mx.float32)
        k_rot_blk.astype(mx.float32)
        vf = v_blk.astype(mx.float32)

        # [B, H_q, L_q, blk_len]
        scores = attention_kernel(q_rot, k_rot_blk, v_blk, scale=scale)

        # Causal mask: skip if block is strictly in the past
        if e > q_start:
            k_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, 1, 1, e - s)
            scores = mx.where(
                k_pos <= q_pos,
                scores,
                mx.array(-1e30, dtype=scores.dtype),
            )

        blk_m = mx.max(scores, axis=-1, keepdims=True)
        new_m = mx.maximum(m, blk_m)
        alpha = mx.exp(m - new_m)  # rescale factor for old accumulator
        p = mx.exp(scores - new_m)  # softmax numerator for this block

        if acc is None:
            Dv = vf.shape[-1]
            acc = mx.zeros((B, H_q, L_q, Dv), dtype=mx.float32)

        lse = lse * alpha + mx.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + mx.matmul(p, vf)
        m = new_m

    if acc is None:
        # Empty view (e.g. first token of a fresh session) — return zeros.
        Dv = q_rot.shape[-1]
        acc = mx.zeros((B, H_q, L_q, Dv), dtype=mx.float32)
        lse = mx.ones((B, H_q, L_q, 1), dtype=mx.float32)

    out = acc / mx.maximum(lse, mx.array(1e-6, dtype=lse.dtype))

    # NaN guard — detect collapsed or corrupted attention
    if mx.any(mx.isnan(out)).item():  # type: ignore[union-attr]
        logger.warning(
            "NaN detected in streaming attention output "
            "(view %d–%d, H_q=%d). Clamping to zero.",
            keys_view.start,
            keys_view.end,
            H_q,
        )
        out = mx.where(mx.isnan(out), mx.zeros_like(out), out)

    return out


# ── Public streaming entry point ──────────────────────────────────────────────


def turboquant_streaming_attention(
    queries: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    """Rotate queries and run streaming causal attention.

    This is the entry point called by model attention layers once they have
    confirmed that *keys_view* is a :class:`TurboQuantKeysView`.

    Parameters
    ----------
    queries:
        [B, H_q, L_q, D] — **un-rotated** query vectors.
    keys_view:
        View into the compressed K/V history.
    scale:
        Attention scale factor (typically ``head_dim ** -0.5``).

    Returns
    -------
    [B, H_q, L_q, Dv]  cast to the same dtype as *queries*.
    """
    cache = keys_view.cache
    # ``rotate_queries_for_attention`` is the canonical name; KVCompressor
    # also exposes ``rotate_queries`` as an alias.
    q_rot = cache.rotate_queries_for_attention(queries)
    return _streaming_softmax_attention(q_rot, keys_view, scale=scale).astype(
        queries.dtype
    )


# ── High-level dispatcher ─────────────────────────────────────────────────────


def maybe_turboquant_attention(
    queries: mx.array,
    keys: Any,
    values: mx.array,
    mask: mx.array | None,
    scale: float,
    fallback: Callable,
    cache: Any = None,
) -> mx.array:
    """Dispatch to TurboQuant streaming attention or *fallback*.

    Use this helper inside model ``Attention.__call__`` to handle both
    compressed and standard cache paths with a single call site.

    Parameters
    ----------
    queries:
        [B, H_q, L_q, D]
    keys:
        Either a :class:`TurboQuantKeysView` (compressed path) or a dense
        [B, H_kv, T, D] tensor (standard path).
    values:
        [B, H_kv, T, Dv] — used only on the standard path.
    mask:
        Attention mask or ``None`` — used only on the standard path.
    scale:
        Attention scale factor.
    fallback:
        Callable with signature ``fallback(queries, keys, values, cache,
        scale, mask)`` used when *keys* is not a
        :class:`TurboQuantKeysView`.
    cache:
        Cache object forwarded to *fallback*.

    Returns
    -------
    [B, H_q, L_q, Dv]
    """
    if isinstance(keys, TurboQuantKeysView):
        logger.debug(
            "attention dispatch: TurboQuant streaming path  " "(view %d–%d, H_q=%d)",
            keys.start,
            keys.end,
            queries.shape[1],
        )
        return turboquant_streaming_attention(queries, keys, scale=scale)
    logger.debug(
        "attention dispatch: dense SDPA fallback  (H_q=%d, T_k=%d)",
        queries.shape[1],
        keys.shape[2],
    )
    return fallback(queries, keys, values, cache=cache, scale=scale, mask=mask)

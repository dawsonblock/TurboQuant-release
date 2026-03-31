import os
from typing import Optional

import mlx.core as mx

from turboquant.core.quantizer import dequantize_groups
from turboquant.core.residual import decode_topk_residual
from turboquant.experimental.kernels.metal.runtime import decode_k_metal


def decode_k_block(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    """
    Single entry point for all MLX-vectorized decode paths.
    Must return rotated K.
    """
    if os.getenv("TQ_USE_METAL", "0") == "1":
        return decode_k_metal(packed_k, scales, resid_vals, resid_idx, config, d_head)

    return decode_k_fallback(packed_k, scales, resid_vals, resid_idx, config, d_head)


_COMP_FALLBACK_CACHE = {}


def _inner_decode_fallback(
    packed_k,
    scales,
    resid_vals,
    resid_idx,
    k_bits,
    k_group_size,
    d_pad,
    d_head,
    residual_topk,
):
    y_hat = dequantize_groups(packed_k, scales, k_bits, k_group_size, d_pad)

    if residual_topk > 0 and resid_vals is not None and resid_idx is not None:
        residual = decode_topk_residual(resid_vals, resid_idx, k_group_size)
        y_hat = y_hat + residual[..., :d_pad]

    return y_hat[..., :d_head]


def decode_k_fallback(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    d_pad = (
        (d_head + config.k_group_size - 1) // config.k_group_size * config.k_group_size
    )

    key = (
        config.k_bits,
        config.k_group_size,
        d_pad,
        d_head,
        config.residual_topk,
    )

    if key not in _COMP_FALLBACK_CACHE:

        def fn(pk, s, rv, ri):
            return _inner_decode_fallback(pk, s, rv, ri, *key)

        _COMP_FALLBACK_CACHE[key] = mx.compile(fn, shapeless=False)

    rv_arg = resid_vals if resid_vals is not None else mx.array(0)
    ri_arg = resid_idx if resid_idx is not None else mx.array(0)

    return _COMP_FALLBACK_CACHE[key](packed_k, scales, rv_arg, ri_arg)

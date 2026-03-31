import os
from pathlib import Path

import mlx.core as mx

_kernels = {}


def get_kernel_source():
    kernel_path = Path(__file__).parent / "decode_k.metal"
    with open(kernel_path) as f:
        return f.read()


def decode_k_metal(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: mx.array,
    resid_idx: mx.array,
    config,
    d_head: int,
):
    global _kernels

    cache_key = (config.k_bits, config.k_group_size, config.residual_topk)

    if cache_key not in _kernels:
        _kernels[cache_key] = mx.fast.metal_kernel(
            name="decode_k",
            input_names=["packed", "scales", "resid_idx", "resid_vals"],
            output_names=["out"],
            source=get_kernel_source(),
        )

    kernel = _kernels[cache_key]
    threadgroup_size = int(os.getenv("TQ_THREADGROUP_SIZE", "64"))

    if resid_vals is None:
        resid_vals = mx.zeros((1,), dtype=mx.float16)
        resid_idx = mx.zeros((1,), dtype=mx.uint16)

    total_elements = scales.size * config.k_group_size
    grid = (total_elements, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    out_shape = packed_k.shape[:-1] + (d_head,)
    n_groups = scales.shape[-1]
    n_words = packed_k.shape[-1]

    out = kernel(
        inputs=[packed_k, scales, resid_idx, resid_vals],
        output_shapes=[out_shape],
        output_dtypes=[mx.float16],
        grid=grid,
        threadgroup=threadgroup,
        template=[
            ("BITS", config.k_bits),
            ("GROUP_SIZE", config.k_group_size),
            ("TOPK", config.residual_topk),
            ("N_GROUPS", n_groups),
            ("N_WORDS", n_words),
            ("D_HEAD", d_head),
        ],
        stream=mx.gpu,
    )

    return out[0]

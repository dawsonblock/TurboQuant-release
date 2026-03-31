import mlx.core as mx
from turboquant.config import TurboQuantConfig

from tests.helpers.mlx_env import MLX_SKIP_MARKER
from turboquant.runtime.kv_interface import KVCompressor

pytestmark = MLX_SKIP_MARKER


def test_memory_ratio_stable():
    # Example logic to prove memory ratio doesn't exceed baseline
    cfg = TurboQuantConfig(k_bits=3, k_group_size=64, residual_topk=2)
    comp = KVCompressor(cfg)

    # Simulate a stream of 1024 tokens
    B, H, L, D = 1, 1, 1024, 64
    k = mx.random.normal((B, H, L, D))
    v = mx.random.normal((B, H, L, D))

    keys_view, v_out = comp.update_and_fetch(k, v)
    mx.eval(comp.k_packed, comp.k_scales)  # Ensure allocation

    # Roughly approx expected ratio check: 3 bits per val + 16 bits scale vs 16 bits dense
    dense_bytes = k.nbytes + v.nbytes
    assert comp.nbytes < dense_bytes * 0.5, "Memory ratio exceeds acceptable threshold"


def test_no_full_decode_allocation():
    cfg = TurboQuantConfig(k_bits=3, k_group_size=64, residual_topk=2)
    comp = KVCompressor(cfg)

    B, H, D = 1, 1, 64
    k = mx.random.normal((B, H, 128, D))
    v = mx.random.normal((B, H, 128, D))

    keys_view, v_out = comp.update_and_fetch(k, v)
    mx.eval(comp.k_packed, comp.k_scales)

    # Ensure properties that check if it dynamically created a generic full flat K tensor are absent
    # Since streaming blocks chunks directly, size should equal the block size limit iterations.
    for s, e, k_blk, v_blk in comp.iter_rotated_kv_blocks(keys_view):
        assert (
            k_blk.shape[-2] <= cfg.block_tokens
        ), "Allocated full dense chunk rather than block limit."

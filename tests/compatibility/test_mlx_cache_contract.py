import mlx.core as mx

from tests.helpers.mlx_env import MLX_SKIP_MARKER
from turboquant import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView

pytestmark = MLX_SKIP_MARKER


def test_cache_contract_initialization():
    config = TurboQuantConfig()
    cache = KVCompressor(config)
    assert hasattr(cache, "update_and_fetch")
    assert hasattr(cache, "offset")


def test_cache_contract_shape():
    config = TurboQuantConfig()
    cache = KVCompressor(config)
    k = mx.random.normal((1, 8, 1, 128))
    v = mx.random.normal((1, 8, 1, 128))
    k_out, v_out = cache.update_and_fetch(k, v)
    assert isinstance(k_out, TurboQuantKeysView) and v_out.shape == (1, 8, 1, 128)
    assert v_out.shape == (1, 8, 1, 128)
    assert cache.offset == 1

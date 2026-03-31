import time

import mlx.core as mx

from turboquant import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor


def run_bench():
    config = TurboQuantConfig()
    cache = KVCompressor(config)

    # Warmup
    k = mx.random.normal((1, 32, 1, 128))
    v = mx.random.normal((1, 32, 1, 128))
    cache.update_and_fetch(k, v)

    start = time.perf_counter()
    for _ in range(100):
        k_out, v_out = cache.update_and_fetch(k, v)
    end = time.perf_counter()

    print(f"Decode Step Benchmark: {(end - start) / 100 * 1000:.2f} ms / step")


if __name__ == "__main__":
    run_bench()

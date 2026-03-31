import time

import mlx.core as mx

from turboquant import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline


def run_bench():
    config = TurboQuantConfig()
    pipeline = TurboQuantPipeline(config)

    # Warmup
    x = mx.random.normal((1, 32, 128, 128))
    out = pipeline.encode_k(x)
    mx.eval(*(out if isinstance(out, tuple) else [out]))

    start = time.perf_counter()
    for _ in range(100):
        out = pipeline.encode_k(x)
        mx.eval(*(out if isinstance(out, tuple) else [out]))
    end = time.perf_counter()

    print(f"K-Encode Benchmark: {(end - start) / 100 * 1000:.2f} ms / step")


if __name__ == "__main__":
    run_bench()

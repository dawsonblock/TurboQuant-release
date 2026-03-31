"""
KVCompressor decode-step microbenchmark.

Measures per-token encode latency after a prefill, using the test
fixtures from tests.integration.test_turboquant_gemma.py so the cache config is identical
to the unit tests.integration.
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests.integration"))

import mlx.core as mx

from tests.integration.test_turboquant_gemma import PREFILL_LEN, _make_kv
from turboquant import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor


def my_make_tq_cache(rm):
    return KVCompressor(TurboQuantConfig(k_bits=3, k_group_size=8, block_tokens=2))


REPS = 100


def bench(label: str, return_mode: str = "dequant") -> None:
    # Warmup
    for _ in range(3):
        tq = my_make_tq_cache(return_mode)
        kw, vw = _make_kv(PREFILL_LEN)
        tq.update_and_fetch(kw, vw)
        mx.eval(tq.k_packed)

    # Fresh cache + prefill for timing
    tq = my_make_tq_cache(return_mode)
    kp, vp = _make_kv(PREFILL_LEN)
    tq.update_and_fetch(kp, vp)
    mx.eval(tq.k_packed)

    k1, v1 = _make_kv(1)

    t0 = time.perf_counter()
    for _ in range(REPS):
        tq.update_and_fetch(k1, v1)
        mx.eval(tq.k_packed)
    t1 = time.perf_counter()

    ms = (t1 - t0) / REPS * 1000
    print(f"  {label:30s}  {ms:.3f} ms/step  ({PREFILL_LEN}+N tokens, {REPS} reps)")


print("=== KVCompressor decode-step latency ===")
bench("dequant mode (encode only)", "dequant")
bench("view mode   (encode only)", "view")
print("done")

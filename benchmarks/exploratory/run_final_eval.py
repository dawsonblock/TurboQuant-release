import platform
import sys
import time

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.runtime.attention import turboquant_streaming_attention
from turboquant.runtime.kv_interface import KVCompressor


def measure_dense_attention(B, H_q, H_kv, L, D):
    # Setup dense cache
    k = mx.random.normal((B, H_kv, L, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, L, D)).astype(mx.float16)
    q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)  # Step attention query
    mx.eval(k, v, q)

    start = time.perf_counter()
    for _ in range(50):
        # basic attention
        kf = k.astype(mx.float32)
        qf = q.astype(mx.float32)

        # repeat K to match Q heads (simplified GQA)
        repeats = H_q // H_kv
        if repeats > 1:
            kf = mx.repeat(kf, repeats, axis=1)
            vf = mx.repeat(v.astype(mx.float32), repeats, axis=1)
        else:
            vf = v.astype(mx.float32)

        scores = mx.matmul(qf, kf.transpose(0, 1, 3, 2)) * (D**-0.5)
        scores = mx.softmax(scores, axis=-1)  # mock softmax overhead
        v_out = mx.matmul(scores, vf)
        mx.eval(v_out)
    mx.metal.device_info()  # sync
    end = time.perf_counter()

    dense_mem = k.nbytes + v.nbytes
    dense_time = (end - start) / 50
    return dense_mem, dense_time


def measure_tq_attention(B, H_q, H_kv, L, D):
    # Mode=fast skips residual topk application so it matches realistic fast-path
    cfg = TurboQuantConfig(
        k_bits=3,
        k_group_size=64,
        residual_topk=0,
        v_bits=4,
        v_group_size=64,
        mode="fast",
        rotation="identity",
    )
    comp = KVCompressor(cfg)

    k = mx.random.normal((B, H_kv, L, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, L, D)).astype(mx.float16)
    q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

    # Pre-compress and push to KVCompressor
    keys_view, vals = comp.update_and_fetch(k, v)
    mx.eval(comp._k_packed, comp._k_scales)

    start = time.perf_counter()
    for _ in range(50):
        out = turboquant_streaming_attention(q, keys_view, scale=(D**-0.5))
        mx.eval(out)
    mx.metal.device_info()  # sync
    end = time.perf_counter()

    tq_mem = sum(
        getattr(comp, k).nbytes
        for k in [
            "_k_packed",
            "_k_scales",
            "_v_packed",
            "_v_scales",
            "_resid_vals",
            "_resid_idx",
        ]
        if getattr(comp, k) is not None
    )
    tq_time = (end - start) / 50
    return tq_mem, tq_time


if __name__ == "__main__":
    print("=== TurboQuant Final Evaluation vs Dense KV Cache ===")

    # 8B model scale: Llama 3 8B setup approx
    B, H_q, H_kv, L, D = 1, 32, 8, 4096, 128

    print(
        f"Shapes: [Batch={B}, Q_Heads={H_q}, KV_Heads={H_kv}, Context={L}, HeadDim={D}]"
    )
    print(
        f"Environment: {platform.platform()} | Python {sys.version.split()[0]} | MLX {mx.__version__}"
    )

    # Warmup
    print("Warming up MLX Metal backend...")
    measure_dense_attention(1, 1, 1, 128, 64)
    measure_tq_attention(1, 1, 1, 128, 64)

    print("\nMeasuring Dense baseline...")
    dense_mem, dense_time = measure_dense_attention(B, H_q, H_kv, L, D)
    print("Measuring TurboQuant...")
    tq_mem, tq_time = measure_tq_attention(B, H_q, H_kv, L, D)

    mem_ratio = tq_mem / dense_mem
    time_ratio = tq_time / dense_time

    print("\n--- Metrics ---")
    print(
        f"Memory:  Dense={dense_mem / 1e6:.2f}MB, TQ={tq_mem / 1e6:.2f}MB (Savings: {(1 - mem_ratio) * 100:.1f}%)"
    )
    print(
        f"Latency: Dense={dense_time * 1000:.2f}ms, TQ={tq_time * 1000:.2f}ms (Slowdown: {time_ratio:.2f}x)"
    )

    print("\n--- Verdict Thresholds ---")
    print(
        "Goal: Memory Savings > 60%  | Allowable Slowdown (pre-Metal dispatch) < 3.0x"
    )
    mem_pass = mem_ratio < 0.40
    time_pass = time_ratio < 3.00

    print(f"Memory Pass:  {'✅' if mem_pass else '❌'}")
    print(f"Latency Pass: {'✅' if time_pass else '❌'}")

    if mem_pass and time_pass:
        print(
            "\n🏆 CONCLUSION: ARCHITECTURE IS SOUND. READY FOR FINAL METAL KERNEL LINKING."
        )
    else:
        print("\n⚠️ CONCLUSION: FURTHER KERNEL OPTIMIZATION REQUIRED.")

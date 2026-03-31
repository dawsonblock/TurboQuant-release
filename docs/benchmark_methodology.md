# Benchmark Methodology

- Use `mx.eval()` to synchronize compute.
- Utilize `time.perf_counter()` for precision.
- Run steps: bench_k_encode.py, bench_decode_step.py, bench_memory.py.

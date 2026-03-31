from turboquant import TurboQuantConfig


def run_bench():
    TurboQuantConfig()

    print("Memory Benchmark (estimated):")
    # Native
    print("Native KV Cache: float16, 1 token -> 128 * 32 * 2 bytes")
    # TQ
    print("TurboQuant Cache: int8/int4 compressed bytes footprint")


if __name__ == "__main__":
    run_bench()

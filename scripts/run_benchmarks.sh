#!/usr/bin/env bash
python benchmarks/bench_k_encode.py > artifacts/benchmarks/k_encode.txt
python benchmarks/bench_decode_step.py > artifacts/benchmarks/decode.txt
python benchmarks/bench_memory.py > artifacts/benchmarks/memory.txt

# Capture environment
python -c "import platform, sys, json; print(json.dumps({'python': sys.version, 'platform': platform.platform()}, indent=2))" > artifacts/benchmarks/env.json

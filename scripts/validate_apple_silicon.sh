#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools nox
python -m pip install -e '.[dev,apple]'

echo "Running strict preflight..."
python3 scripts/preflight.py --strict

echo "Running MLX tests via Nox..."
nox -s tests_mlx

echo "Running structural integration gate..."
python3 -m pytest tests/integration_mlx/ -v --tb=short -k "llama or gemma"

echo "Running path-proof gate..."
python3 -m pytest tests/integration_mlx/test_path_not_dense_fallback.py -v --tb=short

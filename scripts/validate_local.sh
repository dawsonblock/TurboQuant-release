#!/usr/bin/env bash
set -e

echo "Running TurboQuant Local Validation..."

# 1. Preflight
python3 scripts/preflight.py

# 2. Run static tests
echo "\nRunning static tests..."
python3 -m pytest tests/unit_static/

# 3. Optional: Run strictly hardware dependent tests if on platform
if python3 scripts/preflight.py --strict 2>/dev/null; then
    echo "\nRunning MLX-dependent tests..."
    python3 -m pytest tests/unit/
    python3 -m pytest tests/integration/
else
    echo "\nSkipping MLX tests (not on Apple Silicon / missing MLX)."
fi

echo "\nAll local validation checks passed."

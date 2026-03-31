"""
tests/helpers/mlx_env.py — shared Apple Silicon / MLX availability check.

Import and use the module-level ``pytestmark`` in any test file that requires
MLX to be present on Apple Silicon:

    from tests.helpers.mlx_env import MLX_SKIP_MARKER
    pytestmark = MLX_SKIP_MARKER

Or use the boolean helpers for finer-grained control:

    from tests.helpers.mlx_env import HAS_MLX, IS_APPLE_SILICON
"""
from __future__ import annotations

import platform

import pytest

IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

try:
    import mlx.core  # noqa: F401

    HAS_MLX: bool = True
except ImportError:
    HAS_MLX = False


def _skip_reason() -> str | None:
    if not IS_APPLE_SILICON:
        return "requires Apple Silicon (darwin-arm64)"
    if not HAS_MLX:
        return "requires the `mlx` package"
    return None


_REASON = _skip_reason()

# Drop this into any test module that needs MLX on Apple Silicon:
#   pytestmark = MLX_SKIP_MARKER
MLX_SKIP_MARKER = pytest.mark.skipif(
    _REASON is not None,
    reason=_REASON or "Apple Silicon + MLX required",
)

"""
turboquant.integrations.mlx.adapter — single MLX boundary layer.

All interaction between TurboQuant production code and the MLX framework
is channelled through this module.  Any other file that needs an MLX
primitive should import it from here rather than importing ``mlx.core``
directly, so that the boundary is auditable in one place.

Why one boundary file?
-----------------------
* **Testability** — unit tests that mock this module can exercise the full
  TurboQuant stack without a physical Apple Silicon GPU.
* **Auditability** — ``tools/audit_vendored_surface.py`` checks that only
  ``integrations/mlx/adapter.py`` imports directly from ``mlx.*``.
* **Compatibility** — version checks and dtype helpers live here; callers
  never inspect ``mx.__version__`` directly.

What lives here
---------------
``mlx_version()``          — ``mx.__version__`` as a string
``is_mlx_available()``     — safe existence check (no import error on non-Apple)
``eval_and_sync(*arrays)`` — ``mx.eval`` + ``mx.synchronize``; prefer over bare ``mx.eval``
``zeros(shape, dtype)``    — ``mx.zeros``
``ones(shape, dtype)``     — ``mx.ones``
``to_float32(arr)``        — cast to float32
``to_float16(arr)``        — cast to float16
``to_bfloat16(arr)``       — cast to bfloat16
``softmax(arr, axis)``     — ``nn.softmax`` via mlx.nn
``concat(arrays, axis)``   — ``mx.concatenate``
``item(arr)``              — scalar extraction (wraps ``arr.item()``)
``SUPPORTED_MLX_MIN``      — minimum supported MLX version string
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger("turboquant.integrations.mlx.adapter")

# Minimum MLX version that TurboQuant officially supports.
# See docs/support_matrix.md.
SUPPORTED_MLX_MIN: str = "0.16.0"


def is_mlx_available() -> bool:
    """Return ``True`` if ``mlx`` can be imported on this machine."""
    return importlib.util.find_spec("mlx") is not None


def mlx_version() -> str:
    """Return the installed MLX version string (e.g. ``'0.16.1'``).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If the installed version is below ``SUPPORTED_MLX_MIN``.
    """
    import mlx.core as mx  # noqa: PLC0415 (lazy import is intentional)

    ver = str(mx.__version__)
    _check_version(ver)
    return ver


def _check_version(ver: str) -> None:
    """Raise ``RuntimeError`` if *ver* is below ``SUPPORTED_MLX_MIN``."""
    try:
        from packaging.version import Version  # type: ignore[import-untyped]

        if Version(ver) < Version(SUPPORTED_MLX_MIN):
            raise RuntimeError(
                f"TurboQuant requires MLX >= {SUPPORTED_MLX_MIN}, "
                f"found {ver}.  Upgrade with: pip install --upgrade mlx"
            )
    except ImportError:
        # packaging not installed — skip version check, emit warning
        logger.warning(
            "MLX version check skipped: 'packaging' not installed.  "
            "Install it with: pip install packaging"
        )


# ── Array construction ────────────────────────────────────────────────────────


def zeros(shape: tuple[int, ...], dtype=None):
    """Create a zero-filled array of *shape* and optional *dtype*."""
    import mlx.core as mx

    kw = {"dtype": dtype} if dtype is not None else {}
    return mx.zeros(shape, **kw)


def ones(shape: tuple[int, ...], dtype=None):
    """Create a one-filled array of *shape* and optional *dtype*."""
    import mlx.core as mx

    kw = {"dtype": dtype} if dtype is not None else {}
    return mx.ones(shape, **kw)


# ── Type casting ──────────────────────────────────────────────────────────────


def to_float32(arr):
    """Cast *arr* to ``mx.float32``."""
    import mlx.core as mx

    return arr.astype(mx.float32)


def to_float16(arr):
    """Cast *arr* to ``mx.float16``."""
    import mlx.core as mx

    return arr.astype(mx.float16)


def to_bfloat16(arr):
    """Cast *arr* to ``mx.bfloat16``."""
    import mlx.core as mx

    return arr.astype(mx.bfloat16)


# ── Evaluation and synchronisation ───────────────────────────────────────────


def eval_and_sync(*arrays) -> None:
    """Evaluate all pending graphs for *arrays* and synchronise the stream.

    Prefer this over bare ``mx.eval`` in TurboQuant production code so that
    timing measurements and memory footprint readings are accurate.
    """
    import mlx.core as mx

    mx.eval(*arrays)
    mx.synchronize()


# ── Math helpers ──────────────────────────────────────────────────────────────


def softmax(arr, axis: int = -1):
    """Apply softmax along *axis* using ``mlx.core.softmax``."""
    import mlx.core as mx

    return mx.softmax(arr, axis=axis)


def concat(arrays: list, axis: int = 0):
    """Concatenate a list of arrays along *axis*."""
    import mlx.core as mx

    return mx.concatenate(arrays, axis=axis)


def item(arr) -> float | int:
    """Extract a scalar Python value from a single-element MLX array."""
    return arr.item()


# ── Dtype helpers ─────────────────────────────────────────────────────────────


def float32():
    """Return ``mx.float32``."""
    import mlx.core as mx

    return mx.float32


def float16():
    """Return ``mx.float16``."""
    import mlx.core as mx

    return mx.float16


def bfloat16():
    """Return ``mx.bfloat16``."""
    import mlx.core as mx

    return mx.bfloat16


def uint8():
    """Return ``mx.uint8``."""
    import mlx.core as mx

    return mx.uint8


def int32():
    """Return ``mx.int32``."""
    import mlx.core as mx

    return mx.int32

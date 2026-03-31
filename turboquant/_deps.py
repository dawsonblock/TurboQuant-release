"""
Runtime dependency helpers for TurboQuant.

These utilities let the package import cleanly on any platform while
giving clear errors when MLX-dependent features are actually invoked.
"""

from __future__ import annotations

import importlib
import platform

# Minimum and maximum supported MLX versions (exclusive upper bound).
MIN_MLX_VERSION = "0.30.0"
MAX_MLX_VERSION_EXCLUSIVE = "1.0.0"


def _parse_version(ver_str: str) -> tuple[int, ...]:
    """Parse a version string like '0.30.1' into a tuple of ints."""
    parts: list[int] = []
    for p in ver_str.split(".")[:3]:
        # Strip pre-release suffixes (e.g. '0.30.0rc1' → '0')
        digits = ""
        for ch in p:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def has_mlx() -> bool:
    """Return True if ``mlx`` is importable."""
    try:
        importlib.import_module("mlx")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def is_apple_silicon() -> bool:
    """Return True when running on macOS arm64 (Apple Silicon)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def require_mlx(feature: str = "this feature") -> None:
    """Raise ``ImportError`` with a clear message if MLX is missing.

    Parameters
    ----------
    feature
        Human-readable label inserted into the error message, e.g.
        ``"KVCompressor"`` or ``"calibrate()"``.
    """
    if not is_apple_silicon():
        raise RuntimeError(
            "TurboQuant compiler paths are formally supported exclusively "
            "on Apple Silicon running MacOS."
        )
    if not has_mlx():
        raise ImportError(
            f"TurboQuant: {feature} requires the `mlx` package, which is only "
            "available on Apple Silicon macOS.  Install it with:\n\n"
            "  pip install 'turboquant[apple]'\n\n"
            "For packaging and static tests on non-Apple platforms, "
            "use the `turboquant.config` module only."
        )


def check_mlx_version() -> None:
    """Validate that the installed MLX version is within the supported range.

    Raises :class:`turboquant.errors.TurboQuantCompatibilityError` when the
    version is outside ``[0.30.0, 1.0.0)``.  Silently returns if MLX is
    not installed (the import-time check handles that separately).
    """
    if not has_mlx():
        return

    ver_str = None
    try:
        # mlx.core.__version__ is the canonical source
        import mlx.core

        ver_str = getattr(mlx.core, "__version__", None)
    except Exception:
        pass

    if ver_str is None:
        try:
            import importlib.metadata

            ver_str = importlib.metadata.version("mlx")
        except Exception:
            return  # cannot determine version — allow

    try:
        ver = _parse_version(ver_str)
    except Exception:
        return  # if parsing fails, don't block startup

    from turboquant.errors import TurboQuantCompatibilityError

    if ver < _parse_version(MIN_MLX_VERSION):
        raise TurboQuantCompatibilityError(
            f"TurboQuant requires MLX >= {MIN_MLX_VERSION}, "
            f"but found {ver_str}. Please upgrade: pip install -U mlx"
        )
    if ver >= _parse_version(MAX_MLX_VERSION_EXCLUSIVE):
        raise TurboQuantCompatibilityError(
            f"TurboQuant has not been tested with MLX >= "
            f"{MAX_MLX_VERSION_EXCLUSIVE} (found {ver_str}). "
            f"Pin to mlx<{MAX_MLX_VERSION_EXCLUSIVE} or check for "
            f"a TurboQuant update."
        )

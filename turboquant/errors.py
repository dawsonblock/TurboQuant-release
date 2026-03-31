"""
Custom typed errors for TurboQuant to make runtime failures debuggable and testable.
"""


class TurboQuantError(Exception):
    """Base class for all TurboQuant errors."""

    pass


class TurboQuantConfigError(TurboQuantError, ValueError):
    """Raised when the configuration is invalid."""

    pass


class TurboQuantShapeError(TurboQuantError, ValueError):
    """Raised when tensor shapes or dimensions are incompatible or invalid."""

    pass


class TurboQuantStateError(TurboQuantError, ValueError):
    """Raised when there is an issue with saving, loading, or corrupt state."""

    pass


class TurboQuantKernelError(TurboQuantError, RuntimeError):
    """Raised when a fused kernel fails or is unsupported for the parameters."""

    pass


class TurboQuantCompatibilityError(TurboQuantError, TypeError):
    """Raised when there is an issue with mlx_lm upstream compatibility or adapter drift."""

    pass


class UnsupportedModelError(TurboQuantError, ValueError):
    """Raised when a model family is not in the TurboQuant supported allowlist.

    Thrown by :func:`turboquant.runtime.support.assert_supported_model_family`
    before any cache mutation occurs (Gate 2).
    """

    pass


class CompressionFailureError(TurboQuantError, RuntimeError):
    """Raised when the compression pipeline detects a fatal numerical failure.

    The caller is responsible for reverting to a dense cache and emitting a
    failure event.  See :class:`turboquant.runtime.events.UpgradeFailureEvent`.
    """

    pass

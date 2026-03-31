"""Internal persistence scaffolding.

TurboQuant does not currently expose a supported public persistence API.
This module exists only as internal scaffolding for future work and should
not be relied on as a stable surface.
"""

from __future__ import annotations


class PersistenceNotSupportedError(NotImplementedError):
    """Raised when callers attempt to use the unsupported persistence scaffold."""


def save_state(*args, **kwargs):
    raise PersistenceNotSupportedError(
        "TurboQuant persistence is not a supported public feature in this release."
    )


def load_state(*args, **kwargs):
    raise PersistenceNotSupportedError(
        "TurboQuant persistence is not a supported public feature in this release."
    )

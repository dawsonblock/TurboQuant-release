"""
turboquant.runtime.support — central model-family support gate.

This module is the single source of truth for which model families have
TurboQuant attention wiring and runtime-certification coverage.

Callers
-------
- ``integrations.mlx.upgrade.upgrade_cache_list`` — Gate 2 check
- ``tests/unit_static/test_governance.py``          — static contract tests

Rules for adding a new family
------------------------------
1. Wire its attention module to dispatch through TurboQuant's streaming
   attention path (see ``mlx_lm/models/gemma.py`` for the reference pattern).
2. Add runtime-certification coverage in
   ``scripts/certify_apple_runtime.sh`` / ``benchmarks/runtime_cert/``.
3. Add the normalised family name to :data:`SUPPORTED_FAMILIES`.
4. Update ``docs/support_matrix.md``.

Do NOT add a family here before completing steps 1–3.
"""

from __future__ import annotations

from turboquant.errors import UnsupportedModelError

# Families with explicit TurboQuant attention wiring and runtime-cert coverage.
# Any model_family string not in this set will be rejected at the upgrade
# boundary before any cache is mutated.
SUPPORTED_FAMILIES: frozenset[str] = frozenset({"llama", "gemma"})


def _normalize(name: str) -> str:
    """Normalise a raw model family string for allowlist lookup.

    Examples
    --------
    >>> _normalize("llama3_1")
    'llama'
    >>> _normalize("Gemma2")
    'gemma'
    """
    return name.lower().split("_")[0].rstrip("0123456789")


def is_supported_model_family(name: str) -> bool:
    """Return ``True`` if *name* belongs to a TurboQuant-supported family.

    Parameters
    ----------
    name:
        Raw model family string (e.g. ``"llama"``, ``"llama3_1"``,
        ``"gemma2"``).  Normalisation is applied before the lookup.
    """
    return _normalize(name) in SUPPORTED_FAMILIES


def assert_supported_model_family(name: str) -> None:
    """Raise :exc:`~turboquant.errors.UnsupportedModelError` if *name* is not
    in the supported allowlist.

    This is the canonical Gate 2 check.  Call it before mutating any cache.

    Parameters
    ----------
    name:
        Raw model family string.

    Raises
    ------
    UnsupportedModelError
        When ``_normalize(name)`` is not in :data:`SUPPORTED_FAMILIES`.
    """
    if not is_supported_model_family(name):
        raise UnsupportedModelError(
            f"Model family {name!r} is not supported by TurboQuant.  "
            f"Supported families: {sorted(SUPPORTED_FAMILIES)}.  "
            "To add support, wire the attention layer, add runtime-cert "
            "coverage, then add the normalised name to "
            "turboquant.runtime.support.SUPPORTED_FAMILIES."
        )

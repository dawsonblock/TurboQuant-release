"""
tests/unit_static/test_governance.py — structural governance contracts.

These tests verify invariants that must hold WITHOUT executing any MLX code.
They are the static enforcement layer for the repo's stated architectural
claims.  Every test here should pass on any machine (Linux CI, macOS x86,
developer laptops) regardless of whether MLX or Apple Silicon is present.

Contract coverage
-----------------
1. ``no_mlx_import_in_unit_static`` — none of the files in this directory
   import ``mlx``; if they did, they would not be portable and would
   undermine the static-test contract.

2. ``noxfile_excludes_unit_from_mlx_session`` — the noxfile's ``tests_mlx``
   session must NOT include ``tests/unit/`` now that all MLX-requiring tests
   have been moved to ``tests/integration_mlx/``.

3. ``support_module_has_expected_families`` — ``SUPPORTED_FAMILIES`` in
   ``turboquant/runtime/support.py`` contains exactly ``{"llama", "gemma"}``.
   Any addition to this set must be deliberate and come with runtime-cert
   coverage; if it silently changes, this test will catch it.

4. ``unsupported_family_raises_unsupported_model_error`` — calling
   ``assert_supported_model_family`` with an unlisted family must raise
   ``UnsupportedModelError``.  This verifies Gate 2 is wired correctly.

5. ``v_enabled_default_matches_architecture_doc`` — ``turboquant/config.py``
   defaults ``v_enabled=True``; the architecture doc must say "enabled by
   default".  Regression guard against the v_enabled contradiction fixed in
   Phase 4.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 1. No mlx import in this directory
# ---------------------------------------------------------------------------


def test_no_mlx_import_in_unit_static() -> None:
    """No file in tests/unit_static/ may import mlx."""
    this_dir = Path(__file__).parent
    violations: list[str] = []
    for py in sorted(this_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        if re.search(r"\bimport\s+mlx\b|from\s+mlx\b", text):
            violations.append(py.name)
    assert not violations, (
        "These files in tests/unit_static/ import mlx (forbidden):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# 2. noxfile excludes tests/unit/ from tests_mlx
# ---------------------------------------------------------------------------


def test_noxfile_excludes_unit_from_mlx_session() -> None:
    """tests/unit/ must NOT appear inside the tests_mlx nox session."""
    noxfile = REPO_ROOT / "noxfile.py"
    assert noxfile.exists(), "noxfile.py not found at repo root"

    text = noxfile.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # Walk the AST to find the tests_mlx function definition.
    mlx_session_src: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "tests_mlx":
            # Grab the source lines for just this function.
            lines = text.splitlines()
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            mlx_session_src = "\n".join(lines[start:end])
            break

    assert mlx_session_src is not None, (
        "Could not find 'tests_mlx' function in noxfile.py"
    )
    assert "tests/unit/" not in mlx_session_src, (
        "noxfile.py tests_mlx session still references 'tests/unit/'.\n"
        "All MLX-requiring tests must live in tests/integration_mlx/."
    )


# ---------------------------------------------------------------------------
# 3. SUPPORTED_FAMILIES content
# ---------------------------------------------------------------------------


def test_support_module_has_expected_families() -> None:
    """SUPPORTED_FAMILIES must be exactly {llama, gemma} — no silent additions."""
    support_py = REPO_ROOT / "turboquant" / "runtime" / "support.py"
    assert support_py.exists(), (
        "turboquant/runtime/support.py not found; create it as part of Phase 3."
    )

    text = support_py.read_text(encoding="utf-8")
    # Locate the frozenset literal via regex — avoids importing the module.
    m = re.search(
        r'SUPPORTED_FAMILIES\s*:\s*[^=]+=\s*frozenset\(\{([^}]+)\}\)', text
    )
    assert m is not None, (
        "Could not parse SUPPORTED_FAMILIES frozenset literal in support.py"
    )
    raw = m.group(1)
    # Extract quoted strings from the match.
    families = {s.strip().strip("\"'") for s in raw.split(",") if s.strip()}
    expected = {"llama", "gemma"}
    assert families == expected, (
        f"SUPPORTED_FAMILIES mismatch.\n"
        f"  Expected : {sorted(expected)}\n"
        f"  Got      : {sorted(families)}\n"
        "If you are adding a new family, update this test too — but only "
        "after completing the runtime-cert checklist in docs/support_matrix.md."
    )


# ---------------------------------------------------------------------------
# 4. Gate 2: unsupported family raises UnsupportedModelError
# ---------------------------------------------------------------------------


def test_unsupported_family_raises_unsupported_model_error() -> None:
    """assert_supported_model_family('mixtral') must raise UnsupportedModelError."""
    # Patch sys.path so we can import without installing the package.
    repo_str = str(REPO_ROOT)
    injected = repo_str not in sys.path
    if injected:
        sys.path.insert(0, repo_str)
    try:
        from turboquant.errors import UnsupportedModelError
        from turboquant.runtime.support import assert_supported_model_family

        with pytest.raises(UnsupportedModelError):
            assert_supported_model_family("mixtral")

        # Supported families must NOT raise.
        assert_supported_model_family("llama")
        assert_supported_model_family("gemma")
        assert_supported_model_family("llama3_1")  # normalisation check
        assert_supported_model_family("Gemma2")  # case-insensitive

    except ModuleNotFoundError as exc:
        pytest.skip(f"turboquant package not importable in this env: {exc}")
    finally:
        if injected:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# 5. v_enabled default matches architecture.md
# ---------------------------------------------------------------------------


def test_v_enabled_default_matches_architecture_doc() -> None:
    """architecture.md must state v_enabled is enabled by default (Phase 4 fix)."""
    arch_doc = REPO_ROOT / "docs" / "architecture.md"
    assert arch_doc.exists(), "docs/architecture.md not found"

    text = arch_doc.read_text(encoding="utf-8")

    # The fixed text says "enabled by default" — the old (wrong) text said
    # "disabled by default for some model families".
    assert "disabled by default for some model families" not in text, (
        "docs/architecture.md still claims V quantisation is disabled by "
        "default for some model families, which contradicts config.py "
        "(v_enabled=True).  Fix the documentation."
    )
    # Positive assertion: the corrected phrasing must be present.
    assert "enabled by default" in text, (
        "docs/architecture.md must explicitly state that v_enabled is "
        "enabled by default.  Check Phase 4 of the cleanup notes."
    )

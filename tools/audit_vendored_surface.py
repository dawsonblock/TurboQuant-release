#!/usr/bin/env python3
"""
tools/audit_vendored_surface.py — vendored mlx_lm surface governance audit.

This script cross-references ``VENDORED_MLX_LM.md`` against the ``mlx_lm/``
source tree to detect:

  1. **Missing patches** — files documented as TQ-modified that no longer
     contain any TurboQuant-specific identifiers.
  2. **Undocumented modifications** — files in ``mlx_lm/`` that contain
     TurboQuant-specific identifiers but are not listed as modified in the
     documentation.
  3. **Missing files** — files documented as modified that do not exist on
     disk (e.g. after an upstream refactor).

Usage
-----
    python tools/audit_vendored_surface.py           # human-readable
    python tools/audit_vendored_surface.py --json    # machine-readable

Exit codes
----------
  0  No violations found.
  1  One or more violations found (check output / JSON for details).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository root is the directory that contains this script's parent.
REPO_ROOT = Path(__file__).resolve().parents[1]

VENDORED_DOC = REPO_ROOT / "VENDORED_MLX_LM.md"
MLX_LM_DIR = REPO_ROOT / "mlx_lm"

# Identifiers that unambiguously mark TurboQuant-specific code.
TQ_MARKERS: tuple[str, ...] = (
    "TurboQuant",
    "turboquant",
    "TurboQuantKCache",
    "TurboQuantKeysView",
    "maybe_turboquant",
    "upgrade_cache_list",
    "turboquant_streaming_attention",
)

# ---------------------------------------------------------------------------
# Parse VENDORED_MLX_LM.md for the documented modified files
# ---------------------------------------------------------------------------

_MODIFIED_HEADER_RE = re.compile(
    r"^###\s+`(mlx_lm/[^`]+)`",
    re.MULTILINE,
)
_UNCHANGED_SECTION_RE = re.compile(
    r"### Files Unchanged from Upstream(.*?)(?=^##|\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_documented_modifications(doc: Path) -> set[str]:
    """Return set of repo-relative paths documented as TQ-modified."""
    text = doc.read_text(encoding="utf-8")
    return {m.group(1) for m in _MODIFIED_HEADER_RE.finditer(text)}


# ---------------------------------------------------------------------------
# Scan mlx_lm/ for files that contain TQ markers
# ---------------------------------------------------------------------------


def has_tq_marker(path: Path) -> bool:
    """Return True if *path* contains any TurboQuant-specific identifier."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in TQ_MARKERS)


def scan_mlx_lm(root: Path) -> set[str]:
    """Return repo-relative paths of .py files in mlx_lm/ with TQ markers."""
    result: set[str] = set()
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if has_tq_marker(py):
            result.add(str(py.relative_to(REPO_ROOT)))
    return result


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------


def run_audit() -> dict:
    """Run the full audit and return a structured results dict."""
    if not VENDORED_DOC.exists():
        return {
            "ok": False,
            "error": f"VENDORED_MLX_LM.md not found at {VENDORED_DOC}",
            "documented_modified": [],
            "missing_files": [],
            "missing_markers": [],
            "undocumented_modifications": [],
        }

    documented = parse_documented_modifications(VENDORED_DOC)
    actual = scan_mlx_lm(MLX_LM_DIR) if MLX_LM_DIR.exists() else set()

    # Files documented as modified but not found on disk.
    missing_files = sorted(
        p for p in documented if not (REPO_ROOT / p).exists()
    )

    # Files documented as modified but no TQ marker detected on disk.
    missing_markers = sorted(
        p
        for p in documented
        if (REPO_ROOT / p).exists() and not has_tq_marker(REPO_ROOT / p)
    )

    # Files with TQ markers that are NOT in the documented set.
    undocumented = sorted(actual - documented)

    ok = not (missing_files or missing_markers or undocumented)

    return {
        "ok": ok,
        "documented_modified": sorted(documented),
        "missing_files": missing_files,
        "missing_markers": missing_markers,
        "undocumented_modifications": undocumented,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_human(result: dict) -> None:
    print("=== TurboQuant Vendored Surface Audit ===")
    print(f"Status: {'OK' if result['ok'] else 'VIOLATIONS FOUND'}")
    print()

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    print(
        f"Documented modified files ({len(result['documented_modified'])}):"
    )
    for p in result["documented_modified"]:
        print(f"  {p}")

    if result["missing_files"]:
        print("\nMISSING FILES (documented but absent on disk):")
        for p in result["missing_files"]:
            print(f"  MISSING  {p}")

    if result["missing_markers"]:
        print("\nMISSING MARKERS (documented as modified, no TQ code found):")
        for p in result["missing_markers"]:
            print(f"  NO-TQ    {p}")

    if result["undocumented_modifications"]:
        print("\nUNDOCUMENTED MODIFICATIONS (TQ markers present, not listed):")
        for p in result["undocumented_modifications"]:
            print(f"  UNLISTED {p}")

    if result["ok"]:
        print("\nAll documented modifications are accounted for.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit vendored mlx_lm surface against VENDORED_MLX_LM.md"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    result = run_audit()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

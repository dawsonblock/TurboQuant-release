#!/usr/bin/env python3
"""
scripts/write_cert_manifest.py — write a structured certification manifest.

Called as the final step of ``scripts/certify_apple_runtime.sh`` to produce
a single machine-readable summary JSON that downstream tooling (CI artifact
uploaders, release gates) can parse without reading individual stage files.

Usage
-----
    python3 scripts/write_cert_manifest.py \\
        --artifact-dir artifacts/runtime-cert/20241201_120000 \\
        --passed 7 --failed 0 --total 7 \\
        --turboquant-version 0.2.2

Output
------
Writes ``<artifact-dir>/cert_manifest.json``.  Example structure::

    {
      "schema_version": "1",
      "turboquant_version": "0.2.2",
      "timestamp_utc": "2024-12-01T12:00:00Z",
      "platform": "darwin-arm64",
      "stages": {"passed": 7, "failed": 0, "total": 7},
      "result": "PASS",
      "artifact_dir": "artifacts/runtime-cert/20241201_120000",
      "files": ["preflight.json", "junit_cache_roundtrip.xml", ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    return f"{system}-{machine}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a structured cert manifest JSON."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Path to the timestamped artifact directory.",
    )
    parser.add_argument(
        "--passed",
        type=int,
        required=True,
        help="Number of stages that passed.",
    )
    parser.add_argument(
        "--failed",
        type=int,
        required=True,
        help="Number of stages that failed.",
    )
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="Total number of stages attempted.",
    )
    parser.add_argument(
        "--turboquant-version",
        default=os.environ.get("TQ_VERSION", "unknown"),
        help="TurboQuant package version string.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    if not artifact_dir.is_dir():
        print(
            f"ERROR: artifact-dir does not exist: {artifact_dir}",
            file=sys.stderr,
        )
        return 1

    # Enumerate all output files in the artifact directory.
    files = sorted(
        p.name
        for p in artifact_dir.iterdir()
        if p.is_file() and p.name != "cert_manifest.json"
    )

    result = "PASS" if args.failed == 0 else "FAIL"

    manifest = {
        "schema_version": "1",
        "turboquant_version": args.turboquant_version,
        "timestamp_utc": datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "platform": _platform_tag(),
        "python_version": platform.python_version(),
        "stages": {
            "passed": args.passed,
            "failed": args.failed,
            "total": args.total,
        },
        "result": result,
        "artifact_dir": str(artifact_dir.relative_to(Path.cwd()))
        if artifact_dir.is_relative_to(Path.cwd())
        else str(artifact_dir),
        "files": files,
    }

    out = artifact_dir / "cert_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"cert_manifest.json written: {out}")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

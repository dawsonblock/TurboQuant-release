#!/usr/bin/env python3
"""TurboQuant preflight checks — single authoritative runtime gate."""

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant Preflight Checks")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if MLX or Apple Silicon requirements are missing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON to stdout",
    )
    args = parser.parse_args()

    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "macos_version": platform.mac_ver()[0] or "unknown",
        "apple_silicon": False,
        "mlx_available": False,
        "mlx_device": None,
        "turboquant_version": None,
        "strict": args.strict,
        "pass": True,
        "errors": [],
    }

    if not args.json_output:
        print("Running TurboQuant Preflight Checks...")

    py_version = sys.version_info
    if py_version < (3, 9):
        result["pass"] = False
        result["errors"].append("Python >= 3.9 required")
        if not args.json_output:
            print("ERROR: Python >= 3.9 is required. You are running", sys.version)
        print(json.dumps(result, indent=2) if args.json_output else "")
        sys.exit(1)

    if not args.json_output:
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    result["apple_silicon"] = is_mac and is_arm

    if not result["apple_silicon"]:
        if not args.json_output:
            print("WARNING: You are not running on Apple Silicon (macOS + arm64).")
            print("         MLX acceleration will be disabled or unsupported.")
        if args.strict:
            result["pass"] = False
            result["errors"].append("Strict mode requires Apple Silicon")
            if not args.json_output:
                print("ERROR: Strict mode requires Apple Silicon.")
            print(json.dumps(result, indent=2) if args.json_output else "")
            sys.exit(1)
    elif not args.json_output:
        print("✓ Platform is Apple Silicon (darwin-arm64)")

    try:
        import mlx.core as mx

        result["mlx_available"] = True
        result["mlx_device"] = str(mx.default_device())
        if not args.json_output:
            print("✓ MLX backend initialized. Default device:", mx.default_device())
    except ImportError:
        if not args.json_output:
            print("WARNING: Could not import `mlx.core`.")
        if args.strict:
            result["pass"] = False
            result["errors"].append("Strict mode requires MLX")
            if not args.json_output:
                print("ERROR: Strict mode requires MLX. Ensure mlx is installed.")
            print(json.dumps(result, indent=2) if args.json_output else "")
            sys.exit(1)

    try:
        import turboquant

        result["turboquant_version"] = getattr(turboquant, "__version__", "unknown")
        if not args.json_output:
            print(f"✓ turboquant {result['turboquant_version']}")
    except ImportError:
        result["pass"] = False
        result["errors"].append("Cannot import turboquant")
        if not args.json_output:
            print("ERROR: Cannot import turboquant package.")

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print("\nPreflight checks complete.")

    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

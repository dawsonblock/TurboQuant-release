"""
benchmarks.runtime_cert.utils — shared utilities for Apple-Silicon certification.

Provides:
    - environment metadata collection
    - artifact directory helpers
    - JSON / CSV writers
    - memory measurement
    - prompt loading
    - timestamp helpers
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Timestamp / ID helpers
# ---------------------------------------------------------------------------


def now_utc_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def make_run_id(model: str, prompt_id: str, mode: str) -> str:
    """Build a deterministic run identifier."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = model.replace("/", "_").replace(" ", "_")[:40]
    return f"{ts}_{slug}_{prompt_id}_{mode}"


def git_commit_or_unknown() -> str:
    """Return the short git commit hash, or ``'unknown'``."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _macos_version() -> str:
    try:
        return platform.mac_ver()[0] or "unknown"
    except Exception:
        return "unknown"


def _mlx_version() -> str:
    try:
        import mlx  # type: ignore

        return getattr(mlx, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def _turboquant_version() -> str:
    try:
        import turboquant  # type: ignore

        return getattr(turboquant, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def collect_environment_metadata(
    *,
    model: str = "",
    mode: str = "",
    turboquant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect a reproducibility-critical metadata snapshot.

    Every benchmark artifact and certification output should embed this dict.
    """
    return {
        "timestamp": now_utc_iso(),
        "git_commit": git_commit_or_unknown(),
        "platform": platform.system().lower(),
        "macos_version": _macos_version(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": _mlx_version(),
        "turboquant_version": _turboquant_version(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "model": model,
        "mode": mode,
        "turboquant_config": turboquant_config or {},
    }


# ---------------------------------------------------------------------------
# Artifact directory helpers
# ---------------------------------------------------------------------------


def ensure_artifact_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if needed; return as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_artifact_dir(base: str | Path = "artifacts/runtime-cert") -> Path:
    """Return ``<base>/<YYYYMMDD_HHMMSS>`` and create it."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ensure_artifact_dir(Path(base) / ts)


# ---------------------------------------------------------------------------
# JSON / CSV writers
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write *payload* as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return a dict."""
    with Path(path).open() as f:
        return json.load(f)


def append_csv_row(
    path: str | Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
    """Append one row to a CSV file, creating headers on first write."""
    p = Path(path)
    exists = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``.jsonl`` prompt file.  Each line must have ``id`` and ``text``."""
    prompts: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def measure_peak_memory_bytes() -> int | None:
    """Return peak resident-set size in bytes (macOS only, best-effort)."""
    try:
        # macOS: ru_maxrss is in bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Config serialization helper
# ---------------------------------------------------------------------------


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a ``TurboQuantConfig`` (or dataclass) to a plain dict."""
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(config)
    return {}


# ---------------------------------------------------------------------------
# Run-result schema helper
# ---------------------------------------------------------------------------

# Canonical fieldnames for per-run CSV output
RUN_CSV_FIELDS: list[str] = [
    "run_id",
    "timestamp",
    "commit",
    "model",
    "mode",
    "prompt_id",
    "prompt_class",
    "prompt_length",
    "generated_tokens",
    "prefill_seconds",
    "decode_seconds",
    "total_seconds",
    "tokens_per_second",
    "peak_memory_bytes",
    "turboquant_active",
    "status",
    "error",
    "seed",
    "temperature",
]


def build_run_result(
    *,
    run_id: str,
    environment: dict[str, Any],
    model: str,
    mode: str,
    prompt_id: str,
    prompt_class: str,
    prompt_length: int,
    generated_tokens: int,
    prefill_seconds: float,
    decode_seconds: float,
    total_seconds: float,
    tokens_per_second: float,
    peak_memory_bytes: int | None,
    turboquant_active: bool,
    turboquant_config: dict[str, Any] | None,
    status: str,
    error: str | None = None,
    output_preview: str = "",
    seed: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Construct a single-run artifact dict conforming to the certification schema."""
    return {
        "run_id": run_id,
        "timestamp": environment.get("timestamp", now_utc_iso()),
        "commit": environment.get("git_commit", git_commit_or_unknown()),
        "environment": environment,
        "model": model,
        "mode": mode,
        "prompt_id": prompt_id,
        "prompt_class": prompt_class,
        "prompt_length": prompt_length,
        "generated_tokens": generated_tokens,
        "prefill_seconds": round(prefill_seconds, 4),
        "decode_seconds": round(decode_seconds, 4),
        "total_seconds": round(total_seconds, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "peak_memory_bytes": peak_memory_bytes,
        "turboquant_active": turboquant_active,
        "turboquant_config": turboquant_config or {},
        "status": status,
        "error": error,
        "output_preview": output_preview[:200] if output_preview else "",
        "seed": seed,
        "temperature": temperature,
    }

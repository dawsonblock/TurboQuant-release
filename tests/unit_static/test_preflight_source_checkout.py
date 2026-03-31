import json
import subprocess
import sys
from pathlib import Path


def test_preflight_runs_from_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(proc.stdout)
    assert "Cannot import turboquant" not in payload["errors"]
    assert payload["turboquant_version"] is not None

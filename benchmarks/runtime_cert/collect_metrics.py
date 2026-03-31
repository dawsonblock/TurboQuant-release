#!/usr/bin/env python3
"""
collect_metrics — aggregate raw JSON run artifacts into CSV + summary JSON.

Usage::

    python benchmarks/runtime_cert/collect_metrics.py \
        --input-dir artifacts/runtime-cert/20260329_120000 \
        --output-dir artifacts/runtime-cert/20260329_120000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.runtime_cert.utils import (
    RUN_CSV_FIELDS,
    append_csv_row,
    read_json,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate runtime certification metrics")
    p.add_argument(
        "--input-dir", required=True, help="Directory with raw run JSON files"
    )
    p.add_argument(
        "--output-dir", required=True, help="Directory for aggregate outputs"
    )
    return p.parse_args()


def collect_run_artifacts(input_dir: Path) -> list[dict]:
    """Load all ``*.json`` files in *input_dir* as run records."""
    records = []
    for f in sorted(input_dir.glob("*.json")):
        if f.name.startswith("certification_summary"):
            continue
        if f.name.startswith("aggregate"):
            continue
        try:
            records.append(read_json(f))
        except Exception as exc:
            print(f"  WARN: skipping {f.name}: {exc}")
    return records


def summarize_metrics(records: list[dict]) -> dict:
    """Compute aggregate summary from a list of run records."""
    total = len(records)
    passed = sum(1 for r in records if r.get("status") == "ok")
    failed = total - passed

    models = sorted(set(r.get("model", "") for r in records))
    prompt_classes = sorted(set(r.get("prompt_class", "") for r in records))

    # Group by (model, prompt_class, mode)
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = f"{r.get('model', '')}|{r.get('prompt_class', '')}|{r.get('mode', '')}"
        groups.setdefault(key, []).append(r)

    # Memory comparison: dense vs tq for each (model, prompt_class)
    memory_deltas = []
    speed_deltas = []
    for model in models:
        for pc in prompt_classes:
            dense_key = f"{model}|{pc}|dense"
            tq_key = f"{model}|{pc}|turboquant"
            dense_runs = groups.get(dense_key, [])
            tq_runs = groups.get(tq_key, [])
            if dense_runs and tq_runs:
                d_mem = _avg(r.get("peak_memory_bytes") for r in dense_runs)
                t_mem = _avg(r.get("peak_memory_bytes") for r in tq_runs)
                if d_mem and t_mem and d_mem > 0:
                    memory_deltas.append(
                        {
                            "model": model,
                            "prompt_class": pc,
                            "dense_peak_bytes": d_mem,
                            "tq_peak_bytes": t_mem,
                            "reduction_pct": round((1 - t_mem / d_mem) * 100, 1),
                        }
                    )
                d_tps = _avg(r.get("tokens_per_second") for r in dense_runs)
                t_tps = _avg(r.get("tokens_per_second") for r in tq_runs)
                if d_tps and t_tps and d_tps > 0:
                    speed_deltas.append(
                        {
                            "model": model,
                            "prompt_class": pc,
                            "dense_tps": round(d_tps, 2),
                            "tq_tps": round(t_tps, 2),
                            "delta_pct": round((t_tps / d_tps - 1) * 100, 1),
                        }
                    )

    overall_pass = failed == 0

    # Apply thresholds
    MIN_MEMORY_REDUCTION_PCT = 25.0
    MAX_SPEED_DEGRADATION_PCT = -25.0

    for m in memory_deltas:
        if m["reduction_pct"] < MIN_MEMORY_REDUCTION_PCT:
            overall_pass = False
            print(
                f"FAIL: Memory reduction {m['reduction_pct']}% < {MIN_MEMORY_REDUCTION_PCT}% for {m['model']}"
            )

    for s in speed_deltas:
        if s["delta_pct"] < MAX_SPEED_DEGRADATION_PCT:
            overall_pass = False
            print(
                f"FAIL: Speed degradation {s['delta_pct']}% < {MAX_SPEED_DEGRADATION_PCT}% for {s['model']}"
            )

    return {
        "total_runs": total,
        "passed": passed,
        "failed": failed,
        "models": models,
        "prompt_classes": prompt_classes,
        "memory_deltas": memory_deltas,
        "speed_deltas": speed_deltas,
        "overall_pass": overall_pass,
    }


def _avg(values) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def write_summary(
    output_dir: Path,
    records: list[dict],
    summary: dict,
) -> None:
    """Write aggregate CSV and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "aggregate_runs.csv"
    for r in records:
        flat = {
            "run_id": r.get("run_id", ""),
            "timestamp": r.get("timestamp", ""),
            "commit": r.get("commit", ""),
            "model": r.get("model", ""),
            "mode": r.get("mode", ""),
            "prompt_id": r.get("prompt_id", ""),
            "prompt_class": r.get("prompt_class", ""),
            "prompt_length": r.get("prompt_length", 0),
            "generated_tokens": r.get("generated_tokens", 0),
            "prefill_seconds": r.get("prefill_seconds", 0),
            "decode_seconds": r.get("decode_seconds", 0),
            "total_seconds": r.get("total_seconds", 0),
            "tokens_per_second": r.get("tokens_per_second", 0),
            "peak_memory_bytes": r.get("peak_memory_bytes", ""),
            "turboquant_active": r.get("turboquant_active", False),
            "status": r.get("status", ""),
            "error": r.get("error", ""),
        }
        append_csv_row(csv_path, flat, RUN_CSV_FIELDS)

    write_json(output_dir / "certification_summary.json", summary)
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {output_dir / 'certification_summary.json'}")


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    records = collect_run_artifacts(input_dir)
    if not records:
        print(f"No run artifacts found in {input_dir}")
        return 1

    print(f"Collected {len(records)} run artifacts")

    summary = summarize_metrics(records)
    write_summary(output_dir, records, summary)

    if summary["overall_pass"]:
        print(
            f"\nCERTIFICATION PASS — {summary['passed']}/{summary['total_runs']} runs succeeded"
        )
    else:
        print(
            f"\nCERTIFICATION FAIL — {summary['failed']}/{summary['total_runs']} runs failed"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

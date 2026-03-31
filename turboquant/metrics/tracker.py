"""
turboquant.metrics.tracker — per-run metrics collection and artifact writer.

Every generation run that uses TurboQuant should create one
:class:`MetricsTracker` instance, update it with observed measurements,
then call :meth:`MetricsTracker.write` at the end of the run to produce
the canonical artifact layout::

    runs/
      <run_id>/
        metrics.json    ← top-level scalar summary
        events.jsonl    ← one JSON object per upgrade/failure event

The metrics artifact is the **only** authoritative source for claimed
compression numbers.  README examples and docs must be labelled as
illustrative unless backed by an artifact from this tracker.

Usage
-----
::

    from turboquant.metrics.tracker import MetricsTracker

    tracker = MetricsTracker(
        run_id="llama3_3bit_g64_001",
        model="llama",
        artifact_root=Path("runs"),
    )

    tracker.set_dense_bytes(dense_cache_bytes)
    tracker.set_compressed_bytes(compressed_cache_bytes)
    tracker.record_step(tokens_generated=1, latency_ms=12.4)

    metrics = tracker.write()   # writes runs/llama3_3bit_g64_001/metrics.json
    print(metrics)
    # {
    #   "run_id": "llama3_3bit_g64_001",
    #   "model": "llama",
    #   "dense_bytes": 262144,
    #   "compressed_bytes": 65536,
    #   "ratio": 4.0,
    #   "tok_per_sec": 24.3,
    #   "latency_ms": 12.4,
    #   ...
    # }
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class MetricsTracker:
    """Accumulate per-run metrics and write the canonical artifact.

    Parameters
    ----------
    run_id:
        Unique identifier for this run.  If ``None``, a UUID4 is generated.
    model:
        Model family string (e.g. ``"llama"`` or ``"gemma"``).
    artifact_root:
        Directory under which ``<run_id>/`` will be created.  Defaults to
        ``runs/`` relative to the current working directory.
    config_fingerprint:
        Optional :attr:`~turboquant.config.TurboQuantConfig.fingerprint`
        string to link the artifact to a specific compression config.
    """

    def __init__(
        self,
        run_id: str | None = None,
        model: str = "unknown",
        artifact_root: Path | None = None,
        config_fingerprint: str | None = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())[:8]
        self.model = model
        self.artifact_root = artifact_root or Path("runs")
        self.config_fingerprint = config_fingerprint

        # ── Measurements ─────────────────────────────────────────────────────
        self._dense_bytes: int = 0
        self._compressed_bytes: int = 0

        # Step-level accumulators
        self._step_count: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._peak_latency_ms: float = 0.0

        # MLX version (populated lazily)
        self._mlx_version: str | None = None

        # Timestamp
        self._started_at: float = time.time()

    # ── Measurement setters ───────────────────────────────────────────────────

    def set_dense_bytes(self, n: int) -> None:
        """Record the dense KV cache byte count (before compression)."""
        self._dense_bytes = int(n)

    def set_compressed_bytes(self, n: int) -> None:
        """Record the compressed KV cache byte count (after upgrade)."""
        self._compressed_bytes = int(n)

    def record_step(
        self,
        tokens_generated: int = 1,
        latency_ms: float = 0.0,
    ) -> None:
        """Record one generation step.

        Parameters
        ----------
        tokens_generated:
            Number of new tokens produced in this step (usually 1 for
            autoregressive decode, more for speculative or batch decode).
        latency_ms:
            Wall-clock time for this step in milliseconds.
        """
        self._step_count += 1
        self._total_tokens += tokens_generated
        self._total_latency_ms += latency_ms
        if latency_ms > self._peak_latency_ms:
            self._peak_latency_ms = latency_ms

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def ratio(self) -> float:
        """Compression ratio (dense / compressed).  0.0 if unknown."""
        if self._compressed_bytes == 0:
            return 0.0
        return round(self._dense_bytes / self._compressed_bytes, 4)

    @property
    def tok_per_sec(self) -> float:
        """Average tokens-per-second across all recorded steps."""
        if self._total_latency_ms == 0:
            return 0.0
        return round(self._total_tokens / (self._total_latency_ms / 1000.0), 2)

    @property
    def avg_latency_ms(self) -> float:
        """Average per-step wall-clock latency in milliseconds."""
        if self._step_count == 0:
            return 0.0
        return round(self._total_latency_ms / self._step_count, 3)

    # ── Artifact output ───────────────────────────────────────────────────────

    def _get_mlx_version(self) -> str:
        if self._mlx_version is None:
            try:
                import mlx.core as mx

                self._mlx_version = getattr(mx, "__version__", "unknown")
            except ImportError:
                self._mlx_version = "unavailable"
        return self._mlx_version

    def to_dict(self) -> dict:
        """Return the metrics summary as a plain dict."""
        return {
            "run_id": self.run_id,
            "model": self.model,
            "config_fingerprint": self.config_fingerprint,
            "dense_bytes": self._dense_bytes,
            "compressed_bytes": self._compressed_bytes,
            "ratio": self.ratio,
            "tok_per_sec": self.tok_per_sec,
            "avg_latency_ms": self.avg_latency_ms,
            "peak_latency_ms": round(self._peak_latency_ms, 3),
            "total_tokens": self._total_tokens,
            "total_steps": self._step_count,
            "mlx_version": self._get_mlx_version(),
            "elapsed_s": round(time.time() - self._started_at, 3),
        }

    def write(
        self,
        event_log=None,
    ) -> dict:
        """Write ``metrics.json`` (and optionally flush ``events.jsonl``).

        Parameters
        ----------
        event_log:
            Optional :class:`~turboquant.runtime.events.EventLog` instance.
            If provided, its ``flush()`` method is called, and its
            :meth:`~turboquant.runtime.events.EventLog.summary` is included
            in ``metrics.json`` under the key ``"events"``.

        Returns
        -------
        dict
            The metrics dict that was written to disk.
        """
        run_dir = self.artifact_root / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        d = self.to_dict()

        if event_log is not None:
            event_log._artifact_dir = run_dir
            event_log.flush()
            d["events"] = event_log.summary()

        out = run_dir / "metrics.json"
        out.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        return d

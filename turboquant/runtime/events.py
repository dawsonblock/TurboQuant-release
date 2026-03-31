"""
turboquant.runtime.events — structured upgrade and failure event log.

Every cache upgrade or compression failure produces a structured event.
Events are written to ``events.jsonl`` in the run artifact directory so
that CI and offline analysis can verify the exact transition point and its
memory effect without re-running inference.

Event types
-----------
:class:`CacheUpgradeEvent`
    Emitted when a dense ``KVCache`` layer is promoted to
    ``TurboQuantKCache``.  Records the token index, old/new byte counts,
    and the compression ratio achieved.

:class:`UpgradeFailureEvent`
    Emitted when a compression failure (NaN, Inf, scale == 0, or other
    numerical exception) causes the upgrade to be aborted and the layer
    to revert to the original dense cache.

:class:`EventLog`
    Accumulates events in memory and can flush them to ``events.jsonl``
    (newline-delimited JSON).  One ``EventLog`` per generation run.

Invariants
----------
* Each sequence layer is upgraded **at most once**.  A second attempt for
  the same layer index must produce a no-op or an ``UpgradeFailureEvent``.
* ``UpgradeFailureEvent.reverted`` is always ``True`` — the caller must
  restore the original cache entry before emitting the event.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

logger = logging.getLogger("turboquant.runtime.events")

# ── Event dataclasses ─────────────────────────────────────────────────────────


@dataclass
class CacheUpgradeEvent:
    """Record of a successful dense → compressed cache upgrade.

    Attributes
    ----------
    event_type:
        Always ``"upgrade"`` — used as a discriminator in the JSONL stream.
    layer_index:
        Zero-based layer index in the per-layer cache list.
    token_index:
        ``cache.offset`` at the moment the upgrade decision was made.
        This is the first token position that triggered the threshold.
    old_type:
        Class name of the cache before upgrade (typically ``"KVCache"``).
    new_type:
        Class name after upgrade (typically ``"TurboQuantKCache"``).
    old_bytes:
        Estimated byte footprint of the **dense** cache at ``token_index``.
        Computed as ``token_index * n_heads * head_dim * 2 * 2`` (float16,
        both K and V) when the dense cache does not expose ``.nbytes``.
    new_bytes:
        Actual byte footprint of the compressed cache immediately after
        upgrade (``TurboQuantKCache.nbytes``).
    ratio:
        ``old_bytes / new_bytes``.  Values > 1.0 indicate compression.
        ``0.0`` when ``new_bytes == 0`` (defensive).
    timestamp_utc:
        Unix timestamp (float seconds) when the event was recorded.
    """

    event_type: str = field(default="upgrade", init=False)
    layer_index: int = 0
    token_index: int = 0
    old_type: str = "KVCache"
    new_type: str = "TurboQuantKCache"
    old_bytes: int = 0
    new_bytes: int = 0
    ratio: float = 0.0
    timestamp_utc: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.new_bytes > 0:
            self.ratio = round(self.old_bytes / self.new_bytes, 4)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UpgradeFailureEvent:
    """Record of a compression failure that caused a dense-cache revert.

    Attributes
    ----------
    event_type:
        Always ``"upgrade_failure"``.
    layer_index:
        Zero-based layer index that failed to upgrade.
    token_index:
        ``cache.offset`` at the time of failure.
    reason:
        Short human-readable description of the failure cause (e.g.
        ``"NaN in K scales"``, ``"scale == 0"``, ``"Inf in packed codes"``).
    reverted:
        Always ``True`` — the caller must restore the original dense cache
        before emitting this event.
    exception_type:
        ``type(exc).__name__`` of the caught exception, or ``""`` if the
        failure was detected without an exception.
    timestamp_utc:
        Unix timestamp when the event was recorded.
    """

    event_type: str = field(default="upgrade_failure", init=False)
    layer_index: int = 0
    token_index: int = 0
    reason: str = ""
    reverted: bool = True
    exception_type: str = ""
    timestamp_utc: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


AnyEvent = Union[CacheUpgradeEvent, UpgradeFailureEvent]


# ── EventLog ──────────────────────────────────────────────────────────────────


class EventLog:
    """Accumulate events in memory; flush to ``events.jsonl`` on demand.

    Parameters
    ----------
    artifact_dir:
        Directory under which ``events.jsonl`` will be written.  Created
        on first :meth:`flush` if it does not exist.  Pass ``None`` to
        accumulate events in memory only (useful in tests).

    Usage
    -----
    ::

        log = EventLog(artifact_dir=Path("runs/run_001"))

        log.record(CacheUpgradeEvent(
            layer_index=0,
            token_index=512,
            old_bytes=262144,
            new_bytes=65536,
        ))

        log.flush()          # writes runs/run_001/events.jsonl
        log.flush()          # idempotent — appends new events only
    """

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self._artifact_dir = artifact_dir
        self._events: list[AnyEvent] = []
        self._flushed_count: int = 0

    @property
    def events(self) -> list[AnyEvent]:
        """All recorded events (including those already flushed)."""
        return list(self._events)

    def record(self, event: AnyEvent) -> None:
        """Append *event* to the in-memory log.

        Parameters
        ----------
        event:
            A :class:`CacheUpgradeEvent` or :class:`UpgradeFailureEvent`.
        """
        self._events.append(event)
        logger.debug(
            "event recorded: type=%s layer=%s token=%s",
            event.event_type,
            getattr(event, "layer_index", "?"),
            getattr(event, "token_index", "?"),
        )

    def flush(self) -> Path | None:
        """Write any unwritten events to ``events.jsonl``.

        Returns
        -------
        Path | None
            Absolute path to the JSONL file, or ``None`` if no artifact
            directory was configured.

        Notes
        -----
        Only events recorded *since the last flush* are written (append
        semantics).  This makes it safe to call ``flush()`` after every
        generation step without duplicating earlier events.
        """
        new_events = self._events[self._flushed_count :]
        if not new_events:
            return None
        if self._artifact_dir is None:
            self._flushed_count = len(self._events)
            return None

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        out = self._artifact_dir / "events.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            for ev in new_events:
                fh.write(json.dumps(ev.to_dict()) + "\n")

        self._flushed_count = len(self._events)
        logger.debug("events.jsonl flushed: %d new events → %s", len(new_events), out)
        return out

    def upgrade_count(self) -> int:
        """Return number of successful upgrade events recorded."""
        return sum(
            1 for e in self._events if isinstance(e, CacheUpgradeEvent)
        )

    def failure_count(self) -> int:
        """Return number of upgrade failure events recorded."""
        return sum(
            1 for e in self._events if isinstance(e, UpgradeFailureEvent)
        )

    def summary(self) -> dict:
        """Return a summary dict for inclusion in ``metrics.json``.

        Keys: ``upgrades``, ``failures``, ``total_events``.
        """
        return {
            "upgrades": self.upgrade_count(),
            "failures": self.failure_count(),
            "total_events": len(self._events),
        }

"""
integrations.mlx.upgrade — production KV-cache upgrade policy.

This module owns the policy for when and how to promote a dense KVCache to a
TurboQuantKCache.  It is intentionally separate from ``generate.py`` so that
the policy can be unit-tested, reused across frontends, and evolved without
touching the main generation loop.

The legacy helper ``maybe_turboquant_k_cache`` in ``generate.py`` now
delegates to :func:`upgrade_cache_list`.

Usage
-----
    from turboquant.config import TurboQuantConfig
    from integrations.mlx.upgrade import upgrade_cache_list

    config = TurboQuantConfig(k_bits=3, k_group_size=64, ...)
    events = upgrade_cache_list(prompt_cache, k_start=512, config=config)
    for ev in events:
        if ev.upgraded:
            print(f"layer {ev.layer_index}: {ev.old_type} → {ev.new_type} "
                  f"at offset {ev.offset_at_upgrade}")
"""

from __future__ import annotations

from dataclasses import dataclass

from turboquant.runtime.events import EventLog, CacheUpgradeEvent as RTCacheUpgradeEvent
from turboquant.runtime.support import assert_supported_model_family

# Module-level event log — flushed to runs/<run_id>/events.jsonl automatically.
_event_log: EventLog = EventLog()

# ── Event ─────────────────────────────────────────────────────────────────────


@dataclass
class CacheUpgradeEvent:
    """Record of a single cache-layer upgrade decision.

    Fields
    ------
    upgraded:
        ``True`` if the layer was promoted to TurboQuantKCache.
    layer_index:
        Zero-based index of the layer in *prompt_cache*.
    old_type:
        ``type(cache).__name__`` before the upgrade (or the same type if
        no upgrade occurred).
    new_type:
        ``type(cache).__name__`` after the upgrade.
    offset_at_upgrade:
        ``cache.offset`` at the moment the decision was made.
    old_bytes:
        Approximate byte footprint of the dense cache before upgrade.
        Set to 0 when ``upgraded=False``.
    new_bytes:
        Approximate byte footprint of the TurboQuant cache after upgrade.
        Set to 0 when ``upgraded=False``.
    ratio:
        ``new_bytes / old_bytes`` — compression ratio (<1.0 is smaller).
        0.0 when ``upgraded=False`` or ``old_bytes=0``.
    """

    upgraded: bool
    layer_index: int
    old_type: str
    new_type: str
    offset_at_upgrade: int
    old_bytes: int = 0
    new_bytes: int = 0
    ratio: float = 0.0


# ── Upgrade policy ────────────────────────────────────────────────────────────


def upgrade_cache_list(
    prompt_cache: list,
    k_start: int | None,
    config: object,  # type: ignore
    model_family: str | None = None,
) -> list[CacheUpgradeEvent]:
    """Promote KVCache entries to TurboQuantKCache when their offset threshold
    is reached.

    This is the canonical upgrade path used by the mlx-lm generation loop.
    Call once per generation step; the function is idempotent — layers that
    have already been upgraded are skipped.

    Parameters
    ----------
    prompt_cache:
        The per-layer cache list.  Modified in place when an upgrade occurs.
    k_start:
        Minimum ``cache.offset`` before upgrading.  ``None`` disables all
        upgrades (every layer stays as-is).
    config:
        :class:`turboquant.config.TurboQuantConfig` governing compression.
        The production path always uses ``return_mode="view"``; the legacy
        ``return_mode`` kwarg is not surfaced here.
    model_family:
        Model architecture family (e.g. ``"llama"`` or ``"gemma"``).
        Must be in the supported allowlist or
        :class:`~turboquant.errors.UnsupportedModelError` is raised before
        any cache is mutated.  Pass ``None`` only from exploratory code
        paths that intentionally bypass the allowlist check.

    Returns
    -------
    List[CacheUpgradeEvent]
        One event per cache layer, in order.  Inspect ``ev.upgraded`` to
        see which layers were promoted this call.
    """
    # Gate 2 — model allowlist.  Must be checked before any cache mutation.
    if model_family is not None:
        assert_supported_model_family(model_family)

    # Lazy import to avoid circular deps and to keep this module importable
    # even if turboquant or mlx_lm is not fully initialised.
    from integrations.mlx.cache_adapter import TurboQuantConfig, TurboQuantKCache

    events: list[CacheUpgradeEvent] = []

    if k_start is None:
        # Fast path: no upgrade policy in effect.
        for i, c in enumerate(prompt_cache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=type(c).__name__,
                    new_type=type(c).__name__,
                    offset_at_upgrade=getattr(c, "offset", 0),
                )
            )
        return events

    for i, c in enumerate(prompt_cache):
        old_type = type(c).__name__
        cur_offset = getattr(c, "offset", 0)

        # Already upgraded — skip.
        if isinstance(c, TurboQuantKCache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Threshold not yet reached or missing required properties to extract keys/values.
        if cur_offset < k_start or not hasattr(c, "keys") or not hasattr(c, "values"):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Canonical upgrade path: use the production config to populate the cache directly.
        # We wrap it in a legacy shim config for the adapter but pass the true fields.
        legacy_cfg = TurboQuantConfig(
            main_bits=config.k_bits,  # type: ignore
            group_size=config.k_group_size,  # type: ignore
            rotation=config.rotation,  # type: ignore
            return_mode="view",  # type: ignore
            scale_dtype=config.scale_dtype,  # type: ignore
            resid_scale_bits=8,  # legacy fallback  # type: ignore
            residual_topk=config.residual_topk,  # type: ignore
            v_bits=config.v_bits,  # type: ignore
            v_group_size=config.v_group_size,  # type: ignore
            v_scale_dtype=config.v_scale_dtype,  # type: ignore
            v_enabled=config.v_enabled,  # type: ignore
            block_tokens=config.block_tokens,  # type: ignore
        )
        tq = TurboQuantKCache(legacy_cfg)
        if getattr(c, "keys", None) is not None:
            tq.update_and_fetch(
                c.keys[..., :cur_offset, :], c.values[..., :cur_offset, :]
            )

        prompt_cache[i] = tq

        # Compute byte footprints for the event record.
        old_b = getattr(c, "byte_size", lambda: 0)()
        new_b = tq.byte_size() if hasattr(tq, "byte_size") else 0
        ev_ratio = (new_b / old_b) if old_b > 0 else 0.0

        ev = CacheUpgradeEvent(
            upgraded=True,
            layer_index=i,
            old_type=old_type,
            new_type=type(prompt_cache[i]).__name__,
            offset_at_upgrade=cur_offset,
            old_bytes=old_b,
            new_bytes=new_b,
            ratio=round(ev_ratio, 4),
        )
        events.append(ev)
        _event_log.record(
            RTCacheUpgradeEvent(
                layer_index=ev.layer_index,
                token_index=ev.offset_at_upgrade,
                old_type=ev.old_type,
                new_type=ev.new_type,
                old_bytes=ev.old_bytes,
                new_bytes=ev.new_bytes,
                ratio=ev.ratio,
            )
        )

    return events

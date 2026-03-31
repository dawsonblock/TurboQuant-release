"""
turboquant.core.types — canonical compressed tensor types and backend protocol.

This module defines:

* :class:`CompressedK` — a named container for all arrays that represent a
  compressed key block (packed codes + scales + optional sparse residual).
* :class:`CompressedV` — same for values (no residual, optional).
* :class:`KVCompressionBackend` — the protocol every compression backend
  must satisfy.  ``KVCompressor`` is the production implementation.

Nothing in this module imports MLX directly; it re-exports
:class:`~turboquant.runtime.kv_interface.TurboQuantKeysView` as a
convenience so callers have a single import point for the types they need
to work with the streaming attention path.

Design contract
---------------
* ``CompressedK.byte_size()`` and ``CompressedV.byte_size()`` return the
  *allocated* byte count of the underlying arrays, not the *logical* token
  count.  Divide by tokens to get bytes-per-token.
* ``KVCompressionBackend`` is a ``typing.Protocol`` — duck typing is fine;
  you do not need to inherit from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# ── Compressed tensor containers ──────────────────────────────────────────────


@dataclass
class CompressedK:
    """Container for a compressed key block.

    All arrays are MLX tensors; none of them are dense ``[B, H, T, D]``
    float arrays.  The only way to recover a dense tensor is via
    :meth:`KVCompressionBackend.decompress_k`.

    Attributes
    ----------
    packed:
        Bit-packed N-bit codes.  Shape ``[B, H, T, n_words]`` uint32.
    scales:
        Per-group quantisation scales.  Shape ``[B, H, T, n_groups]`` fp16.
    resid_vals:
        Top-k residual values per group.  Shape ``[B, H, T, n_groups, k]``
        fp16, or ``None`` when ``residual_topk == 0``.
    resid_idx:
        Top-k residual indices.  Shape ``[B, H, T, n_groups, k]`` uint8,
        or ``None`` when ``residual_topk == 0``.
    k_bits:
        Bit width used for the packed codes (e.g. 3 or 4).
    d_head:
        Original (unpadded) head dimension — required for correct unpack.
    """

    packed: object  # mx.array — avoid importing mlx at module level
    scales: object  # mx.array
    resid_vals: object | None = None
    resid_idx: object | None = None
    k_bits: int = 3
    d_head: int = 0

    def byte_size(self) -> int:
        """Return total allocated bytes across all stored arrays."""
        total = 0
        for attr in ("packed", "scales", "resid_vals", "resid_idx"):
            a = getattr(self, attr)
            if a is not None:
                total += int(a.nbytes)
        return total


@dataclass
class CompressedV:
    """Container for a compressed value block.

    Values use a separate bit-width and group size from keys; no rotation
    and no residual are applied.

    Attributes
    ----------
    packed:
        Bit-packed M-bit codes.  Shape ``[B, H, T, n_words]`` uint32.
    scales:
        Per-group scales.  Shape ``[B, H, T, n_groups]`` fp16.
    v_bits:
        Bit width used for the packed codes (e.g. 4).
    d_head:
        Original (unpadded) value head dimension.
    """

    packed: object  # mx.array
    scales: object  # mx.array
    v_bits: int = 4
    d_head: int = 0

    def byte_size(self) -> int:
        """Return total allocated bytes across all stored arrays."""
        total = 0
        for attr in ("packed", "scales"):
            a = getattr(self, attr)
            if a is not None:
                total += int(a.nbytes)
        return total


# ── Backend protocol ──────────────────────────────────────────────────────────


class KVCompressionBackend(Protocol):
    """Protocol every compression backend must satisfy.

    The production implementation is
    :class:`~turboquant.runtime.kv_interface.KVCompressor`.  An in-memory
    dense backend (for testing / baseline comparison) just stores float16
    arrays and returns them unchanged.

    Execution contract
    ------------------
    The path is exactly::

        Dense KV → compress_k / compress_v
                 → store CompressedK / CompressedV
                 → decompress_k / decompress_v (on read, inside attention)
                 → Attention

    No path exists where a dense tensor is stored after upgrade.  Every
    read from the cache must go through ``decompress_*``.
    """

    def compress_k(self, k: object) -> CompressedK:
        """Compress a dense key tensor to ``CompressedK``.

        Parameters
        ----------
        k:
            ``mx.array`` of shape ``[B, H, T, D]`` — any float dtype.

        Returns
        -------
        CompressedK
            Compressed representation.  ``byte_size()`` must be measurably
            smaller than ``k.nbytes`` for any non-trivial *T*.
        """
        ...

    def compress_v(self, v: object) -> CompressedV:
        """Compress a dense value tensor to ``CompressedV``.

        Parameters
        ----------
        v:
            ``mx.array`` of shape ``[B, H, T, D_v]`` — any float dtype.
        """
        ...

    def decompress_k(self, ck: CompressedK) -> object:
        """Recover a dense key tensor from ``CompressedK``.

        Returns
        -------
        mx.array
            Approximately reconstructed ``[B, H, T, D]`` in the *rotated*
            coordinate frame.  Queries must also be rotated before attending.
        """
        ...

    def decompress_v(self, cv: CompressedV) -> object:
        """Recover a dense value tensor from ``CompressedV``.

        Returns
        -------
        mx.array
            Approximately reconstructed ``[B, H, T, D_v]``.
        """
        ...

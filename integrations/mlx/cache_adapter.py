import warnings
from dataclasses import dataclass
from typing import Optional

from turboquant.config import TurboQuantConfig as _ProdTurboQuantConfig

from mlx_lm.models.cache import _BaseCache, create_attention_mask
from turboquant.runtime.kv_interface import (
    KVCompressor as _KVCompressor,
)

#   - TurboQuantConfig   (legacy field names: main_bits, group_size, ...)
#   - TurboQuantKeysView (re-exported from production package)
#   - TurboQuantKCache   (adapter; same attribute surface as the old class)
# ---------------------------------------------------------------------------
# Re-export the canonical view/compressor from the production package.
# gemma.py and tests import TurboQuantKeysView from .cache (this module).
from turboquant.runtime.kv_interface import (
    TurboQuantKeysView,
)


@dataclass
class TurboQuantConfig:
    """Legacy TurboQuant config shim that maps old mlx-lm field names to the
    production :class:`turboquant.config.TurboQuantConfig`.

    Fields kept verbatim for backward compatibility with existing callers,
    serialised checkpoints, and test fixtures.
    """

    main_bits: int = 3
    group_size: int = 64
    rotation: str = "identity"  # "identity" | "hadamard" | "random_orthogonal"
    residual: str = "group_proj"  # legacy; ignored in production path
    return_mode: str = (
        "view"  # legacy metadata only; production path always returns a view
    )
    block_tokens: int = 256
    scale_dtype: str = "float16"
    resid_scale_bits: int = 8  # legacy adapter metadata only
    residual_topk: int = 2  # production sparse residual count
    v_bits: int = 4
    v_group_size: int = 64
    v_scale_dtype: str = "float16"
    v_enabled: bool = True
    eps: float = 1e-6


def _to_prod_config(cfg: TurboQuantConfig) -> _ProdTurboQuantConfig:
    """Map legacy TurboQuantConfig field names to the production dataclass."""
    return _ProdTurboQuantConfig(
        k_bits=cfg.main_bits,
        k_group_size=cfg.group_size,
        v_bits=cfg.v_bits,
        v_group_size=cfg.v_group_size,
        v_enabled=cfg.v_enabled,
        rotation=cfg.rotation,  # type: ignore
        residual_topk=cfg.residual_topk,
        block_tokens=cfg.block_tokens,
        scale_dtype=cfg.scale_dtype,  # type: ignore
        v_scale_dtype=cfg.v_scale_dtype,  # type: ignore
        eps=cfg.eps,
    )


class TurboQuantKCache(_BaseCache):
    """Thin adapter: preserves the legacy TurboQuantKCache API while
    delegating all compression/rotation/bit-packing to KVCompressor.

    Preserved public attributes (required by tests and callers):
        offset, k_codes, k_scales, v_codes, v_scales,
        state (property), meta_state (property),
        from_state (classmethod, 2-arg legacy signature),
        update_and_fetch, iter_rotated_kv_blocks,
        rotate_queries_for_attention, trim, is_trimmable,
        size, nbytes, storage_breakdown, config.block_tokens
    """

    step = 512

    def __init__(self, config: Optional[TurboQuantConfig] = None) -> None:
        self.config = config or TurboQuantConfig()
        self._return_mode: str = "view"
        if self.config.return_mode not in ("view", "", None):
            warnings.warn(
                "TurboQuant legacy return_mode is ignored by the production adapter; "
                "streaming path always returns TurboQuantKeysView.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._impl = _KVCompressor(_to_prod_config(self.config))

    # ------------------------------------------------------------------
    # _BaseCache size API
    # ------------------------------------------------------------------

    def size(self) -> int:
        return self._impl.offset

    def __len__(self) -> int:
        return self._impl.offset

    def empty(self) -> bool:
        return self._impl.k_packed is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return self._impl.trim(n)

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self._impl.offset, **kwargs)

    # ------------------------------------------------------------------
    # offset
    # ------------------------------------------------------------------

    @property
    def offset(self) -> int:
        return self._impl.offset

    @offset.setter
    def offset(self, v: int) -> None:
        self._impl.offset = v

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        return self._impl.memory_breakdown()["total"]  # type: ignore

    def storage_breakdown(self) -> dict:
        bd = self._impl.memory_breakdown()
        return {
            "k_codes": bd.get("k_packed", 0),
            "k_scales": bd.get("k_scales", 0),
            "k_resid_scale_q": 0,
            "k_resid_scale_max": 0,
            "k_resid_proj_signs": 0,
            "v_codes": bd.get("v_packed", 0),
            "v_scales": bd.get("v_scales", 0),
            "total": bd.get("total", 0),
        }

    # ------------------------------------------------------------------
    # Buffer access (for tests that inspect k_codes, k_scales, ...)
    # ------------------------------------------------------------------

    @property
    def k_codes(self):
        """Packed 3/4-bit K codes -- [B, H, T, n_words] uint32."""
        return self._impl.k_packed

    @property
    def k_scales(self):
        """Per-group K scales -- [B, H, T, n_groups]."""
        return self._impl.k_scales

    @property
    def k_resid_scale_q(self):
        """Legacy field -- not present in production path; always None."""
        return None

    @property
    def k_resid_scale_max(self):
        """Legacy field -- not present in production path; always None."""
        return None

    @property
    def k_resid_proj_signs(self):
        """Legacy field -- not present in production path; always None."""
        return None

    @property
    def v_codes(self):
        """Packed V codes -- [B, H, T, n_words] uint32."""
        return self._impl.v_packed

    @property
    def v_scales(self):
        """Per-group V scales -- [B, H, T, n_groups]."""
        return self._impl.v_scales

    # ------------------------------------------------------------------
    # Main cache API
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys, values):
        """Compress and store keys/values; return (k_out, v_out).

        Legacy return_mode metadata is ignored by the production adapter.
        Always returns (TurboQuantKeysView, values) for the streaming path.
        """
        view, _ = self._impl.update_and_fetch(keys, values)
        return view, values

    def iter_rotated_kv_blocks(
        self,
        view: TurboQuantKeysView,
        values_unused=None,
        block_tokens: Optional[int] = None,
    ):
        """Yield (start, end, k_rotated, v_block) for streaming attention."""
        yield from self._impl.iter_rotated_kv_blocks(view, block_tokens=block_tokens)

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    @property
    def state(self):
        """7-tuple of MLX arrays for backward-compatible state roundtrip.

        Residual fields (indices 2-4) are always None -- the production path
        uses top-k sparse residuals stored inside KVCompressor, not the legacy
        sign-sketch arrays.
        """
        impl = self._impl
        if impl._k_packed is None:
            return (None, None, None, None, None, None, None)

        T = impl.offset

        def _crop(a):
            if a is None:
                return None
            return a[:, :, :T, ...] if a.shape[2] > T else a

        return (
            _crop(impl._k_packed),
            _crop(impl._k_scales),
            None,  # k_resid_scale_q   (sign-sketch; not used in production)
            None,  # k_resid_scale_max
            None,  # k_resid_proj_signs
            _crop(impl._v_packed),
            _crop(impl._v_scales),
        )

    @state.setter
    def state(self, v):
        k_codes, k_scales, _rsq, _rsmax, _rssigns, v_codes, v_scales = v
        impl = self._impl
        impl._k_packed = k_codes
        impl._k_scales = k_scales
        impl._v_packed = v_codes
        impl._v_scales = v_scales
        if k_codes is not None:
            impl.offset = k_codes.shape[2]
            impl._cap = k_codes.shape[2]
            impl._B = k_codes.shape[0]
            impl._H = k_codes.shape[1]
        else:
            impl.offset = 0

    @property
    def meta_state(self):
        """18-tuple of strings for backward-compatible state roundtrip.

        The loader also accepts the older 17-field tuple that predates
        ``residual_topk``.
        """
        impl = self._impl
        pipeline = impl.pipeline
        d_head = getattr(pipeline, "_d_head", None)
        d_pad = getattr(pipeline, "_d_pad", None)
        v_dim = getattr(pipeline, "_v_dim", None)
        v_pad = getattr(pipeline, "_v_pad", None)
        cfg = self.config

        if d_head is None:
            return ("",) * 18

        return (
            str(impl.offset),
            str(d_head),
            str(d_pad if d_pad is not None else ""),
            str(v_dim if v_dim is not None else ""),
            str(v_pad if v_pad is not None else ""),
            str(getattr(impl, "_dtype", None) or ""),
            str(cfg.main_bits),
            str(cfg.group_size),
            cfg.rotation,
            cfg.return_mode,
            cfg.scale_dtype,
            str(cfg.resid_scale_bits),
            str(cfg.residual_topk),
            str(cfg.v_bits),
            str(cfg.v_group_size),
            cfg.v_scale_dtype,
            "1" if cfg.v_enabled else "0",
            str(cfg.block_tokens),
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) == 17:
            (
                offset,
                d_head,
                d_pad,
                value_dim,
                v_pad,
                dtype_name,
                main_bits,
                group_size,
                rotation,
                return_mode,
                scale_dtype,
                resid_scale_bits,
                v_bits,
                v_group_size,
                v_scale_dtype,
                v_enabled,
                block_tokens,
            ) = v
            residual_topk = "2"
        elif len(v) == 18:
            (
                offset,
                d_head,
                d_pad,
                value_dim,
                v_pad,
                dtype_name,
                main_bits,
                group_size,
                rotation,
                return_mode,
                scale_dtype,
                resid_scale_bits,
                residual_topk,
                v_bits,
                v_group_size,
                v_scale_dtype,
                v_enabled,
                block_tokens,
            ) = v
        else:
            raise ValueError(
                f"Unexpected TurboQuant meta_state length {len(v)}; expected 17 or 18"
            )
        impl = self._impl
        pipeline = impl.pipeline
        pipeline._d_head = int(d_head) if d_head else None
        pipeline._d_pad = int(d_pad) if d_pad else None
        pipeline._v_dim = int(value_dim) if value_dim else None
        pipeline._v_pad = int(v_pad) if v_pad else None
        impl.offset = int(offset) if offset else 0
        impl._dtype = dtype_name or None

        self.config = TurboQuantConfig(
            main_bits=int(main_bits) if main_bits else 3,
            group_size=int(group_size) if group_size else 64,
            rotation=rotation or "identity",
            return_mode="view",
            scale_dtype=scale_dtype or "float16",
            resid_scale_bits=int(resid_scale_bits) if resid_scale_bits else 8,
            residual_topk=int(residual_topk) if residual_topk else 2,
            v_bits=int(v_bits) if v_bits else 4,
            v_group_size=int(v_group_size) if v_group_size else 64,
            v_scale_dtype=v_scale_dtype or "float16",
            v_enabled=(v_enabled == "1"),
            block_tokens=int(block_tokens) if block_tokens else 256,
        )
        self._return_mode = "view"

    @classmethod
    def from_state(cls, state, meta_state):
        """Restore from (state, meta_state) -- 2-arg legacy classmethod."""
        if len(meta_state) == 17:
            (
                offset,
                d_head,
                d_pad,
                value_dim,
                v_pad,
                dtype_name,
                main_bits,
                group_size,
                rotation,
                return_mode,
                scale_dtype,
                resid_scale_bits,
                v_bits,
                v_group_size,
                v_scale_dtype,
                v_enabled,
                block_tokens,
            ) = meta_state
            residual_topk = "2"
        elif len(meta_state) == 18:
            (
                offset,
                d_head,
                d_pad,
                value_dim,
                v_pad,
                dtype_name,
                main_bits,
                group_size,
                rotation,
                return_mode,
                scale_dtype,
                resid_scale_bits,
                residual_topk,
                v_bits,
                v_group_size,
                v_scale_dtype,
                v_enabled,
                block_tokens,
            ) = meta_state
        else:
            raise ValueError(
                f"Unexpected TurboQuant meta_state length {len(meta_state)}; expected 17 or 18"
            )

        cfg = TurboQuantConfig(
            main_bits=int(main_bits) if main_bits else 3,
            group_size=int(group_size) if group_size else 64,
            rotation=rotation or "identity",
            return_mode="view",
            scale_dtype=scale_dtype or "float16",
            resid_scale_bits=int(resid_scale_bits) if resid_scale_bits else 8,
            residual_topk=int(residual_topk) if residual_topk else 2,
            v_bits=int(v_bits) if v_bits else 4,
            v_group_size=int(v_group_size) if v_group_size else 64,
            v_scale_dtype=v_scale_dtype or "float16",
            v_enabled=(v_enabled == "1"),
            block_tokens=int(block_tokens) if block_tokens else 256,
        )
        obj = cls(cfg)
        impl = obj._impl
        pipeline = impl.pipeline
        pipeline._d_head = int(d_head) if d_head else None
        pipeline._d_pad = int(d_pad) if d_pad else None
        pipeline._v_dim = int(value_dim) if value_dim else None
        pipeline._v_pad = int(v_pad) if v_pad else None
        impl._dtype = dtype_name or None
        obj.state = state
        return obj

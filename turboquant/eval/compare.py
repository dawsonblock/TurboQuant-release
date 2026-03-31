"""
turboquant.eval.compare — accuracy comparison between dense and compressed inference.

This module provides the tools to measure, quantify, and report the accuracy
degradation introduced by KV-cache compression.  The comparison is always
between an identical prompt run twice:

1. **Baseline** — dense ``KVCache`` (no compression)
2. **Compressed** — ``TurboQuantKCache`` (compression active from token 0)

Metrics computed
----------------
* **KL divergence** — per-token KL( softmax(p_dense) || softmax(p_tq) ).
  High KL indicates the model's token probability distribution has shifted.
* **Token match rate** — fraction of positions where both runs agree on the
  argmax (greedy) token.  A drop below 0.95 is a strong signal of
  meaningful degradation.

Usage
-----
::

    from turboquant.eval.compare import AccuracyComparison

    comp = AccuracyComparison(model, tokenizer, config)
    report = comp.run(prompt="The quick brown fox", max_tokens=64)

    print(report)
    # {
    #   "mean_kl": 0.004,
    #   "max_kl": 0.021,
    #   "token_match_rate": 0.984,
    #   "n_tokens": 64,
    #   "kl_bound_ok": True,    # mean_kl <= threshold
    #   "match_bound_ok": True, # token_match_rate >= threshold
    # }

Thresholds
----------
Default thresholds (from docs/evaluation.md):
  * ``mean_kl``          ≤ 0.1 → ``kl_bound_ok``
  * ``token_match_rate`` ≥ 0.95 → ``match_bound_ok``

Both thresholds are configurable at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("turboquant.eval.compare")

# Default thresholds from docs/evaluation.md
DEFAULT_MAX_MEAN_KL: float = 0.1
DEFAULT_MIN_MATCH_RATE: float = 0.95


@dataclass
class ComparisonReport:
    """Structured result of a dense-vs-compressed accuracy comparison.

    Attributes
    ----------
    mean_kl:
        Mean KL divergence per token across the full generated sequence.
    max_kl:
        Maximum single-token KL divergence observed.
    token_match_rate:
        Fraction of positions where greedy argmax agrees between the two
        runs.  Range [0.0, 1.0].
    n_tokens:
        Number of tokens compared (excluding the prompt prefix).
    kl_bound_ok:
        ``True`` iff ``mean_kl <= max_mean_kl_threshold``.
    match_bound_ok:
        ``True`` iff ``token_match_rate >= min_match_rate_threshold``.
    model:
        Model family string from the config (e.g. ``"llama"``).
    k_bits:
        Key bit width used in the compressed run.
    v_bits:
        Value bit width used in the compressed run.
    divergence_statement:
        Human-readable summary: "compression causes X.X% KL divergence …".
    """

    mean_kl: float
    max_kl: float
    token_match_rate: float
    n_tokens: int
    kl_bound_ok: bool
    match_bound_ok: bool
    model: str = "unknown"
    k_bits: int = 3
    v_bits: int = 4
    divergence_statement: str = ""

    def __post_init__(self) -> None:
        if not self.divergence_statement:
            pct = round(self.token_match_rate * 100, 1)
            self.divergence_statement = (
                f"Compression causes mean KL={self.mean_kl:.4f} per token "
                f"with {pct}% greedy-token agreement on this model "
                f"({self.k_bits}-bit K, {self.v_bits}-bit V, "
                f"{self.n_tokens} tokens)."
            )

    def to_dict(self) -> dict:
        return {
            "mean_kl": self.mean_kl,
            "max_kl": self.max_kl,
            "token_match_rate": self.token_match_rate,
            "n_tokens": self.n_tokens,
            "kl_bound_ok": self.kl_bound_ok,
            "match_bound_ok": self.match_bound_ok,
            "model": self.model,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "divergence_statement": self.divergence_statement,
        }

    @property
    def passed(self) -> bool:
        """Both bounds satisfied — system is within certified accuracy limits."""
        return self.kl_bound_ok and self.match_bound_ok


class AccuracyComparison:
    """Run identical prompts under dense and compressed caches and compare.

    Parameters
    ----------
    model:
        An mlx-lm model supporting ``model(inputs, cache=...)`` API.
    tokenizer:
        A tokenizer with an ``encode(text) → list[int]`` method.
    config:
        :class:`~turboquant.config.TurboQuantConfig` for the compressed run.
    model_family:
        Model family string passed to Gate 2.
    max_mean_kl_threshold:
        Upper bound on mean KL divergence for ``kl_bound_ok``.
    min_match_rate_threshold:
        Lower bound on token match rate for ``match_bound_ok``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Any,
        model_family: str = "unknown",
        max_mean_kl_threshold: float = DEFAULT_MAX_MEAN_KL,
        min_match_rate_threshold: float = DEFAULT_MIN_MATCH_RATE,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._model_family = model_family
        self._max_mean_kl = max_mean_kl_threshold
        self._min_match_rate = min_match_rate_threshold

    def run(
        self,
        prompt: str,
        max_tokens: int = 64,
    ) -> ComparisonReport:
        """Compare dense vs compressed logits for *prompt*.

        Parameters
        ----------
        prompt:
            Text prompt to encode and run through both paths.
        max_tokens:
            Number of autoregressive steps to generate and compare.

        Returns
        -------
        ComparisonReport
            Structured comparison result with accuracy bounds.

        Notes
        -----
        This method imports MLX lazily so the module is importable on
        non-Apple platforms (for documentation / type checking).
        """
        import mlx.core as mx

        from turboquant.eval.generation_drift import logit_kl_divergence

        ids = self._tokenizer.encode(prompt)
        input_ids = mx.array([ids], dtype=mx.int32)

        dense_logits = self._collect_logits_dense(input_ids, max_tokens)
        tq_logits = self._collect_logits_compressed(input_ids, max_tokens)

        n = min(len(dense_logits), len(tq_logits))
        if n == 0:
            logger.warning(
                "AccuracyComparison: no logits collected — "
                "returning zero-value report"
            )
            return self._zero_report(max_tokens)

        # Stack into [n, vocab]
        p = mx.stack(dense_logits[:n])
        q = mx.stack(tq_logits[:n])

        kl_per_token = logit_kl_divergence(p, q)  # [n]
        mean_kl = float(mx.mean(kl_per_token).item())
        max_kl = float(mx.max(kl_per_token).item())

        # Token match rate — greedy argmax comparison
        p_tokens = mx.argmax(p, axis=-1)  # [n]
        q_tokens = mx.argmax(q, axis=-1)  # [n]
        match_rate = float(mx.mean((p_tokens == q_tokens).astype(mx.float32)).item())

        logger.info(
            "AccuracyComparison: n=%d mean_kl=%.4f max_kl=%.4f "
            "token_match=%.3f",
            n,
            mean_kl,
            max_kl,
            match_rate,
        )

        return ComparisonReport(
            mean_kl=round(mean_kl, 6),
            max_kl=round(max_kl, 6),
            token_match_rate=round(match_rate, 6),
            n_tokens=n,
            kl_bound_ok=mean_kl <= self._max_mean_kl,
            match_bound_ok=match_rate >= self._min_match_rate,
            model=self._model_family,
            k_bits=getattr(self._config, "k_bits", 3),
            v_bits=getattr(self._config, "v_bits", 4),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_logits_dense(self, input_ids, max_tokens: int) -> list:
        """Run the model with a standard dense KVCache; return logits list."""
        import mlx.core as mx

        from mlx_lm.models.cache import KVCache

        logits_list: list = []
        cache = [KVCache() for _ in range(len(self._model.layers))]

        x = input_ids
        for _ in range(max_tokens):
            logits = self._model(x, cache=cache)
            if logits.ndim == 3:
                logits = logits[:, -1, :]  # [1, vocab] → last token
            logits_list.append(logits[0])  # [vocab]
            next_tok = mx.argmax(logits, axis=-1)
            x = next_tok[:, None] if next_tok.ndim == 1 else next_tok

        return logits_list

    def _collect_logits_compressed(self, input_ids, max_tokens: int) -> list:
        """Run the model with TurboQuantKCache; return logits list."""
        import mlx.core as mx

        from integrations.mlx.cache_adapter import TurboQuantConfig, TurboQuantKCache

        logits_list: list = []

        # Build a per-layer legacy config from the production config
        legacy_cfg = TurboQuantConfig(
            main_bits=getattr(self._config, "k_bits", 3),
            group_size=getattr(self._config, "k_group_size", 64),
            rotation=getattr(self._config, "rotation", "hadamard"),
            v_bits=getattr(self._config, "v_bits", 4),
            v_group_size=getattr(self._config, "v_group_size", 64),
            v_enabled=getattr(self._config, "v_enabled", True),
            residual_topk=getattr(self._config, "residual_topk", 2),
            block_tokens=getattr(self._config, "block_tokens", 256),
        )
        cache = [
            TurboQuantKCache(legacy_cfg)
            for _ in range(len(self._model.layers))
        ]

        x = input_ids
        for _ in range(max_tokens):
            logits = self._model(x, cache=cache)
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            logits_list.append(logits[0])
            next_tok = mx.argmax(logits, axis=-1)
            x = next_tok[:, None] if next_tok.ndim == 1 else next_tok

        return logits_list

    def _zero_report(self, n_tokens: int) -> ComparisonReport:
        return ComparisonReport(
            mean_kl=0.0,
            max_kl=0.0,
            token_match_rate=0.0,
            n_tokens=n_tokens,
            kl_bound_ok=False,
            match_bound_ok=False,
            model=self._model_family,
            k_bits=getattr(self._config, "k_bits", 3),
            v_bits=getattr(self._config, "v_bits", 4),
            divergence_statement="No logits collected — comparison failed.",
        )

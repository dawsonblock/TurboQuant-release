"""turboquant.eval — evaluation utilities for TurboQuant KV compression.

Modules
-------
perplexity      : token-level perplexity with dense vs turboquant caches
generation_drift: KL-divergence between dense and turboquant logit distributions
memory          : peak-memory profiling helpers
"""

from .generation_drift import drift_report, logit_kl_divergence
from .memory import memory_report, peak_memory_bytes
from .perplexity import perplexity_from_logits, perplexity_report

__all__ = [
    "perplexity_from_logits",
    "perplexity_report",
    "logit_kl_divergence",
    "drift_report",
    "peak_memory_bytes",
    "memory_report",
]

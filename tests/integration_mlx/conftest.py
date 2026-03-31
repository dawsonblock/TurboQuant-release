"""
conftest.py — shared fixtures for the Apple-Silicon MLX integration tests.

Enforcement rule
----------------
The entire ``integration_mlx`` suite is **skipped** when any of the
following are true:

    1. Platform is not macOS arm64 (Apple Silicon).
    2. ``mlx`` is not importable.
    3. The ``TQ_TEST_LLAMA_MODEL`` and ``TQ_TEST_GEMMA_MODEL`` environment
       variables are not set (model fixtures skip individually).

No integration test file should repeat platform checks — this conftest
handles the boundary once.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Platform / MLX gate
# ---------------------------------------------------------------------------

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

try:
    import mlx.core as mx  # noqa: F401

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def _skip_reason() -> str | None:
    if not _IS_APPLE_SILICON:
        return "integration_mlx tests require Apple Silicon (darwin-arm64)"
    if not _HAS_MLX:
        return "integration_mlx tests require the `mlx` package"
    return None


_SKIP_REASON = _skip_reason()

# Auto-skip every test collected in this directory
pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=_SKIP_REASON or "",
)


# Register a custom marker so ``-m mlx_integration`` works
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mlx_integration: marks tests that need Apple Silicon + MLX",
    )


# ---------------------------------------------------------------------------
# Model ID fixtures (env-var gated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llama_model_id() -> str:
    """Return the Llama-family model ID from ``TQ_TEST_LLAMA_MODEL``."""
    mid = os.environ.get("TQ_TEST_LLAMA_MODEL", "")
    if not mid:
        pytest.skip(
            "TQ_TEST_LLAMA_MODEL not set — provide a small Llama-family HF model ID"
        )
    return mid


@pytest.fixture(scope="session")
def gemma_model_id() -> str:
    """Return the Gemma-family model ID from ``TQ_TEST_GEMMA_MODEL``."""
    mid = os.environ.get("TQ_TEST_GEMMA_MODEL", "")
    if not mid:
        pytest.skip(
            "TQ_TEST_GEMMA_MODEL not set — provide a small Gemma-family HF model ID"
        )
    return mid


# ---------------------------------------------------------------------------
# Shared decode settings
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def decode_settings() -> dict:
    """Fixed deterministic decode settings used across all runtime tests."""
    return {
        "max_tokens": 32,
        "temp": 0.0,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# TurboQuant config fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def default_tq_config():
    """Return a default ``TurboQuantConfig`` for certification tests."""
    from turboquant.config import TurboQuantConfig

    return TurboQuantConfig(
        k_bits=3,
        k_group_size=64,
        v_bits=4,
        v_group_size=64,
        v_enabled=True,
        rotation="hadamard",
        rotation_seed=42,
        residual_topk=2,
        block_tokens=256,
        scale_dtype="float16",
        v_scale_dtype="float16",
    )


# ---------------------------------------------------------------------------
# Prompt fixtures
# ---------------------------------------------------------------------------

SHORT_PROMPT = "Explain what a KV cache is in one sentence."

MEDIUM_PROMPT = (
    "A transformer model uses a key-value cache during autoregressive "
    "decoding to avoid recomputing attention over previously generated "
    "tokens. Describe three techniques for reducing the memory footprint "
    "of such a cache, including quantization-based approaches."
)

LONG_PROMPT = (
    "Below is a technical passage about attention mechanisms in large "
    "language models.\n\n"
    + (
        "The transformer architecture relies on multi-head self-attention, "
        "where each head independently computes scaled dot-product attention "
        "over queries, keys, and values derived from the input sequence. "
        "During autoregressive generation, the key and value tensors from "
        "all previous tokens must be retained in memory, forming the KV "
        "cache. As the sequence length grows, this cache becomes the "
        "dominant consumer of GPU or accelerator memory, often exceeding "
        "the memory used by the model weights themselves. "
    )
    * 8
    + "\n\nBased on the above, what is the primary benefit of compressing "
    "the KV cache during long-context generation?"
)


@pytest.fixture
def short_prompt() -> str:
    return SHORT_PROMPT


@pytest.fixture
def medium_prompt() -> str:
    return MEDIUM_PROMPT


@pytest.fixture
def long_prompt() -> str:
    return LONG_PROMPT


# ---------------------------------------------------------------------------
# Artifact directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a session-scoped temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("integration_mlx_artifacts")


# ---------------------------------------------------------------------------
# Model loading helpers (session-scoped to avoid re-downloading)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llama_model_and_tokenizer(llama_model_id: str):
    """Load the Llama-family model and tokenizer once per session."""
    from mlx_lm import load

    result = load(llama_model_id)
    model, tokenizer = result[0], result[1]
    return model, tokenizer


@pytest.fixture(scope="session")
def gemma_model_and_tokenizer(gemma_model_id: str):
    """Load the Gemma-family model and tokenizer once per session."""
    from mlx_lm import load

    result = load(gemma_model_id)
    model, tokenizer = result[0], result[1]
    return model, tokenizer

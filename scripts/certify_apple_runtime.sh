#!/usr/bin/env bash
# scripts/certify_apple_runtime.sh — one-command Apple-Silicon runtime certification.
#
# Usage:
#     ./scripts/certify_apple_runtime.sh
#
# Environment variables (required for model smoke tests):
#     TQ_TEST_LLAMA_MODEL   — small Llama-family HF model ID
#     TQ_TEST_GEMMA_MODEL   — small Gemma-family HF model ID
#
# Artifacts are written to: artifacts/runtime-cert/<timestamp>/
# Exit code is 0 only if every stage passes.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Timestamp and artifact directory
# ---------------------------------------------------------------------------
TS="$(date -u '+%Y%m%d_%H%M%S')"
ARTIFACT_DIR="$REPO_ROOT/artifacts/runtime-cert/$TS"
mkdir -p "$ARTIFACT_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo "  TurboQuant Apple-Silicon Runtime Certification"
echo "  Timestamp : $TS"
echo "  Artifacts : $ARTIFACT_DIR"
echo "═══════════════════════════════════════════════════════════════"

FAILURES=0

# Helper: run a stage, capture exit code, log result
run_stage() {
    local name="$1"; shift
    echo ""
    echo "──── Stage: $name ────"
    if "$@" ; then
        echo "  ✓ $name PASSED"
    else
        echo "  ✗ $name FAILED"
        FAILURES=$((FAILURES + 1))
    fi
}

# ---------------------------------------------------------------------------
# Stage 0: Create / activate venv (optional — skip if already in one)
# ---------------------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    VENV_DIR="$REPO_ROOT/.venv-cert"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating certification venv at $VENV_DIR ..."
        python3.11 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    echo "Installing package with Apple extras ..."
    pip install --quiet -e '.[apple,test]'
fi

# ---------------------------------------------------------------------------
# Stage 1: Strict preflight
# ---------------------------------------------------------------------------
run_stage "Strict Preflight" \
    python3.11 scripts/preflight.py --strict --json

python3.11 scripts/preflight.py --strict --json > "$ARTIFACT_DIR/preflight.json" 2>&1 || true

# ---------------------------------------------------------------------------
# Stage 2: Cache upgrade roundtrip
# ---------------------------------------------------------------------------
run_stage "Cache Upgrade Roundtrip" \
    python3.11 -m pytest -q --tb=short \
    tests/integration_mlx/test_cache_upgrade_roundtrip.py \
    --junitxml="$ARTIFACT_DIR/junit_cache_roundtrip.xml"

# ---------------------------------------------------------------------------
# Stage 3: Streaming attention equivalence
# ---------------------------------------------------------------------------
run_stage "Attention Equivalence" \
    python3.11 -m pytest -q --tb=short \
    tests/integration_mlx/test_streaming_attention_equivalence.py \
    --junitxml="$ARTIFACT_DIR/junit_attention_equiv.xml"

# ---------------------------------------------------------------------------
# Stage 4: Llama smoke test
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    run_stage "Llama Smoke" \
        python3.11 -m pytest -q --tb=short \
        tests/integration_mlx/test_llama_runtime_smoke.py \
        --junitxml="$ARTIFACT_DIR/junit_llama_smoke.xml"
else
    echo ""
    echo "──── Stage: Llama Smoke ────"
    echo "  SKIPPED (TQ_TEST_LLAMA_MODEL not set)"
fi

# ---------------------------------------------------------------------------
# Stage 5: Gemma smoke test
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
    run_stage "Gemma Smoke" \
        python3.11 -m pytest -q --tb=short \
        tests/integration_mlx/test_gemma_runtime_smoke.py \
        --junitxml="$ARTIFACT_DIR/junit_gemma_smoke.xml"
else
    echo ""
    echo "──── Stage: Gemma Smoke ────"
    echo "  SKIPPED (TQ_TEST_GEMMA_MODEL not set)"
fi

# ---------------------------------------------------------------------------
# Stage 5.5: Quality evaluation (perplexity + KL divergence)
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    for CLASS in short medium; do
        run_stage "Quality Eval $CLASS (Llama)" \
            python3.11 benchmarks/runtime_cert/run_quality_eval.py \
            --model "$TQ_TEST_LLAMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --max-delta-ppl 0.5 \
            --max-mean-kl 0.1 \
            --seed 42
    done
else
    echo ""
    echo "──── Stage: Quality Evaluation ────"
    echo "  SKIPPED (TQ_TEST_LLAMA_MODEL not set)"
fi

# ---------------------------------------------------------------------------
# Stage 6: Long-context stability
# ---------------------------------------------------------------------------
run_stage "Long-Context Stability" \
    python3.11 -m pytest -q --tb=short \
    tests/integration_mlx/test_long_context_stability.py \
    --junitxml="$ARTIFACT_DIR/junit_long_context.xml"

# ---------------------------------------------------------------------------
# Stage 7: Dense vs TurboQuant benchmark (requires model env vars)
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    for CLASS in short medium long; do
        run_stage "Benchmark $CLASS (Llama)" \
            python3.11 benchmarks/runtime_cert/run_dense_vs_tq.py \
            --model "$TQ_TEST_LLAMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --max-new-tokens 64 \
            --seed 42 \
            --mode both
    done
fi

if [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
    for CLASS in short medium long; do
        run_stage "Benchmark $CLASS (Gemma)" \
            python3.11 benchmarks/runtime_cert/run_dense_vs_tq.py \
            --model "$TQ_TEST_GEMMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --max-new-tokens 64 \
            --seed 42 \
            --mode both
    done
fi

# ---------------------------------------------------------------------------
# Stage 8: Aggregate metrics
# ---------------------------------------------------------------------------
if ls "$ARTIFACT_DIR"/*_dense.json "$ARTIFACT_DIR"/*_turboquant.json >/dev/null 2>&1; then
    run_stage "Metric Aggregation" \
        python3.11 benchmarks/runtime_cert/collect_metrics.py \
        --input-dir "$ARTIFACT_DIR" \
        --output-dir "$ARTIFACT_DIR"
else
    echo ""
    echo "──── Stage: Metric Aggregation ────"
    echo "  SKIPPED (no benchmark artifacts to aggregate)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Certification artifacts: $ARTIFACT_DIR"
echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "  ✓ ALL STAGES PASSED"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo "  ✗ $FAILURES STAGE(S) FAILED"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi

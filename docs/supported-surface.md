# Supported surface

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation for selected Llama-family and Gemma-family models. Custom Metal kernels are experimental and not part of the default supported runtime.

This repository does **not** claim broad `mlx_lm` model coverage. The codebase vendors a large upstream tree, but the TurboQuant-specific attention path is only wired and discussed for a narrow slice.

## Supported slice

What this repository currently intends to support:

- Apple Silicon Macs
- Python 3.9+
- MLX runtime installed locally
- Research and local evaluation workflows
- TurboQuant core package: `turboquant/*`
- `mlx_lm` adapter path used to upgrade dense prompt caches into `KVCompressor`
- Llama-family integration path
- Gemma-family integration path

## Model Support Matrix

| Model Architecture | Explicit Integration Tested | Support Status | Notes |
| :--- | :--- | :--- | :--- |
| Llama | Yes | **Supported** | Explicit memory/latency benchmarks validated local. |
| Gemma | Yes | **Supported** | Explicit memory/latency benchmarks validated local. |
| Mistral | No | Unsupported | Provided via upstream sync only. |
| Qwen | No | Unsupported | Provided via upstream sync only. |
| Phi | No | Unsupported | Provided via upstream sync only. |
| &lt;All Others&gt; | No | Unsupported | Uncertified. Vended for structural scaffolding. |

## Not claimed

What is **not** claimed by the current repository state:

- Public CI runtime certification of MLX-backed generation
- Production SLOs
- Broad compatibility across every model in the vendored `mlx_lm/models/` tree
- Fused Metal kernels for encoding or decoding (Metal kernel integration remains experimental and is not the default supported path)
- Large-scale perplexity validation
- Generic Linux or Windows runtime support

## Validation boundary

Two validation layers exist:

1. **Public static checks**
   - packaging metadata
   - source-tree integrity
   - syntax compilation
2. **Local Apple Silicon checks**
   - MLX install
   - unit and integration tests
   - model smoke runs
   - manual memory and latency comparison

Use `scripts/preflight.py` for the first layer and `scripts/validate_apple_silicon.sh` for the second.

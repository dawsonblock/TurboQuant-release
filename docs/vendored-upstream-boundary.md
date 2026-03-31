# Upstream mlx-lm Boundary

TurboQuant maintains a vendored slice of `mlx-lm` (`integrations/mlx_lm`) solely for end-to-end framework scaffolding. It is NOT an integration package representing full compatibility. 

## Supported Scope
- Custom Cache implementation injected for local evaluation.
- Attention logic modifications explicitly scoped to Llama and Gemma architectures executing locally on Apple Silicon via MLX Compiler paths.

## Unsupported
- Upstream changes to other model architectures. 
- Metal-specific generation tools (we strictly rely on the compiler).
- VLM support or multimodality.

## Maintenance
Changes to `mlx-lm` models other than Llama and Gemma are preserved strictly to maintain structural equivalence with upstream and are inherently uncertified by TurboQuant.

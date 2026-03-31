# turboquant/kernels
#
# Platform: Apple Silicon (MLX / Metal)
# Status:   vectorised/JIT MLX ops ARE the supported kernel layer.
#
# ── What this directory is for ─────────────────────────────────────────────
#
# On CUDA/Triton, custom kernels would live here. On Apple Silicon, the
# default supported runtime uses MLX-backed vectorized/JIT paths.
# handwritten Metal or raw shader paths are experimental only and no release
# claims depend on them. They exist in turboquant/experimental/kernels/metal
# but are NOT part of the supported path.
#
# The vectorised pack / unpack / quantise ops in turboquant/core/quantizer.py
# are dispatched as MLX ops — no hand-written shaders are part of the core
# package.
#
# ── Current hotspot latency (M-series, bs=1, 2-head Gemma) ─────────────────
#
#   Full dequant slice   0.48 ms / step   (vectorised MLX ops)
#   View (no dequant)    0.38 ms / step   (metadata + slicing)
#
# ── Future Metal shader candidates ─────────────────────────────────────────
#
# 1. Fused rotate+pack — single pass over K avoiding the intermediate buffer.
# 2. Fused unpack+dequant+residual-scatter — avoids two temporary tensors.
# 3. On-device topk scatter — replaces the broadcast-comparison trick in
#    turboquant/core/residual.py with a single Metal scatter_nd call.
#
# ── How to add a custom op ─────────────────────────────────────────────────
#
# MLX exposes `mx.fast.metal_kernel` (experimental, ≥ 0.8) for inlining a
# Metal shader string.  When that API stabilises, fused kernels should be
# wired in here and exposed via __init__.py.
#
# For now this package is intentionally empty except for this README.

__all__: list = []

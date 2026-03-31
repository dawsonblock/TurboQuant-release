# TurboQuant blocker-fix patch notes

This patch set addresses the two main runtime blockers and one restore-safety gap.

## Fixed

- **Non-power-of-two Hadamard bug**: `turboquant/core/rotation.py`
  now guarantees an orthogonal matrix for every dimension. Power-of-two dims
  still use exact normalized Hadamard. Other dims use a deterministic
  Hadamard-seeded QR construction instead of a cropped non-orthogonal matrix.

- **Residual-topk upgrade drift**: the canonical `mlx_lm` upgrade path now
  carries `residual_topk` through `generate.py -> cache_upgrade.py ->
  mlx_lm.models.cache.TurboQuantConfig -> turboquant.config.TurboQuantConfig`.
  The old hardcoded `residual_topk=0` path is removed.

- **Restore safety for calibrated quantizers**: `KVCompressor.state()` now
  serializes optional calibrated K/V scales plus the config fingerprint needed
  to fail closed on restore when the new config would diverge.

## Still legacy

- `turboquant_return_mode` remains adapter-only. The canonical upgrade path
  still uses streaming view mode in production attention.
- `turboquant_resid_scale_bits` remains adapter metadata only.
- `TurboQuantKCache.state/meta_state` still exposes the legacy wrapper format.
  This patch preserves backward compatibility there and extends meta-state to
  carry `residual_topk`, but it does not replace the wrapper format with the
  underlying `KVCompressor.state()` dict.

## Validation performed here

- Python syntax compilation across the patched tree.
- Structural grep checks to confirm the residual path is no longer hardcoded to 0.

Runtime MLX tests were not executed in this environment because `mlx` is not
installed here.

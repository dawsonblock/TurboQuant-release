import re, sys

turbo = "/Users/dawsonblock/Downloads/Turbo-master"
path = turbo + "/turboquant/runtime/kv_interface.py"
with open(path) as f:
    src = f.read()

pattern = re.compile(
    r"        # Encode K\n"
    r"        pk, ks, rv, ri = self\.pipeline\.encode_k\(keys\)\n"
    r"\n"
    r"        # NaN guard.*?poisons the cache\n"
    r"        if mx\.any\(mx\.isnan\(ks\)\)\.item\(\):.*?\n"
    r"            raise TurboQuantKernelError\(\n"
    r'                f"NaN detected in K scales after encode at offset \{prev\}.*?\n'
    r'                "This indicates.*?\n'
    r"            \)\n"
    r"\n"
    r"        # Store K\n"
    r"        self\._k_packed\[:, :, prev : prev \+ T, :\] = pk.*?\n"
    r"        self\._k_scales\[:, :, prev : prev \+ T, :\]"
    r" = ks\.astype\(self\._k_scales\.dtype\).*?\n"
    r"\n"
    r"        if self\.config\.residual_topk > 0 and rv is not None:\n"
    r"            self\._resid_vals\[:, :, prev : prev \+ T, :, :\] = rv.*?\n"
    r"            self\._resid_idx\[:, :, prev : prev \+ T, :, :\] = ri.*?\n"
    r"\n"
    r"        # Encode V\n"
    r"        if self\.config\.v_enabled:\n"
    r"            pv, vs = self\.pipeline\.encode_v\(values\)\n"
    r"            self\._v_packed\[:, :, prev : prev \+ T, :\] = pv.*?\n"
    r"            self\._v_scales\[:, :, prev : prev \+ T, :\] = vs\.astype\(.*?\n"
    r"                self\._v_scales\.dtype.*?\n"
    r"            \)\n"
    r"\n"
    r"        self\.offset \+= T",
    re.DOTALL,
)

replacement = (
    "        # Encode K \u2014 wrapped in fallback guard.  Any NaN or scale collapse\n"
    "        # raises CompressionFailureError so the caller can revert to dense.\n"
    "        try:\n"
    "            pk, ks, rv, ri = self.pipeline.encode_k(keys)\n"
    "\n"
    "            # NaN guard \u2014 detect corrupted encode before poisoning cache\n"
    "            if mx.any(mx.isnan(ks)).item():  # type: ignore[union-attr]\n"
    '                raise TurboQuantKernelError(\n'
    '                    f"NaN in K scales at offset {prev}\u2013{prev+T}."\n'
    "                )\n"
    "\n"
    "            # Store K\n"
    "            self._k_packed[:, :, prev : prev + T, :] = pk  # type: ignore\n"
    "            self._k_scales[:, :, prev : prev + T, :] = ks.astype(\n"
    "                self._k_scales.dtype  # type: ignore\n"
    "            )\n"
    "            if self.config.residual_topk > 0 and rv is not None:\n"
    "                self._resid_vals[\n"
    "                    :, :, prev : prev + T, :, :\n"
    "                ] = rv  # type: ignore\n"
    "                self._resid_idx[\n"
    "                    :, :, prev : prev + T, :, :\n"
    "                ] = ri  # type: ignore\n"
    "\n"
    "            # Encode V\n"
    "            if self.config.v_enabled:\n"
    "                pv, vs = self.pipeline.encode_v(values)\n"
    "                self._v_packed[\n"
    "                    :, :, prev : prev + T, :\n"
    "                ] = pv  # type: ignore\n"
    "                self._v_scales[\n"
    "                    :, :, prev : prev + T, :\n"
    "                ] = vs.astype(self._v_scales.dtype)  # type: ignore\n"
    "\n"
    "        except (TurboQuantKernelError, Exception) as exc:\n"
    "            # Restore offset so caller sees consistent state and can\n"
    "            # choose to fall back to the original dense cache.\n"
    "            self.offset = prev\n"
    "            from turboquant.errors import (  # noqa: PLC0415\n"
    "                CompressionFailureError,\n"
    "            )\n"
    "            raise CompressionFailureError(\n"
    '                f"KV compression failed at offset {prev}: {exc}"\n'
    "            ) from exc\n"
    "\n"
    "        self.offset += T"
)

m = pattern.search(src)
if m:
    src2 = src[:m.start()] + replacement + src[m.end():]
    with open(path, "w") as f:
        f.write(src2)
    print("REPLACED OK")
else:
    print("PATTERN NOT FOUND")
    idx = src.find("        # Encode K\n")
    print(f"Encode K at char {idx}")
    if idx >= 0:
        print(repr(src[idx:idx+500]))
    sys.exit(1)

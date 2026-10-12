# Qwen3.5 9B experiments

| Experiment | Decision / qualification |
| --- | --- |
| Prefill attention for 16 query heads | Retained: the WMMA prefill kernel is instantiated for 16 and 24 query heads (4 KV heads of 256). Before, 9B fell back to the scalar baseline, whose score buffer exceeds 64 KiB of LDS beyond 16,128 visible tokens and left stale attention output. The baseline now refuses such prompts before any state changes. The 24-head kernels keep byte-identical gfx1151 instructions and resources (only symbol names change); the 16-head instances use the same registers and LDS (253/228 VGPR, 24,832/20,992 B, no spills). GPU qualification pending. |
| Template rules per artifact | Retained: Qwen3.5 rendering (no effort levels, thinking only after the last query, inline `<think>` extraction, Python-form tool arguments) and the thinking default of each pinned template; 38 Hugging Face-equivalent goldens match. |
| 27B shape-specialized fast paths | Not extended: register-cached RMSNorm (5120), packed gate/up SwiGLU (17408 × 5120), joined DeltaNet alpha/beta (48 heads), wave64 small-batch groups, Q8_0 SSM-control grid, BF16 small-batch widths, split-K KV sharing for GQA 6 and FP16 low-bit prefill. 9B uses the generic routes; extending them needs gfx1151 measurements and requalification. |
| DFlash for 9B | Not supported: the only 9B draft is DFlash v1 (8 taps, mixed sliding/full attention), which the DFlash-2 loader rejects. |

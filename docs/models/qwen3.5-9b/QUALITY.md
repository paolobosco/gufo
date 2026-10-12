# Qwen3.5 9B quality

**Not yet qualified on gfx1151.** The checks below ran without a GPU: they
cover loading rules, prompt rendering, CPU reference paths and compilation of
the GPU code. GPU numerics, model quality and every execution-consistency
check remain open; a skipped GPU test is not a pass.

| Check | Status |
| --- | --- |
| `qwen35` metadata of the 9B shape (32 layers, 16/4 heads, DeltaNet 16/32), with and without the MTP layer and with `nextn` tensors but no `nextn_predict_layers` | CPU test passes |
| Chat rendering of the three pinned templates against Hugging Face `apply_chat_template` (38 cases: thinking defaults, empty/blank system, tools, tool loops, inline `<think>`, empty `reasoning_content`, typed tool arguments, images) | CPU test passes |
| Tool arguments from the request JSON rendered as Python `str()` / `json.dumps` | CPU test passes |
| Qwen3.8 prompts and cache-identity fields unchanged | CPU tests pass |
| MTP binding at `blk.{num_layers}` and quantized MTP oracle on synthetic GGUFs | CPU test passes |
| CPU dequantization of IQ4_NL, IQ4_XS and IQ3_S against the canonical routines | CPU test passes |
| 24-head WMMA prefill kernels before and after the 16-head instantiation | Identical gfx1151 instruction bytes and resources (same toolchain); only the kernel symbol names change |
| WMMA prefill attention, 16 query heads, FP64 oracle on KV-group boundary heads and log-sum-exp | Compiled; GPU run pending |
| Baseline attention refuses more than 16,128 visible tokens before writing KV | Compiled; GPU run pending |
| `qwen35_9b_target_test`: prefill versus decode across 16,128 tokens, chunk invariance, verification widths 2–8, C2–C8, CPU teacher forcing, 262,144-token cache | Compiled; GPU run pending |
| Agreement with an independent runtime (llama.cpp, same GGUF) | Pending |
| MTP drafting, image input | Pending; not declared as supported modes |

## Target test

`qwen35_9b_target_test [model.gguf [--prefill-only | --consistency-only | reference.gguf]]`
reads `GUFO_QWEN35_9B_MODEL` when no path is given and exits 77 without one.

- `--prefill-only`: prefill of (depth, tokens) (0, 64), (0, 1025),
  (15361, 2059) and (16384, 1025) against prefill of all but the last token
  plus one decode step, compared on hidden taps after layers 3, 7 and 31 and
  on the full logits. Chunk budgets 512 and 2048 at depth 15361 must give
  identical logits when every projection uses the native WMMA prefill formats.
- `--consistency-only`: verification widths 2–8, partial commit and replay,
  mixed FP16/FP32 KV batches and C2/C4/C6/C8 against scalar decode.
- Default: both, plus CPU teacher forcing (skipped when a tensor format has no
  CPU path) and a 262,144-token cache.
- A second GGUF measures KL, total variation and ΔNLL against it.

Skipped sub-checks are listed on the final line (`skipped (not passes)`).

Provisional limits, to be replaced by measured values and a negative control:
relative L2 0.05 / 0.08 / 0.15 for taps 3 / 7 / 31, KL 0.05 nats for
prefill versus decode and for CPU versus GPU. Prefill versus decode requires
the same top-1 token; CPU versus GPU accepts a different top-1 only when the
GPU choice lies within 0.1 logits of the CPU maximum.

## Reproduce

CPU checks:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --parallel 4
ctest --preset cpu-test -R 'qwen_chat_template_test|gguf_reader_test|openai_chat_test|qwen_mtp_weights_test|ggml_gemm_parity_test'
```

GPU checks on Strix Halo, with the GGUF of the
[usage guide](README.md#load-and-run):

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test
nix develop -c ctest --preset gpu-full \
  -R 'qwen_attention_policy_test|qwen_attention_wmma_prefill_ops_test|qwen_attention_decode_ops_test' \
  --output-on-failure
GUFO_QWEN35_9B_MODEL=models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  nix develop -c ctest --preset gpu-full -R '^qwen35_9b_target_test$' \
  --output-on-failure
```

Run the target test on Q8_0 and on UD-Q4_K_XL, and record the measured
metrics here.

Changes to the shared WMMA attention kernel must leave Qwen3.8 27B unchanged:
compare `qwen27b_target_test MODEL --prefill-only` fingerprints of the
previous and the new build, and run `tools/qwen27b/check.py kernels`.

## Open gaps

- Original-checkpoint and GGUF-conversion parity, including the DeltaNet
  value-head order for 32 value / 16 key heads.
- Long-context behavior with the native 262,144-token context.
- Template hashes of individual downloaded GGUF files (read here from
  repository metadata); `tools/qwen35_9b/inspect_gguf.py FILE` prints them
  with the thinking default, the layer counts and every tensor whose format
  the loader rejects, also for vision projectors.
- MTP: `qwen_mtp_gpu_test MTP.gguf MTP.gguf` with `unsloth/Qwen3.5-9B-MTP-GGUF`,
  acceptance and greedy MTP = AR.
- Vision: `qwen27b_vision_test MMPROJ IMAGE OUT 4096` with
  `tools/qwen27b/vision_reference.py`, and `qwen_vision_serving_test`.

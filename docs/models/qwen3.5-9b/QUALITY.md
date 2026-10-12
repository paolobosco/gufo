# Qwen3.5 9B quality

**Not yet qualified on gfx1151.** The checks below were run without a GPU:
they cover loading rules, prompt rendering and compilation. GPU numerics,
model quality and every execution-consistency check remain open.

| Check | Status |
| --- | --- |
| `qwen35` metadata of the 9B shape (32 layers, 16/4 heads, DeltaNet 16/32) passes structural validation | CPU test passes |
| Chat rendering of the three pinned templates against Hugging Face `apply_chat_template` (38 cases: thinking defaults, empty/blank system, tools, tool loops, inline `<think>`, empty `reasoning_content`, typed tool arguments, images) | CPU test passes |
| Python rendering of tool arguments from the request JSON (`str()` for scalars, `json.dumps` for containers) | CPU test passes |
| Qwen3.8 prompts and cache identity unchanged | CPU tests pass |
| WMMA prefill attention, 16 query heads, against an FP64 oracle | Test compiled; GPU run pending |
| Prefill versus decode logits beyond 16,128 tokens, verification and batched decode consistency | `qwen35_9b_target_test` compiled; GPU run pending |
| Agreement with an independent runtime (llama.cpp, same GGUF) | Pending |
| MTP drafting, image input | Pending; not declared as supported modes |

## Reproduce

CPU checks:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --parallel 4
ctest --preset cpu-test -R 'qwen_chat_template_test|gguf_reader_test|openai_chat_test'
```

GPU checks on Strix Halo, with the GGUF of the
[usage guide](README.md#load-and-run):

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test
nix develop -c ctest --preset gpu-full \
  -R 'qwen_attention_policy_test|qwen_attention_wmma_prefill_ops_test' \
  --output-on-failure
GUFO_QWEN35_9B_MODEL=models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  nix develop -c ctest --preset gpu-full -R '^qwen35_9b_target_test$' \
  --output-on-failure
```

Run the target test once on Q8_0 and once on UD-Q4_K_XL. The model-level
thresholds of the prefill/decode comparison are provisional until this first
run records the measured values here.

Changes to the shared WMMA attention kernel must also leave Qwen3.8 27B
unchanged: compare `qwen27b_target_test MODEL --prefill-only` fingerprints of
the previous and the new build, and run `tools/qwen27b/check.py kernels`.

## Open gaps

- Original-checkpoint and GGUF-conversion parity, including the DeltaNet
  value-head order for 32 value / 16 key heads.
- Long-context behavior with the native 262,144-token context.
- Template hashes of individual downloaded GGUF files (read here from
  repository metadata).
- MTP acceptance and greedy MTP = AR on `unsloth/Qwen3.5-9B-MTP-GGUF`.
- Vision embeddings against `tools/qwen27b/vision_reference.py` and
  `qwen_vision_serving_test` with the 9B sidecar.

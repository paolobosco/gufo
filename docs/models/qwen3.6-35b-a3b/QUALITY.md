# Qwen3.6-35B-A3B quality

**UD-Q6_K_XL qualification on gfx1151: in progress (October 2026).** The
checks below are the gate for this file. Results come from `release` and
`gpu-test` builds against the host ROCm in `/opt/rocm` (Linux 7.0); on that
host the Nix ROCm 7.2.3 runtime failed GPU allocations, the upstream
Flash-Next control included. The runtime was first
developed and compared with llama.cpp on a different quantization of this
model ([provenance](README.md#provenance)); those results do not carry over
and are not reported here. These checks compare implementations of one
quantized file; they do not qualify the quantization against the original
BF16 model.

## Artifact identity

| Item | Value |
| --- | --- |
| Repository | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` |
| Revision | Not recorded |
| File | `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` |
| File SHA-256 | `35fce994cd36104a7dc1bd8a4bdf13778145664c00fdef6773aebc9246e5019c` |
| Embedded chat template SHA-256 | `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`, the pinned Unsloth template (the official one is `e84f32a2`) |
| Layout | Expected `qwen35moe`, `block_count` 41, `nextn_predict_layers` 1, 753 tensors: Q8_0 303, Q6_K 80, F32 368, BF16 2 |

## Qualification checks

| Check | Criterion | Status |
| --- | --- | --- |
| Identity | `inventory` reproduces the expected layout; revision, file and template hashes recorded above | Passed: the layout matches; file and template hashes above; revision open |
| Chat template | Needed only for a template hash outside the pins. A Jinja harness with the `transformers` conventions first reproduces the six Qwen3.6 goldens from the official template (revision `995ad96e`); the embedded template must then give the same six render hashes before its hash is pinned. `chat_template_hf_token_golden_test` passes with `GUFO_QWEN36_GGUF` set; a skip (77) is a failure | Passed with the embedded template; rerun after the `preserve_thinking` default change open |
| Operators | `qwen36_35b_a3b.*` CPU and HIP tests pass, including routed Q6_K/Q8_0 MMQ invariance across tile widths, grid bounds and batch splits, and gate/up decode invariance at widths 1 to 32 | Passed: 11 of 11 `qwen36_35b_a3b.*` HIP tests |
| Full model | `session_test` (`--prefill-only`, `--batch-only`, then complete) and `snapshot_test` pass: greedy MTP equals AR token for token, C2 to C8 equal isolated execution, prefill chunks are bit-identical, seeded replay is exact | Passed: prefill-only, batch-only and complete session tests; snapshot round trip exact |
| Upstream agreement, llama.cpp `b11069` | 8 prompts × up to 64 teacher-forced positions, then recall prompts of about 7K and 15K tokens: tokenization matches for every prompt; top-1 at least 99%; mean KL at most 0.005; KL p99 at most 0.1 | Measured against a llama.cpp Vulkan build on the same file. Short prompts: tokenization 8 of 8, top-1 435 of 440 (98.9%; all 5 misses on the raw-text prompt), mean KL 0.0020, KL p99 0.040. Recall prompts of 6,958 and 14,944 tokens: top-1 128 of 128, mean KL 0.0015, KL p99 0.013. Short-prompt top-1 is below the 99% target; recorded as parity with the Vulkan reference |
| CLI | `gufo prompt` at temperature 0 prints the same `[TokenTrace]` for AR and MTP. `gufo bench --validate-prefill 2048` reports `finite=1` and winner rank 1; cosine, RMSE and maximum error are recorded here as measurements | Passed: AR and MTP `[TokenTrace]` identical; `--validate-prefill 2048`: finite=1, winner rank 1, cosine 0.994155, RMSE 0.52365, maximum error 2.49744 |
| Serving | `tests/functional/run.py --record-baseline` with preset `qwen36` and `--context 32768`, one server per suite that `all` selects; each prints `record-baseline passed` for AR and for MTP | AR and MTP: 35 of 38 suites and `cache` pass. Open: `tool-reasoning` (envelope case: the model writes the instruction text instead of the call), `tool-edges` (foreign-marker prose: malformed call opener) and `cache-concurrency` (6,858-token cold prompt: the model answers with the system prompt's label). All three are greedy model outputs with no cache restore involved |
| Shared paths | Flash-Next, Qwen3.8 27B and DeepSeek V4 Flash session and serving checks pass before and after on the same files; `gufo prompt` and `gufo chat` with `--speculative mtp` and no `--mtp-model` exit 2 before loading weights on those models; `gufo bench` and `gufo serve llm` keep their upstream exit status 1. A skip leaves the check open | Flash-Next UD-Q4_K_XL: `mtp_ops` passes; `tools`, `tool-streaming` and `tool-edges` pass. `tool-native-schemas` fails at `native_type_array_auto` (the generated call, rendered byte for byte the same, is not reused) identically with the upstream binary, so it predates this branch. Open: Qwen3.8 27B and DeepSeek not run. CLI exit statuses are covered by `prompt_cli_test` |
| Benchmarks | [Benchmarks](BENCHMARKS.md) cells measured | TODO |

Decode attention reads an 8-bit copy of the K/V cache (one F16 scale per 32
values); prefill keeps F16 K/V. Batched prefill and one-token decoding of the
same prefix therefore differ slightly, and `gufo bench --validate-prefill` can
exit 1 on the cosine gate tuned for Flash-Next while selecting the same top-1
token. The CLI check records those values instead of gating on them.

Q6_K routed gate/up projections take the MMQ route in prefill; no routed WMMA
decoder reads Q6_K. The operator tests above cover its batch and tile
invariance; the full-model checks cover the result.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration.

## Reproduce

Tests live in [`tests/models/qwen36_35b_a3b`](../../../tests/models/qwen36_35b_a3b).
`session_test` accepts `--batch-only`, `--prefill-only`, `--sampling-only` or
`--cache-only`.

```sh
# Identity; the CPU build is enough.
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --target qwen36_35b_a3b_inventory
build/cpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_inventory --model "$MODEL"
sha256sum "$MODEL"

# Operators and full model
nix develop -c cmake --build --preset gpu-test --target qwen36_35b_a3b_tests \
  qwen36_35b_a3b_model_tests qwen36_35b_a3b_gpu_probe chat_template_hf_token_golden_test
GUFO_QWEN36_GGUF="$MODEL" nix develop -c ctest --preset gpu-full \
  -R '^chat_template_hf_token_golden_test$' --output-on-failure
nix develop -c ctest --preset gpu-full -R '^qwen36_35b_a3b\.' --output-on-failure
build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_session_test --model "$MODEL"
build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_snapshot_test --model "$MODEL"

# llama.cpp agreement, against a running llama-server on the same file
python3 tests/models/qwen36_35b_a3b/llama_parity.py \
  --probe build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_gpu_probe \
  --model "$MODEL" --url http://127.0.0.1:8080
# Long prompts; llama-server needs -c 16384 or more
python3 tests/models/qwen36_35b_a3b/llama_parity.py \
  --probe build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_gpu_probe \
  --model "$MODEL" --url http://127.0.0.1:8080 --long 7000 --long 15000 --long-only

# CLI
./result/bin/gufo prompt -m "$MODEL" -p "$PROMPT" -n 256 --temperature 0 -v
./result/bin/gufo prompt -m "$MODEL" -p "$PROMPT" -n 256 --temperature 0 -v \
  --speculative mtp
./result/bin/gufo bench -m "$MODEL" --validate-prefill 2048

# Serving: one server per suite, since a long run fills the 128 RAM
# checkpoint records that unchanged retries need. Repeat with
# --speculative mtp and a new output directory.
SUITES="discovery messages-count-tokens messages-image-counts responses stops
  conversation structured structured-limits tool-reasoning tool-streaming
  reasoning-separator tools tool-edges tool-agent tool-agent-loop tool-history
  messages-tools tool-untyped tool-mixed tool-native-schemas tool-schema-edges
  sampling-defaults sampling-ranges batch long-context state-edges progress
  stream-start metrics cache-edits cache-growth cache-depth cache-rotation
  cache-concurrency cache-shared-prefix system-injection cache"
for suite in $SUITES; do
  nix develop -c python3 tests/functional/run.py --record-baseline \
    --output "$OUT/ar/$suite" --sampling-preset qwen36 --suite "$suite" \
    --expected-input-modalities text -- \
    ./result/bin/gufo serve llm --model "$MODEL" --sessions 4 --context 32768
done
```

## Benchmark method

Planned for every table: the production `nix build` binary and llama.cpp
`b11069` on the same file; one warmed sample per point, greedy, thinking off.
Single-user uses pp2048/tg128 over a cached prefix; MTP pp is the maximum
across mixed/repetitive workloads. C1/2/4/6/8 use the same d0 prompts; every
session prefills before measured tg128. Loading: cold model file (evicted with
`POSIX_FADV_DONTNEED`, zero resident pages verified), C1/MTP/capacity 262144.
Memory: C1/AR/capacity 133121, peak global HIP allocation including idle
memory. Workloads are declared in [bench.json](artifacts/bench.json); follow
the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md) and
record the Gufo binary SHA-256 prefix and the file identity with the results.

```sh
nix develop -c python3 tools/bench/model-bench.py --model qwen3.6-35b-a3b \
  --gguf "$MODEL" run --target gufo
nix develop -c nix shell .#llama-cpp-reference -c python3 \
  tools/bench/model-bench.py --model qwen3.6-35b-a3b --gguf "$MODEL" \
  run --target reference
nix develop -c python3 tools/bench/model-bench.py --model qwen3.6-35b-a3b render
```

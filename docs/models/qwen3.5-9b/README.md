# Qwen3.5 9B

Dense text model on gfx1151, served by the shared dense Qwen runtime
(`src/models/qwen`). Qwen3.5-9B uses the `qwen35` GGUF architecture of
Qwen3.8 27B at a smaller shape: hidden 4096, 32 layers (8 × three Gated
DeltaNet + one gated attention), FFN 12288, 16 query / 4 KV heads of 256,
DeltaNet 16 key / 32 value heads of 128, vocabulary 248,320, native context
262,144. Fine-tunes that keep this shape and the llama.cpp `qwen35` tensor
names load the same way.

Support is functional; quality and speed are not yet qualified on gfx1151.
See [Quality](QUALITY.md) for the checks to run and [Experiments](EXPERIMENTS.md)
for the shape-specific decisions.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --revision 3885219b6810b007914f3a7950a8d1b469d598a5 \
  --local-dir models/qwen3.5-9b
nix build
MODEL=models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --context 32768
```

Without `--context` the server allocates the native 262,144-token context.

| Mode | Selection | Status |
| --- | --- | --- |
| AR | No speculative option | Text; CLI and HTTP. |
| MTP | `--speculative mtp --mtp-model PATH` in `gufo prompt`/`chat` | Needs a GGUF with the MTP layer at `blk.32`, e.g. `unsloth/Qwen3.5-9B-MTP-GGUF`, passed as both `--model` and `--mtp-model`; MTP matrices may use any accepted format except F16. Not qualified. |
| DFlash2 | — | No DFlash-2 draft exists for 9B; `z-lab/Qwen3.5-9B-DFlash` is DFlash v1. |

Accepted tensor formats: BF16, Q8_0, Q3_K–Q6_K, IQ4_XS, IQ4_NL and IQ3_S
(Unsloth UD-Q4_K_XL, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16). F16, Q4_0/Q4_1, Q2_K
and IQ2/IQ1 artifacts are rejected at load.

## Chat template and thinking

The compiled formatter follows the Qwen3.5 template rules: no reasoning-effort
levels and no preserved thinking; reasoning of earlier turns is rendered only
after the last user query, and inline `<think>` in assistant content is split
out when `reasoning_content` is absent. `reasoning_effort` and
`preserve_thinking` controls are accepted and ignored, except that a non-off
top-level `reasoning_effort` enables thinking.

The thinking default (`--think auto`) comes from the artifact's template:

| Template (SHA-256) | Source | Thinking by default |
| --- | --- | --- |
| `7f0e5290…` | `unsloth/Qwen3.5-9B-GGUF` | off |
| `a4aee8af…` | `Qwen/Qwen3.5-9B` | on |
| `8452ca85…` | `unsloth/Qwen3.5-9B-MTP-GGUF` | on |

Other templates of non-Qwen3.8 artifacts (fine-tunes) use the same rules; their
thinking default is read from the template's generation branch. A fine-tune
template's own default system prompt or extra variables are not reproduced.
References: [src/models/qwen/reference](../../../src/models/qwen/reference/README.md).

The server samples greedily unless the request sets sampling. The model card
recommends temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 with
thinking, and temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5
without.

## Images

Gufo accepts a BF16 `qwen3vl_merger` sidecar with projection width 4096 and
`general.basename` `Qwen3.5-9B`, such as `mmproj-BF16.gguf` from
`unsloth/Qwen3.5-9B-GGUF`. It is discovered beside the target or in its parent
directory, or passed with `--mmproj PATH`. A different sidecar found
there fails the load; pass the matching one. The vision tower is the one of
Qwen3.8 27B. Image input is not qualified for 9B.

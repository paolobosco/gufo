# Qwen3.6-35B-A3B

Hybrid Gated DeltaNet / gated-attention mixture-of-experts text model on
gfx1151: 40 layers (30 DeltaNet, 10 full attention), 256 experts with 8 routed
plus one shared, about 3B active parameters, and a native one-layer MTP block.
Supported target: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, **UD-Q6_K_XL** (one file
with the MTP block). Text only. Support is in progress: the file identity,
qualification and measurements on gfx1151 are TODO.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Supported file

`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`, architecture `qwen35moe`: 41 blocks
(`block_count` 41, `nextn_predict_layers` 1; block 40 is the MTP block) and
753 tensors. Expected formats:

| Tensors | Format |
| --- | --- |
| Token embeddings, output head, attention, DeltaNet and shared-expert projections, `nextn.eh_proj` | Q8_0 |
| Routed `ffn_down_exps`, every block | Q8_0 |
| Routed `ffn_gate_exps` / `ffn_up_exps` | Q6_K; Q8_0 in block 39 |
| Norms, DeltaNet convolution, alpha/beta, `ssm_a`, `ssm_dt`; `ffn_gate_inp` and `ffn_gate_inp_shexp` of blocks 0 to 39 | F32 |
| `ffn_gate_inp` and `ffn_gate_inp_shexp` of block 40 | BF16 |

Totals: Q8_0 303, Q6_K 80, F32 368, BF16 2. This layout is inferred from a
published inventory and the file size; confirm it with the inventory tool
([Quality](QUALITY.md#artifact-identity)) before qualification. The loader
also accepts other `qwen35moe` files with Q4_K, Q5_K, Q6_K or Q8_0 experts
and Q8_0, K-quant, BF16, F16 or F32 dense weights, but only UD-Q6_K_XL is the
documented target. An optional
`qwen35moe.attention.recurrent_layers` key must agree with
`full_attention_interval` on every trunk block, or loading fails.

Prompts use Gufo's compiled Qwen3.6 template, so loading requires the GGUF's
embedded `tokenizer.chat_template` to have a pinned SHA-256: the official
Qwen/Qwen3.6-35B-A3B template (revision `995ad96e`) or Unsloth's variant.
Other templates are rejected with their hash. The UD-Q6_K_XL template hash is
TODO.

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  --revision TODO \
  --include "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf" \
  --local-dir models/qwen3.6-35b-a3b
nix build
MODEL=models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
./result/bin/gufo chat --model "$MODEL" --speculative mtp
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --sessions 2 --context 32768
```

The revision is not pinned yet: replace `TODO` with the commit recorded with
the file's SHA-256 in [Quality](QUALITY.md#artifact-identity). Do not
download an unpinned revision for measurements.

MTP uses the draft block stored in the same GGUF; there is no sidecar.
`--speculative mtp` rejects `--mtp-model` and requires `--min-draft-tokens 1`;
no other speculative mode is supported. `--mmproj` and images are rejected.
Omit `--speculative` for AR: AR sessions allocate no predictor state. Draft
length adapts to acceptance and is capped by `--draft-tokens` (1 to 7, default
7); drafts after the first are verified only while the draft probability is
at least 0.5. Greedy MTP output must equal AR token for token.

The official template defaults to thinking on and has no reasoning-effort
control. It drops reasoning from turns before the last user message unless
`preserve_thinking` is set. Gufo sets it by default, as the model card
recommends for agents and as Flash-Next does, so each new user turn extends
the cached previous generation. `--preserve-thinking off` or
`chat_template_kwargs.preserve_thinking=false` restores the template default;
reasoning inside the current tool cycle is kept either way. Sampling
defaults follow the model card (thinking: temperature 1.0, top-p 0.95; off:
0.7, 0.8; top-k 20 and presence penalty 1.5 in both). Native context is 262144;
YaRN extension is unsupported.

Memory, estimated from the file and the session layout (measurements are
TODO): weights take about 30.4 GiB, with no Q8_0 prefill copies because the
file has no K-quant dense matrix and no Q6_K routed down projection. Each full-attention layer
keeps an F16 K/V cache for prefill and an 8-bit copy for decode, about 31 KiB
per token over the ten trunk layers (34 KiB with the MTP block). Recurrent
state is about 63 MiB per session, about 147 MiB with MTP at seven draft
tokens (rollback copies). One 262144-token MTP session needs about 40 GiB;
admission reserves the configured capacity before creating sessions.

## Tools and artifacts

Model tests are in `tests/models/qwen36_35b_a3b`: CPU metadata, UD layout and
upload-policy checks, HIP operator tests against FP64 references, and the
full-model `session_test` / `snapshot_test` (MTP versus AR, batching,
rollback, prefill chunking, cache replay). `inventory` prints a file's
identity without reading weights: metadata, formats per role, template
SHA-256 and stop tokens. `reference_probe` runs the CPU oracle; `gpu_probe`
times single and multi-stream decoding and serves the llama.cpp comparison.
Retained result summaries belong in `artifacts/`; generated traces stay in the
ignored top-level `artifacts/` tree.

## Provenance

The model package, HIP kernels, serving integration and tests are adapted from
[NinjaPear/gufo-Qwen3.6-35B-A3B-Q6dense](https://github.com/NinjaPear/gufo-Qwen3.6-35B-A3B-Q6dense/tree/efcd90dd8a265857a5034ae818c052adba82ea26)
(MIT), which developed and compared them with llama.cpp on a different
quantization of this model (Q6_K dense weights, routed experts as
UD-Q4_K_XL). Its results do not carry over to UD-Q6_K_XL. See
[NOTICE](../../../NOTICE), [third-party notices](../../../THIRD_PARTY_NOTICES.md)
and the UD-Q6_K_XL changes in [Experiments](EXPERIMENTS.md).

# Qwen3.6-35B-A3B experiments

## UD-Q6_K_XL adaptations

Decisions for the UD-Q6_K_XL type mix. Their speed effect on gfx1151 is TODO;
exactness is gated by the checks in [Quality](QUALITY.md).

| Decision | Rationale / evidence |
| --- | --- |
| No Q8_0 prefill copies of routed gate/up experts | Retained: prefill swaps in the Q8_0 copy of a trunk Q6_K `ffn_down_exps` only; gate/up stacks and the MTP block never read one. UD-Q6_K_XL stores 80 Q6_K gate/up stacks and Q8_0 down projections, so copying every Q6_K expert stack would add about 21 GiB of unread memory and a long host dequantization at load; the file now gets no copies. Rule in `upload_policy.hpp`, checked for every role, format and block kind by `upload_policy_test`. |
| MMQ route for Q6_K gate/up in prefill | Retained until measured: the routed WMMA route decodes Q4_K/Q5_K gate/up only, so every UD block runs the vendored MMQ expert kernels (Q6_K; Q8_0 in block 39). `routed_wmma_ops_test` checks row-by-row invariance across tile widths, grid bounds and batch splits, and FP64 agreement. A Q6_K/Q8_0 WMMA route is a candidate once the prefill rate is measured. |
| `attention.recurrent_layers` validation | Retained: when the key is present it must hold one 0/1 entry per block and match `full_attention_interval` on every trunk block (the MTP entry is free); otherwise loading fails and names the key. A reader that follows the key must run the same layer schedule as these kernels. Covered by `config_test`. |
| Formats per role | Retained: `inventory` prints and `weights_layout_test` asserts the formats of every role in the 753-tensor UD layout, so a different file revision is caught before GPU work. |

## Original implementation

Decisions from the original implementation, developed on the Q6dense GGUF
(Q6_K dense weights, routed experts as UD-Q4_K_XL; see
[provenance](README.md#provenance)). They remain the design rationale.
Timings were taken on a shared gfx1151 host and are indicative; decisions
kept only changes that repeated across runs. Exactness gates (operator tests,
`session_test`, llama.cpp greedy parity) applied to every retained change.

| Experiment | Decision / evidence |
| --- | --- |
| MTP hidden-input normalization | Retained: the draft block's hidden input is the trunk row through `output_norm`, or its own carried residual through `nextn.shared_head_norm`, as in llama.cpp and vLLM. Raised draft acceptance to the llama.cpp level. |
| Draft confidence gating | Retained: drafts after the first are verified only while the draft probability is at least 0.5; about 1% faster decode, output unchanged. |
| Acceptance decay 0.95 | Retained for this model (Flash-Next keeps 0.75); length control follows this model's longer accepted runs. |
| Draft vocabulary | Retained 65,536 rows. 32K, 98K, 131K and the full vocabulary were each not better overall. |
| Draft attention splits | Retained: four times the key splits when one or two rows decode. |
| 8-bit K/V copy for decode attention | Retained: quantized after each cache write, read by the GQA decode kernel with 16-byte loads; prefill keeps F16. Fewer decode bytes at depth with unchanged judge scores. |
| Dense causal prefill attention | Retained: 16-key rounds, four heads per block, 21.9 versus 17.6 TFLOPS for the sparse-capable kernel it replaced. |
| Per-row diagonal PV in attention | Retained: on gfx1151 the WMMA PV sum depends on V rows whose probability is zero, so a block-wide choice made results depend on the prefill chunk boundary and the decode batch width. Each row now uses WMMA for keys it sees whole and sums its own diagonal tile alone. Exact across chunkings and widths 1-8. |
| F16 router and alpha/beta GEMMs | Retained: prefill uses a dedicated F16 GEMM (k=2048, m=64/257), decode a split-K F16 kernel with rounded intrinsics so every row width shares one arithmetic. |
| Q8_0 prefill copies | Retained for K-quant dense matrices and Q6_K routed down projections; decode reads the original rows. |
| Fused MoE epilogue and residual norm | Retained (8-wide, F32 and F16 expert rows); the output norm uses the same pass on every path. |
| Q6_K dense decode kernel | Retained: rows regrouped by block field at load, two rows per wave for 2-8 tokens, fused shared-expert gate/up/SwiGLU. |
| Grouped expert-down rows | Retained in the shared MMVQ for up to 64 single-expert rows of Q4_K/Q5_K/Q6_K; bit-identical per row. |
| Q5_K dense weights | Rejected: no speed gain over Q6_K; Q5 is the lowest precision considered. |
| bartowski Q4_K_M dense | Rejected: correct, but slower MTP decode (82 versus about 100 tok/s), lower acceptance and half the prefill rate. |
| Prefill chunk 4096/8192 | Rejected: not faster than 2048. |
| In-kernel MoE grouping, 4-pair grouped blocks, 64-token Q5_K down tiles | Rejected: slower. |
| Transposed int8 V, double-buffered K/V prefetch, 128 splits for every row | Rejected: no gain. |
| Fused F16 norm output | Rejected: slower than the F32 path. |

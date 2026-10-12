# Qwen3.5 9B benchmarks

No retained measurements yet. The runtime has not been timed on gfx1151 with
the 9B shapes. Every cell below is TODO.

| Workload | Gufo | llama.cpp | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL AR, pp2048 / tg128, depth 0 | TODO | TODO | TODO |
| UD-Q4_K_XL AR, depth 16,384 and 65,536 | TODO | TODO | TODO |
| Q8_0 AR, pp2048 / tg128, depth 0 | TODO | TODO | TODO |
| UD-Q4_K_XL AR, C2/C4/C8 aggregate tg | TODO | TODO | TODO |

Measure after the [quality checks](QUALITY.md#reproduce) pass, with the
[benchmark methodology](../../BENCHMARKS.md), the same GGUF on both engines and
the thinking mode set explicitly on both (`--think off` for Gufo,
`--reasoning off` for llama-server); the `unsloth/Qwen3.5-9B-GGUF` template
defaults to thinking off in Gufo, while the pinned llama-server enables it.

Expected shape effects, to confirm by measurement: 27B fast paths keyed on
hidden 5120 / FFN 17408 / 48 DeltaNet heads do not apply, so 9B uses the
generic GEMV/GEMM, RMSNorm, packed-FFN and SSM-control routes listed in
[Experiments](EXPERIMENTS.md).

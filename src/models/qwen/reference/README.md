# Qwen chat-template references

## Qwen3.8

The compiled formatter is pinned to `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The official `Qwen/Qwen3.8-Flash-Next` template at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` has the same SHA-256.
The two pinned `tokenizer.json` files are also byte-identical.
Both default to `enable_thinking=true`, `reasoning_effort="xhigh"`, and
`preserve_thinking=true`. Native efforts are `low`, `medium`, and `xhigh`;
medium adds no effort instruction. Explicit thinking-off suppresses the
effort instruction and closes the generation prompt's thinking block.

- Official `chat_template.jinja` SHA-256:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
- Repository copy with its terminating LF SHA-256:
  `514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37`
- Official `tokenizer.json` SHA-256:
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- Recognized Unsloth Qwen3.8 GGUF template SHA-256:
  `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`

`chat_template.jinja` is reference data only. Gufo validates recognized
artifact hashes and executes the bounded C++ implementation.

Full rendered-byte goldens cover tools before system instructions, the Unsloth
artifact's leading developer-message extension, tool-loop reasoning retention,
Unicode trimming around images and optional `Picture N:` numbering.
The reference rejects late system/developer messages; Gufo hoists them into
the leading system turn in their original order. Tool/system cases also compare
complete token sequences using the real GGUF tokenizer.

The tokenizer applies NFC normalization and the pinned Unicode split pattern,
including Unicode case-insensitive contractions, letter/mark runs and
trailing-whitespace backtracking. Ten raw-text
goldens check exact token IDs for spacing, code, accents, scripts with combining
marks, Unicode number categories and special-token boundaries on both models.

## Qwen3.6

`qwen36_chat_template.jinja` is the official `Qwen/Qwen3.6-35B-A3B`
`chat_template.jinja` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.
It defaults to `enable_thinking=true` and `preserve_thinking=false` and has no
reasoning-effort control. Gufo resolves an unset `preserve_thinking` to true
for Qwen3.6; an explicit false renders the template default. A leading system message renders a system turn even
when its content is empty. Assistant content holding inline
`<think>...</think>` without `reasoning_content` is split into reasoning and
reply.

- Official `chat_template.jinja` SHA-256:
  `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`
- Repository copy with its terminating LF SHA-256:
  `c726bb991c3150f3ee3d690930f43ed1e7c28dcef27cfbc6de3400db1da2a9a6`
- Official `tokenizer.json` SHA-256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- Recognized Unsloth Qwen3.6 GGUF template SHA-256:
  `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`

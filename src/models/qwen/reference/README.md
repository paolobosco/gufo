# Qwen3.8 chat-template reference

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
Late system/developer messages are rejected. Tool/system cases also compare
complete token sequences using the real GGUF tokenizer.

The tokenizer applies NFC normalization and the pinned Unicode split pattern,
including Unicode case-insensitive contractions, letter/mark runs and
trailing-whitespace backtracking. Ten raw-text
goldens check exact token IDs for spacing, code, accents, scripts with combining
marks, Unicode number categories and special-token boundaries on both models.

## Qwen3.5 chat-template references

The same compiled formatter renders Qwen3.5 artifacts with the Qwen3.5 rules
(`QwenTemplateDialect::kQwen35`). Three templates are pinned; each carries its
own thinking default and empty-system rule:

| File | Source | SHA-256 of the embedded template | Thinking by default |
| --- | --- | --- | --- |
| `qwen35_chat_template.jinja` | `Qwen/Qwen3.5-9B` `chat_template.jinja` (also 4B, 27B, 35B-A3B) | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | on |
| `qwen35_unsloth_gguf_chat_template.jinja` | `unsloth/Qwen3.5-9B-GGUF` revision `3885219b6810b007914f3a7950a8d1b469d598a5` | `7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67` | off |
| `qwen35_unsloth_chat_template.jinja` | `unsloth/Qwen3.5-9B` and `unsloth/Qwen3.5-9B-MTP-GGUF` | `8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8` | on |

The first two repository copies add a terminating LF (SHA-256 with it:
`d2cb9a5730cdd5f44bce3ada2dc1b0e00c6c59788b6d1c4d8d49c40a274dffb0` and
`ee1c4923e67f3f382fdf5e6b0f2156b236e2240f13b19731dbc4a2c82ca3d6dc`); the
Unsloth template already ends with one. GGUF template hashes were read from
Hugging Face repository metadata on October 12, 2026; confirm the per-file
`tokenizer.chat_template` of a downloaded artifact before relying on it.

`tools/qwen35_9b/chat_template_goldens.py` renders 38 cases from these files in
the Jinja environment of Transformers `apply_chat_template` and stores their
SHA-256 in `tests/fixtures/chat_template_hf_goldens.json` (`qwen35`).

Accepted differences from the templates, shared with the Qwen3.8 formatter:
leading system and developer messages are merged with `\n`, empty ones are
skipped, and a developer message is accepted by the official template's rules;
late system messages and transcripts without a user query are rejected.

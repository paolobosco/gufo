#!/usr/bin/env python3
"""Print the GGUF facts Gufo checks when loading a Qwen3.5 artifact.

Usage: inspect_gguf.py [--chat-template] FILE...

Accepts target models and vision projectors (mmproj). --chat-template prints
only each file's template text, with the terminating LF of the reference
copies in src/models/qwen/reference.
"""
import collections
import hashlib
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
         8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
         14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
         29: "IQ1_M", 30: "BF16"}
# Formats per tensor role, as in src/models/qwen/weights.cpp and the vision
# projector checks of src/models/qwen/vision/encoder.hip.
EMBEDDING = {"F32", "BF16", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL",
             "IQ4_XS", "IQ3_S"}
PROJECTION = EMBEDDING | {"Q8_K"}
F32_SUFFIXES = ("norm.weight", "ssm_a", "ssm_a.weight", "ssm_conv1d",
                "ssm_conv1d.weight", "ssm_dt", "ssm_dt.bias", "ssm_dt.weight")
VISION_BF16 = ("attn_qkv.weight", "attn_out.weight", "ffn_up.weight",
               "ffn_down.weight")
# Thinking defaults of the pinned Qwen3.5 templates (chat_template.cpp).
PINNED = {
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715":
        ("Qwen/Qwen3.5-9B", True),
    "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67":
        ("unsloth/Qwen3.5-9B-GGUF", False),
    "8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8":
        ("unsloth/Qwen3.5-9B-MTP-GGUF", True),
}
QWEN38 = {"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
          "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"}
THINKING_OFF = "enable_thinking is defined and enable_thinking is true"
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, handle):
        self.handle = handle

    def take(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.handle.read(size))[0]

    def string(self):
        return self.handle.read(self.take("<Q")).decode("utf-8", "replace")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            inner, count = self.take("<I"), self.take("<Q")
            if inner == 8:
                for _ in range(count):
                    self.string()
            else:
                self.handle.seek(struct.calcsize(SCALARS[inner]) * count, 1)
            return f"<array of {count}>"
        return self.take(SCALARS[kind])


def accepted_types(arch, name, mtp_first_layer):
    if arch == "clip":
        bf16 = name in ("mm.0.weight", "mm.2.weight") or (
            name.startswith("v.blk.") and name.endswith(VISION_BF16))
        return {"BF16"} if bf16 else {"F32"}
    parts = name.split(".")
    if parts[0] == "blk" and int(parts[1]) >= mtp_first_layer:
        return {"F32", "BF16"} if name.endswith("norm.weight") else PROJECTION
    if name == "token_embd.weight":
        return EMBEDDING
    if name.endswith(F32_SUFFIXES):
        return {"F32"}
    return PROJECTION


def describe_template(template):
    if not isinstance(template, str) or not template:
        return "missing: compiled Qwen3.5 template, thinking by default on"
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest in QWEN38:
        return "Qwen3.8 (pinned)"
    if digest in PINNED:
        source, thinking = PINNED[digest]
        state = "on" if thinking else "off"
        return f"pinned {source}, thinking by default {state}"
    state = "off" if THINKING_OFF in template else "on"
    return f"unpinned: Qwen3.5 rules, thinking by default {state}"


def main(path, show_template):
    with open(path, "rb") as handle:
        reader = Reader(handle)
        if handle.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        reader.take("<I")
        tensor_count, kv_count = reader.take("<Q"), reader.take("<Q")
        kv = {}
        for _ in range(kv_count):
            key = reader.string()
            kv[key] = reader.value(reader.take("<I"))
        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dims = [reader.take("<Q") for _ in range(reader.take("<I"))]
            kind = reader.take("<I")
            reader.take("<Q")
            tensors.append((name, TYPES.get(kind, f"T{kind}"), dims))
    arch = kv.get("general.architecture", "?")
    template = kv.get("tokenizer.chat_template")
    if show_template:
        print(template if isinstance(template, str) else "")
        return
    print(path)
    print("  general.name        ", kv.get("general.name"))
    print("  general.basename    ", kv.get("general.basename"))
    print("  architecture        ", arch)
    for key in ("block_count", "nextn_predict_layers", "embedding_length",
                "context_length"):
        if arch != "clip":
            print(f"  {arch}.{key:<21}", kv.get(f"{arch}.{key}"))
    if isinstance(template, str):
        print("  chat template sha256", hashlib.sha256(template.encode()).hexdigest())
    if arch == "qwen35":
        print("  chat template       ", describe_template(template))
    print("  projection_dim      ", kv.get("clip.vision.projection_dim"))
    counts = collections.Counter(kind for _, kind, _ in tensors)
    print("  tensor types        ", dict(sorted(counts.items())))
    if arch in ("qwen35", "clip"):
        layers = kv.get(f"{arch}.block_count") or 0
        mtp_first_layer = layers - (kv.get(f"{arch}.nextn_predict_layers") or 0)
        rejected = [f"{name}:{kind}" for name, kind, _ in tensors
                    if kind not in accepted_types(arch, name, mtp_first_layer)]
        print("  tensors Gufo rejects", len(rejected), rejected[:8])
    if arch == "clip":
        print("  tensor count        ", len(tensors), "(Gufo expects 334)")
    nextn = [(name, kind) for name, kind, _ in tensors if ".nextn." in name]
    if nextn:
        print("  MTP tensors         ", nextn)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    show = "--chat-template" in arguments
    for argument in arguments:
        if argument != "--chat-template":
            main(argument, show)

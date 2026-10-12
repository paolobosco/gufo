#!/usr/bin/env python3
"""Print the GGUF facts Gufo checks when loading a Qwen3.5 artifact."""
import collections
import hashlib
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
         8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
         14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
         29: "IQ1_M", 30: "BF16"}
ACCEPTED = {"F32", "BF16", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
            "IQ4_NL", "IQ4_XS", "IQ3_S"}
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


def main(path):
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
    print(path)
    print("  general.name        ", kv.get("general.name"))
    print("  general.basename    ", kv.get("general.basename"))
    print("  architecture        ", arch)
    for key in ("block_count", "nextn_predict_layers", "embedding_length",
                "context_length"):
        print(f"  {arch}.{key:<21}", kv.get(f"{arch}.{key}"))
    if isinstance(template, str):
        print("  chat template sha256", hashlib.sha256(template.encode()).hexdigest())
    print("  projection_dim      ", kv.get("clip.vision.projection_dim"))
    counts = collections.Counter(kind for _, kind, _ in tensors)
    print("  tensor types        ", dict(sorted(counts.items())))
    rejected = sorted(set(counts) - ACCEPTED)
    if arch == "qwen35":
        print("  formats Gufo rejects", rejected or "none")
    nextn = [(name, kind) for name, kind, _ in tensors if ".nextn." in name]
    if nextn:
        print("  MTP tensors         ", nextn)


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        main(argument)

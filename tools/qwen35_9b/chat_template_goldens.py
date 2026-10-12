#!/usr/bin/env python3
"""Render Qwen3.5 chat-template goldens for the compiled Gufo formatter.

The pinned templates under src/models/qwen/reference/ are rendered with the
same Jinja environment as Hugging Face transformers' apply_chat_template:
an immutable sandbox with trim_blocks, lstrip_blocks and loopcontrols, and a
tojson filter equal to json.dumps(ensure_ascii=False). Plain Jinja tojson sorts
keys and escapes HTML characters, so it is not used.

The case inputs and the SHA-256 of every rendered prompt are written to the
"qwen35" section of tests/fixtures/chat_template_hf_goldens.json, where
qwen_chat_template_test replays them.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "src/models/qwen/reference"
FIXTURE = ROOT / "tests/fixtures/chat_template_hf_goldens.json"

# Bytes embedded in the GGUF artifacts. The repository copies of the official
# and GGUF templates add a terminating LF; the unsloth template already ends
# with one.
TEMPLATES = {
    "official": {
        "file": "qwen35_chat_template.jinja",
        "strip_final_lf": True,
        "repo": "Qwen/Qwen3.5-9B",
        "template_sha256":
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    },
    "gguf": {
        "file": "qwen35_unsloth_gguf_chat_template.jinja",
        "strip_final_lf": True,
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "template_sha256":
            "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
    },
    "unsloth": {
        "file": "qwen35_unsloth_chat_template.jinja",
        "strip_final_lf": False,
        "repo": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "template_sha256":
            "8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8",
    },
}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Temperature <°C> & 'wind' in a city",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}

TYPED_ARGUMENTS = (
    '{"b": true, "n": null, "f": 1.0, "e": 1E5, "small": 1e-5, "big": 1e16,'
    ' "tiny": 0.0001, "i": 100000, "huge": 123456789012345678901234567890,'
    ' "neg0": -0, "negf": -0.0, "frac": 2.50, "obj": {"x": [1, true, null,'
    ' 2.50], "s": "<&>\' \\u00e9\\n"}, "arr": [], "str": "plain"}')

WEATHER_CALL = {"name": "get_weather", "arguments": '{"city": "Rome"}'}

# Messages use the OpenAI request shape that the server parses: tool-call
# arguments are JSON text and `reasoning_content` is present only when sent.
CASES = {
    "basic": (["official", "gguf", "unsloth"], {
        "messages": [{"role": "user", "content": "Name one color."}],
    }),
    "thinking_on": (["official", "gguf", "unsloth"], {
        "messages": [{"role": "user", "content": "Name one color."}],
        "enable_thinking": True,
    }),
    "thinking_off": (["official", "gguf", "unsloth"], {
        "messages": [{"role": "user", "content": "Name one color."}],
        "enable_thinking": False,
    }),
    "system": (["official", "gguf"], {
        "messages": [{"role": "system", "content": "  Be concise.\n"},
                     {"role": "user", "content": "Hi"}],
    }),
    "empty_system": (["official", "gguf", "unsloth"], {
        "messages": [{"role": "system", "content": ""},
                     {"role": "user", "content": "Hi"}],
    }),
    "blank_system": (["official", "unsloth"], {
        "messages": [{"role": "system", "content": " \n\t"},
                     {"role": "user", "content": "Hi"}],
    }),
    "empty_system_tools": (["official", "unsloth"], {
        "messages": [{"role": "system", "content": ""},
                     {"role": "user", "content": "Weather in Rome?"}],
        "tools": [WEATHER],
    }),
    "system_tools": (["official", "gguf", "unsloth"], {
        "messages": [{"role": "system", "content": "Be concise."},
                     {"role": "user", "content": "Weather in Rome?"}],
        "tools": [WEATHER],
    }),
    "developer": (["unsloth"], {
        "messages": [{"role": "developer", "content": "Use metric units."},
                     {"role": "user", "content": "Weather in Rome?"}],
        "tools": [WEATHER],
    }),
    "system_developer": (["unsloth"], {
        "messages": [{"role": "system", "content": "Be concise."},
                     {"role": "developer", "content": "Use metric units."},
                     {"role": "user", "content": "Weather in Rome?"}],
    }),
    "history": (["official", "gguf"], {
        "messages": [
            {"role": "user", "content": "Name one color."},
            {"role": "assistant", "content": "Red",
             "reasoning_content": "I should answer with one color."},
            {"role": "user", "content": "Name another."},
        ],
        "enable_thinking": True,
    }),
    "tool_loop": (["official", "gguf", "unsloth"], {
        "messages": [
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "content": "",
             "reasoning_content": "Use the tool.", "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "21 C"},
        ],
        "tools": [WEATHER],
        "enable_thinking": True,
    }),
    "parallel_tools": (["official"], {
        "messages": [
            {"role": "user", "content": "Weather in Rome and Oslo?"},
            {"role": "assistant", "content": "Checking both.",
             "reasoning_content": "",
             "tool_calls": [WEATHER_CALL,
                            {"name": "get_weather",
                             "arguments": '{"city": "Oslo", "days": 2}'}]},
            {"role": "tool", "content": "21 C"},
            {"role": "tool", "content": " 4 C\n"},
        ],
        "tools": [WEATHER],
    }),
    "inline_think": (["official", "gguf"], {
        "messages": [
            {"role": "user", "content": "First?"},
            {"role": "assistant", "content": "<think>\nold\n</think>\n\nA1"},
            {"role": "user", "content": "Second?"},
            {"role": "assistant", "content": "text <think>r</think>  ans",
             "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "21 C"},
        ],
        "enable_thinking": True,
    }),
    "reasoning_empty": (["official"], {
        "messages": [
            {"role": "user", "content": "Second?"},
            {"role": "assistant", "content": "text <think>r</think>  ans",
             "reasoning_content": "", "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "21 C"},
        ],
        "enable_thinking": True,
    }),
    "reasoning_newline": (["official"], {
        "messages": [
            {"role": "user", "content": "Second?"},
            {"role": "assistant", "content": "text <think>r</think>  ans",
             "reasoning_content": "\n", "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "content": "21 C"},
        ],
        "enable_thinking": True,
    }),
    "typed_arguments": (["official", "gguf", "unsloth"], {
        "messages": [
            {"role": "user", "content": "Call it."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "typed", "arguments": TYPED_ARGUMENTS}]},
            {"role": "user", "content": "Again."},
        ],
    }),
    "vision": (["official", "gguf"], {
        "messages": [{"role": "user", "content": "Describe.",
                      "image_offsets": [0]}],
    }),
}


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None,
           sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def load_template(spec):
    text = (REFERENCE / spec["file"]).read_text(encoding="utf-8")
    if spec["strip_final_lf"]:
        if not text.endswith("\n"):
            raise SystemExit(f"{spec['file']} must end with one LF")
        text = text[:-1]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != spec["template_sha256"]:
        raise SystemExit(f"{spec['file']} hashes to {digest}")
    return text


def hf_messages(messages):
    converted = []
    for message in messages:
        item = {"role": message["role"]}
        offsets = message.get("image_offsets")
        if offsets:
            parts, cursor = [], 0
            for offset in offsets:
                if offset > cursor:
                    parts.append({"type": "text",
                                  "text": message["content"][cursor:offset]})
                parts.append({"type": "image"})
                cursor = offset
            if cursor < len(message["content"]):
                parts.append({"type": "text",
                              "text": message["content"][cursor:]})
            item["content"] = parts
        else:
            item["content"] = message["content"]
        if "reasoning_content" in message:
            item["reasoning_content"] = message["reasoning_content"]
        if "tool_calls" in message:
            item["tool_calls"] = [{
                "type": "function",
                "function": {"name": call["name"],
                             "arguments": json.loads(call["arguments"])},
            } for call in message["tool_calls"]]
        converted.append(item)
    return converted


def render(template_text, case):
    kwargs = {"messages": hf_messages(case["messages"]),
              "add_generation_prompt": True}
    if "tools" in case:
        kwargs["tools"] = case["tools"]
    if "enable_thinking" in case:
        kwargs["enable_thinking"] = case["enable_thinking"]
    return environment().from_string(template_text).render(**kwargs)


def build():
    templates = {name: load_template(spec) for name, spec in TEMPLATES.items()}
    section = {
        "generator": "tools/qwen35_9b/chat_template_goldens.py",
        "renderer": "jinja2 " + jinja2.__version__ +
                    " transformers apply_chat_template environment",
        "templates": {name: {key: value for key, value in spec.items()
                             if key not in ("file", "strip_final_lf")}
                      for name, spec in TEMPLATES.items()},
        "cases": {},
    }
    for case_name, (variants, case) in CASES.items():
        for variant in variants:
            rendered = render(templates[variant], case)
            entry = {"template": variant, **case,
                     "rendered_sha256":
                         hashlib.sha256(rendered.encode("utf-8")).hexdigest()}
            section["cases"][f"{variant}.{case_name}"] = entry
    return section


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="update the qwen35 section of the fixture")
    parser.add_argument("--show", metavar="CASE",
                        help="print the rendered prompt of one case")
    args = parser.parse_args()
    if args.show:
        variant, case_name = args.show.split(".", 1)
        text = load_template(TEMPLATES[variant])
        sys.stdout.write(render(text, CASES[case_name][1]))
        return 0
    section = build()
    if not args.write:
        json.dump(section, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["qwen35"] = section
    FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

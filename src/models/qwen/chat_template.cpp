#include "src/models/qwen/chat_template.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

ChatTemplateOptions ResolveQwenChatOptions(const QwenTemplateTraits& traits,
                                           const ReasoningOptions& reasoning,
                                           bool add_vision_id) {
  ChatTemplateOptions options;
  options.add_vision_id = add_vision_id;
  options.dialect = traits.dialect;
  options.emit_empty_system = traits.emit_empty_system;
  options.enable_thinking =
      reasoning.enabled.value_or(traits.default_enable_thinking);
  if (traits.dialect == QwenTemplateDialect::kQwen35) {
    // The Qwen3.5 template reads neither reasoning_effort nor
    // preserve_thinking: thinking survives only after the last user query.
    options.preserve_thinking = false;
    return options;
  }
  options.preserve_thinking =
      reasoning.preserve_thinking.value_or(options.preserve_thinking);
  switch (reasoning.effort.value_or(ReasoningEffort::kXHigh)) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      options.reasoning_effort = QwenReasoningEffort::kLow;
      break;
    case ReasoningEffort::kMedium:
      options.reasoning_effort = QwenReasoningEffort::kMedium;
      break;
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      options.reasoning_effort = QwenReasoningEffort::kXHigh;
      break;
  }
  return options;
}

namespace {

constexpr std::string_view kDefaultChatmlTemplate =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

constexpr std::string_view kLowReasoningInstruction =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstruction =
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer.";

std::string TemplateSha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

bool IsQwen38ReasoningTemplate(std::string_view value) {
  const std::string hash = TemplateSha256(value);
  return hash == QwenChatTemplate::OfficialTemplateSha256() ||
         hash == QwenChatTemplate::UnslothArtifactTemplateSha256();
}

bool IsQwen38Artifact(const core::GgufReader& reader) {
  const auto name = reader.GetMetadataString("general.name");
  if (name.has_value() && name->find("Qwen3.8") != std::string_view::npos) {
    return true;
  }
  const auto config = reader.ExtractModelConfig();
  return config.has_value() && config->num_layers == 64 &&
         config->hidden_size == 5120 && config->vocab_size == 248320;
}

constexpr QwenTemplateTraits kQwen35OfficialTraits{
    .dialect = QwenTemplateDialect::kQwen35,
    .default_enable_thinking = true,
    .emit_empty_system = true,
};

// Qwen3.5 0.8B/2B and the unsloth Qwen3.5-9B GGUF open the thinking block only
// when a caller enables it explicitly.
constexpr std::string_view kThinkingOffByDefault =
    "enable_thinking is defined and enable_thinking is true";

std::optional<QwenTemplateTraits> PinnedQwen35Traits(std::string_view hash) {
  if (hash == QwenChatTemplate::Qwen35OfficialTemplateSha256()) {
    return kQwen35OfficialTraits;
  }
  if (hash == QwenChatTemplate::Qwen35UnslothGgufTemplateSha256()) {
    auto traits = kQwen35OfficialTraits;
    traits.default_enable_thinking = false;
    return traits;
  }
  if (hash == QwenChatTemplate::Qwen35UnslothTemplateSha256()) {
    auto traits = kQwen35OfficialTraits;
    traits.emit_empty_system = false;
    return traits;
  }
  return std::nullopt;
}

// Unpinned templates of non-Qwen3.8 artifacts (Qwen3.5 fine-tunes) follow the
// official Qwen3.5 rules; only the thinking default is read from the text.
QwenTemplateTraits UnpinnedQwen35Traits(std::string_view template_text) {
  auto traits = kQwen35OfficialTraits;
  traits.default_enable_thinking =
      template_text.find(kThinkingOffByDefault) == std::string_view::npos;
  return traits;
}

// Re-serializes JSON the way Python renders `json.loads(text)`: `str()` for a
// top-level scalar and `json.dumps(ensure_ascii=False)` for containers.
class PythonJsonWriter {
public:
  explicit PythonJsonWriter(std::string_view text) : text_(text) {}

  std::optional<std::string> Convert() {
    std::string output;
    SkipWhitespace();
    if (!Value(output, true)) {
      return std::nullopt;
    }
    SkipWhitespace();
    if (position_ != text_.size()) {
      return std::nullopt;
    }
    return output;
  }

private:
  static constexpr std::size_t kMaxDepth = 64;

  void SkipWhitespace() {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\t' ||
            text_[position_] == '\n' || text_[position_] == '\r')) {
      ++position_;
    }
  }

  bool Consume(std::string_view literal) {
    if (text_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  bool Value(std::string& output, bool top_level) {
    if (position_ >= text_.size()) {
      return false;
    }
    switch (text_[position_]) {
      case '{':
        return Object(output);
      case '[':
        return Array(output);
      case '"': {
        std::string decoded;
        if (!String(decoded)) {
          return false;
        }
        if (top_level) {
          output.append(decoded);
          return true;
        }
        return AppendDumpedString(output, decoded);
      }
      case 't':
        output.append(top_level ? "True" : "true");
        return Consume("true");
      case 'f':
        output.append(top_level ? "False" : "false");
        return Consume("false");
      case 'n':
        output.append(top_level ? "None" : "null");
        return Consume("null");
      default:
        return Number(output);
    }
  }

  bool Object(std::string& output) {
    if (++depth_ > kMaxDepth) {
      return false;
    }
    ++position_;
    // Python dicts keep the first position of a repeated key and its last
    // value.
    std::vector<std::pair<std::string, std::string>> members;
    SkipWhitespace();
    if (position_ < text_.size() && text_[position_] == '}') {
      ++position_;
    } else {
      while (true) {
        SkipWhitespace();
        std::string key;
        if (position_ >= text_.size() || text_[position_] != '"' ||
            !String(key)) {
          return false;
        }
        SkipWhitespace();
        if (!Consume(":")) {
          return false;
        }
        SkipWhitespace();
        std::string value;
        if (!Value(value, false)) {
          return false;
        }
        const auto existing = std::find_if(
            members.begin(), members.end(),
            [&](const auto& member) { return member.first == key; });
        if (existing != members.end()) {
          existing->second = std::move(value);
        } else {
          members.emplace_back(std::move(key), std::move(value));
        }
        SkipWhitespace();
        if (Consume("}")) {
          break;
        }
        if (!Consume(",")) {
          return false;
        }
      }
    }
    output.push_back('{');
    for (std::size_t index = 0; index < members.size(); ++index) {
      if (index != 0) {
        output.append(", ");
      }
      if (!AppendDumpedString(output, members[index].first)) {
        return false;
      }
      output.append(": ");
      output.append(members[index].second);
    }
    output.push_back('}');
    --depth_;
    return true;
  }

  bool Array(std::string& output) {
    if (++depth_ > kMaxDepth) {
      return false;
    }
    ++position_;
    output.push_back('[');
    SkipWhitespace();
    if (Consume("]")) {
      output.push_back(']');
      --depth_;
      return true;
    }
    for (bool first = true;; first = false) {
      if (!first) {
        output.append(", ");
      }
      SkipWhitespace();
      if (!Value(output, false)) {
        return false;
      }
      SkipWhitespace();
      if (Consume("]")) {
        break;
      }
      if (!Consume(",")) {
        return false;
      }
    }
    output.push_back(']');
    --depth_;
    return true;
  }

  bool Hex4(std::uint32_t& value) {
    if (position_ + 4 > text_.size()) {
      return false;
    }
    value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
      const char digit = text_[position_++];
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<std::uint32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<std::uint32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<std::uint32_t>(digit - 'A' + 10);
      } else {
        return false;
      }
    }
    return true;
  }

  static void AppendUtf8(std::string& output, std::uint32_t code_point) {
    if (code_point < 0x80) {
      output.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
      output.push_back(static_cast<char>(0xC0 | (code_point >> 6U)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3FU)));
    } else if (code_point < 0x10000) {
      output.push_back(static_cast<char>(0xE0 | (code_point >> 12U)));
      output.push_back(static_cast<char>(0x80 | ((code_point >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xF0 | (code_point >> 18U)));
      output.push_back(static_cast<char>(0x80 | ((code_point >> 12U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80 | ((code_point >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3FU)));
    }
  }

  // Decodes a JSON string. A lone surrogate is rejected: Python keeps it in
  // the string, but the rendered prompt could not be encoded as UTF-8.
  bool String(std::string& decoded) {
    ++position_;
    while (position_ < text_.size()) {
      const auto character = static_cast<unsigned char>(text_[position_++]);
      if (character == '"') {
        return true;
      }
      if (character < 0x20) {
        return false;
      }
      if (character != '\\') {
        decoded.push_back(static_cast<char>(character));
        continue;
      }
      if (position_ >= text_.size()) {
        return false;
      }
      switch (text_[position_++]) {
        case '"':
          decoded.push_back('"');
          break;
        case '\\':
          decoded.push_back('\\');
          break;
        case '/':
          decoded.push_back('/');
          break;
        case 'b':
          decoded.push_back('\b');
          break;
        case 'f':
          decoded.push_back('\f');
          break;
        case 'n':
          decoded.push_back('\n');
          break;
        case 'r':
          decoded.push_back('\r');
          break;
        case 't':
          decoded.push_back('\t');
          break;
        case 'u': {
          std::uint32_t code_point = 0;
          if (!Hex4(code_point)) {
            return false;
          }
          if (code_point >= 0xD800 && code_point <= 0xDBFF) {
            std::uint32_t low = 0;
            if (!Consume("\\u") || !Hex4(low) || low < 0xDC00 || low > 0xDFFF) {
              return false;
            }
            code_point =
                0x10000 + ((code_point - 0xD800) << 10U) + (low - 0xDC00);
          } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
            return false;
          }
          AppendUtf8(decoded, code_point);
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }

  // json.dumps(ensure_ascii=False) escaping.
  static bool AppendDumpedString(std::string& output, std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    output.push_back('"');
    for (const char raw : value) {
      const auto character = static_cast<unsigned char>(raw);
      switch (character) {
        case '"':
          output.append("\\\"");
          break;
        case '\\':
          output.append("\\\\");
          break;
        case '\b':
          output.append("\\b");
          break;
        case '\f':
          output.append("\\f");
          break;
        case '\n':
          output.append("\\n");
          break;
        case '\r':
          output.append("\\r");
          break;
        case '\t':
          output.append("\\t");
          break;
        default:
          if (character < 0x20) {
            output.append("\\u00");
            output.push_back(kHex[character >> 4U]);
            output.push_back(kHex[character & 0x0FU]);
          } else {
            output.push_back(raw);
          }
          break;
      }
    }
    output.push_back('"');
    return true;
  }

  // Integers keep their digits (Python ints are exact); floats use Python's
  // shortest round-trip repr.
  bool Number(std::string& output) {
    const std::size_t begin = position_;
    if (position_ < text_.size() && text_[position_] == '-') {
      ++position_;
    }
    const auto digits = [&] {
      const std::size_t start = position_;
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
      return position_ - start;
    };
    const std::size_t integer_begin = position_;
    const std::size_t integer_digits = digits();
    if (integer_digits == 0 ||
        (integer_digits > 1 && text_[integer_begin] == '0')) {
      return false;
    }
    bool is_float = false;
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      is_float = true;
      if (digits() == 0) {
        return false;
      }
    }
    if (position_ < text_.size() &&
        (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      is_float = true;
      if (position_ < text_.size() &&
          (text_[position_] == '+' || text_[position_] == '-')) {
        ++position_;
      }
      if (digits() == 0) {
        return false;
      }
    }
    const std::string_view lexeme = text_.substr(begin, position_ - begin);
    if (!is_float) {
      output.append(lexeme == "-0" ? std::string_view("0") : lexeme);
      return true;
    }
    double value = 0.0;
    const auto parsed =
        std::from_chars(lexeme.data(), lexeme.data() + lexeme.size(), value);
    if (parsed.ec != std::errc{} || !std::isfinite(value)) {
      return false;
    }
    return AppendPythonFloat(output, value);
  }

  static bool AppendPythonFloat(std::string& output, double value) {
    char buffer[64];
    const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                         std::chars_format::scientific);
    if (converted.ec != std::errc{}) {
      return false;
    }
    const std::string_view scientific(buffer, converted.ptr - buffer);
    const auto exponent_mark = scientific.find('e');
    std::string_view mantissa = scientific.substr(0, exponent_mark);
    if (mantissa.starts_with('-')) {
      output.push_back('-');
      mantissa.remove_prefix(1);
    }
    std::string significant;
    for (const char character : mantissa) {
      if (character != '.') {
        significant.push_back(character);
      }
    }
    int exponent = 0;
    const auto exponent_text = scientific.substr(exponent_mark + 1);
    const auto* exponent_begin = exponent_text.data();
    if (*exponent_begin == '+') {
      ++exponent_begin;
    }
    if (std::from_chars(exponent_begin,
                        exponent_text.data() + exponent_text.size(), exponent)
            .ec != std::errc{}) {
      return false;
    }
    // CPython float_repr_style "short": fixed notation for -4 < decpt <= 16,
    // where the value is 0.<digits> * 10^decpt.
    const int decimal_point = exponent + 1;
    const auto length = static_cast<int>(significant.size());
    if (decimal_point > -4 && decimal_point <= 16) {
      if (decimal_point <= 0) {
        output.append("0.");
        output.append(static_cast<std::size_t>(-decimal_point), '0');
        output.append(significant);
      } else if (decimal_point >= length) {
        output.append(significant);
        output.append(static_cast<std::size_t>(decimal_point - length), '0');
        output.append(".0");
      } else {
        output.append(significant.substr(0, decimal_point));
        output.push_back('.');
        output.append(significant.substr(decimal_point));
      }
      return true;
    }
    output.push_back(significant.front());
    if (length > 1) {
      output.push_back('.');
      output.append(significant.substr(1));
    }
    const int printed_exponent = decimal_point - 1;
    output.push_back('e');
    output.push_back(printed_exponent < 0 ? '-' : '+');
    const std::string magnitude = std::to_string(std::abs(printed_exponent));
    if (magnitude.size() < 2) {
      output.push_back('0');
    }
    output.append(magnitude);
    return true;
  }

  std::string_view text_;
  std::size_t position_{0};
  std::size_t depth_{0};
};

// Splits the JSON object text into member names and raw value texts.
std::optional<std::vector<std::pair<std::string, std::string_view>>>
SplitJsonObject(std::string_view text) {
  std::size_t position = 0;
  const auto skip = [&] {
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\t' ||
            text[position] == '\n' || text[position] == '\r')) {
      ++position;
    }
  };
  // Advances past one JSON value without validating it; the caller already
  // decoded the same text with the strict parser.
  const auto skip_value = [&]() -> bool {
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    const std::size_t start = position;
    for (; position < text.size(); ++position) {
      const char character = text[position];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (character == '\\') {
          escaped = true;
        } else if (character == '"') {
          in_string = false;
          if (depth == 0) {
            ++position;
            return true;
          }
        }
        continue;
      }
      if (character == '"') {
        in_string = true;
      } else if (character == '{' || character == '[') {
        ++depth;
      } else if (character == '}' || character == ']') {
        if (depth == 0) {
          return position != start;
        }
        if (--depth == 0) {
          ++position;
          return true;
        }
      } else if (depth == 0 &&
                 (character == ',' || character == ' ' || character == '\t' ||
                  character == '\n' || character == '\r')) {
        return position != start;
      }
    }
    return position != start && !in_string && depth == 0;
  };
  std::vector<std::pair<std::string, std::string_view>> members;
  skip();
  if (position >= text.size() || text[position] != '{') {
    return std::nullopt;
  }
  ++position;
  skip();
  if (position < text.size() && text[position] == '}') {
    return members;
  }
  while (position < text.size()) {
    skip();
    const std::size_t key_begin = position;
    if (key_begin >= text.size() || text[key_begin] != '"' || !skip_value()) {
      return std::nullopt;
    }
    std::string key;
    const auto key_json = text.substr(key_begin, position - key_begin);
    PythonJsonWriter key_writer(key_json);
    auto decoded_key = key_writer.Convert();
    if (!decoded_key) {
      return std::nullopt;
    }
    key = std::move(*decoded_key);
    skip();
    if (position >= text.size() || text[position] != ':') {
      return std::nullopt;
    }
    ++position;
    skip();
    const std::size_t value_begin = position;
    if (!skip_value()) {
      return std::nullopt;
    }
    members.emplace_back(std::move(key),
                         text.substr(value_begin, position - value_begin));
    skip();
    if (position < text.size() && text[position] == ',') {
      ++position;
      continue;
    }
    if (position < text.size() && text[position] == '}') {
      return members;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::string_view Trim(std::string_view value) {
  const auto whitespace = [](UChar32 cp) {
    // Python str.strip also includes the four ASCII information separators.
    return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
  };
  std::int32_t first = 0;
  auto last = static_cast<std::int32_t>(value.size());
  while (first < last) {
    auto next = first;
    UChar32 cp;
    U8_NEXT(value.data(), next, last, cp);
    if (!whitespace(cp))
      break;
    first = next;
  }
  while (first < last) {
    auto previous = last;
    UChar32 cp;
    U8_PREV(value.data(), first, previous, cp);
    if (!whitespace(cp))
      break;
    last = previous;
  }
  return value.substr(first, last - first);
}

std::string PythonJsonSpacing(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    output.push_back(character);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

void AppendReasoningInstruction(std::string& output,
                                const ChatTemplateOptions& options) {
  if (!options.enable_thinking) {
    return;
  }
  switch (options.reasoning_effort) {
    case QwenReasoningEffort::kLow:
      output.append(kLowReasoningInstruction);
      return;
    case QwenReasoningEffort::kMedium:
      return;
    case QwenReasoningEffort::kXHigh:
      output.append(kXHighReasoningInstruction);
      return;
  }
}

void AppendJsonString(std::string& output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        output.push_back(static_cast<char>(character));
        break;
    }
  }
  output.push_back('"');
}

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools,
                       bool require_tool_call) {
  if (tools.empty()) {
    return;
  }

  if (!output.empty()) {
    output.append("\n\n");
  }
  output.append(
      "# Tools\n\nYou have access to the following functions:\n\n<tools>");
  for (const auto& tool : tools) {
    if (!tool.definition_json.empty()) {
      output.push_back('\n');
      output.append(PythonJsonSpacing(tool.definition_json));
      continue;
    }
    output.append("\n{\"type\": \"function\", \"function\": {\"name\": ");
    AppendJsonString(output, tool.name);
    output.append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output.append(", \"parameters\": ");
    output.append(tool.parameters_json.empty()
                      ? "{}"
                      : PythonJsonSpacing(tool.parameters_json));
    output.append("}}");
  }
  output.append(
      "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
      "following format with NO suffix:\n\n<tool_call>\n"
      "<function=example_function_name>\n<parameter=example_parameter_1>\n"
      "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
      "This is the value for the second parameter\nthat can span\nmultiple "
      "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
      "Reminder:\n- Function calls MUST follow the specified format: an inner "
      "<function=...></function> block must be nested within "
      "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
      "specified\n- You may provide optional reasoning for your function call "
      "in natural language BEFORE the function call, but NOT after\n- If "
      "there is no function call available, answer the question like normal "
      "with your current knowledge and do not tell the user about function "
      "calls\n</IMPORTANT>");
  if (require_tool_call) {
    output.append(
        "\n\nYou must call at least one available function. Do not answer the "
        "user directly.");
  }
}

void AppendToolCalls(std::string& output,
                     std::span<const ChatMessage::ToolCall> calls,
                     QwenTemplateDialect dialect) {
  bool first_call = true;
  for (const auto& call : calls) {
    if (!first_call) {
      output.push_back('\n');
    }
    first_call = false;
    output.append("<tool_call>\n<function=");
    output.append(call.name);
    output.append(">\n");
    for (const auto& argument : call.arguments) {
      output.append("<parameter=");
      output.append(argument.name);
      output.append(">\n");
      if (dialect == QwenTemplateDialect::kQwen35 && !argument.is_string) {
        // Qwen3.5 applies `|string` to scalars and `tojson` to containers.
        if (!argument.python_value.empty()) {
          output.append(argument.python_value);
        } else {
          output.append(
              QwenPythonToolValue(argument.value).value_or(argument.value));
        }
      } else {
        output.append(argument.value);
      }
      output.append("\n</parameter>\n");
    }
    output.append("</function>\n</tool_call>");
  }
}

}  // namespace

std::optional<std::string> QwenPythonToolValue(std::string_view json_value) {
  PythonJsonWriter writer(json_value);
  return writer.Convert();
}

bool AssignQwenPythonToolValues(
    std::string_view arguments_json,
    std::span<ChatMessage::ToolArgument> arguments) {
  const auto members = SplitJsonObject(arguments_json);
  if (!members.has_value() || members->size() != arguments.size()) {
    return false;
  }
  std::vector<std::string> values(arguments.size());
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    if ((*members)[index].first != arguments[index].name) {
      return false;
    }
    if (arguments[index].is_string) {
      continue;
    }
    auto value = QwenPythonToolValue((*members)[index].second);
    if (!value.has_value()) {
      return false;
    }
    values[index] = std::move(*value);
  }
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    arguments[index].python_value = std::move(values[index]);
  }
  return true;
}

QwenTemplateCacheIdentity QwenCacheIdentityFields(
    QwenTemplateDialect dialect) noexcept {
  if (dialect == QwenTemplateDialect::kQwen35) {
    return {
        .model_kind = "qwen3.5",
        .chat_template = "qwen35-compiled-v1",
        .reference_sha256 = QwenChatTemplate::Qwen35OfficialTemplateSha256()};
  }
  return {.model_kind = "qwen3.8",
          .chat_template = "qwen38-reasoning-compiled-v3",
          .reference_sha256 = QwenChatTemplate::OfficialTemplateSha256()};
}

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto template_str = reader.GetMetadataString("tokenizer.chat_template");
  if (!template_str.has_value() || template_str->empty()) {
    if (IsQwen38Artifact(reader)) {
      if (error_msg != nullptr) {
        *error_msg = "Qwen3.8 GGUF is missing tokenizer.chat_template metadata";
      }
      return nullptr;
    }
    if (error_msg != nullptr) {
      *error_msg =
          "GGUF metadata missing 'tokenizer.chat_template', using the compiled "
          "Qwen3.5 template";
    }
    return std::unique_ptr<QwenChatTemplate>(
        new QwenChatTemplate(std::string(kDefaultChatmlTemplate),
                             TemplateSha256(kDefaultChatmlTemplate),
                             Profile::kQwen35, kQwen35OfficialTraits, false));
  }
  const std::string hash = TemplateSha256(*template_str);
  if (IsQwen38ReasoningTemplate(*template_str)) {
    return std::unique_ptr<QwenChatTemplate>(
        new QwenChatTemplate(std::string(*template_str), hash,
                             Profile::kQwen38Reasoning, kQwen38TemplateTraits));
  }
  if (IsQwen38Artifact(reader)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned "
          "version: " +
          hash;
    }
    return nullptr;
  }
  const auto pinned = PinnedQwen35Traits(hash);
  return std::unique_ptr<QwenChatTemplate>(
      new QwenChatTemplate(std::string(*template_str), hash, Profile::kQwen35,
                           pinned.value_or(UnpinnedQwen35Traits(*template_str)),
                           pinned.has_value()));
}

bool QwenChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                            std::string* error_msg) {
  if (!IsQwen38Artifact(reader)) {
    return true;
  }
  return CreateFromGguf(reader, error_msg) != nullptr;
}

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateDefault(
    std::string_view raw_template) {
  if (raw_template.empty()) {
    return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
        std::string(kDefaultChatmlTemplate),
        TemplateSha256(kDefaultChatmlTemplate), Profile::kLegacyChatMl));
  }
  const std::string hash = TemplateSha256(raw_template);
  if (const auto pinned = PinnedQwen35Traits(hash)) {
    return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
        std::string(raw_template), hash, Profile::kQwen35, *pinned));
  }
  return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
      std::string(raw_template), hash,
      IsQwen38ReasoningTemplate(raw_template) ? Profile::kQwen38Reasoning
                                              : Profile::kLegacyChatMl));
}

std::string_view QwenChatTemplate::GetTemplateId() const noexcept {
  switch (profile_) {
    case Profile::kLegacyChatMl:
      return "qwen-chatml-compiled-v1";
    case Profile::kQwen38Reasoning:
      return "qwen38-reasoning-compiled-v3";
    case Profile::kQwen35:
      return "qwen35-compiled-v1";
  }
  return "qwen-chatml-compiled-v1";
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, const ChatTemplateOptions& options,
    std::string* error_msg) {
  return Render(messages, {}, options, error_msg);
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets) {
  if (image_offsets != nullptr)
    image_offsets->clear();
  if (messages.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "No messages provided";
    }
    return std::nullopt;
  }

  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len +=
        msg.content.size() + msg.thought.size() + 32 + msg.images.size() * 64;
    std::size_t previous = 0;
    for (const auto& image : msg.images) {
      if (msg.role != ChatRole::kUser || image.bytes == nullptr ||
          image.offset < previous || image.offset > msg.content.size()) {
        if (error_msg != nullptr)
          *error_msg = "invalid user image content";
        return std::nullopt;
      }
      previous = image.offset;
    }
    for (const auto& call : msg.tool_calls) {
      estimated_len += call.name.size() + 64;
      for (const auto& argument : call.arguments) {
        estimated_len += argument.name.size() + argument.value.size() + 40;
      }
    }
  }
  for (const auto& tool : tools) {
    estimated_len += tool.name.size() + tool.description.size() +
                     tool.parameters_json.size() +
                     2 * tool.definition_json.size() + 96;
  }
  if (options.add_generation_prompt) {
    estimated_len += 64;
  }
  estimated_len += kXHighReasoningInstruction.size() + 64;

  if (estimated_len > options.max_output_bytes ||
      estimated_len >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    if (error_msg != nullptr) {
      *error_msg = "Rendered prompt estimated length (" +
                   std::to_string(estimated_len) +
                   " bytes) exceeds maximum bound (" +
                   std::to_string(options.max_output_bytes) + " bytes)";
    }
    return std::nullopt;
  }

  output.reserve(estimated_len);

  std::size_t message_index = 0;
  std::string system_content;
  while (message_index < messages.size() &&
         (messages[message_index].role == ChatRole::kSystem ||
          messages[message_index].role == ChatRole::kDeveloper)) {
    const auto content = Trim(messages[message_index].content);
    if (!content.empty()) {
      if (!system_content.empty())
        system_content.push_back('\n');
      system_content.append(content);
    }
    ++message_index;
  }

  const bool qwen35 = options.dialect == QwenTemplateDialect::kQwen35;
  std::string system_prefix;
  if (!qwen35) {
    AppendReasoningInstruction(system_prefix, options);
  }
  AppendToolsPrompt(system_prefix, tools, options.require_tool_call);
  if (!system_prefix.empty() && !system_content.empty()) {
    system_prefix.append("\n\n");
  }
  system_prefix.append(system_content);
  // The official Qwen3.5 template opens a system turn for any leading system
  // message, even an empty one, when no tools are declared.
  const bool empty_system_turn =
      options.emit_empty_system && message_index > 0 && tools.empty();
  if (!system_prefix.empty() || empty_system_turn) {
    output.append("<|im_start|>system\n");
    output.append(system_prefix);
    output.append("<|im_end|>\n");
  }

  std::size_t last_user_index = messages.size();
  for (std::size_t index = messages.size(); index > 0; --index) {
    if (messages[index - 1].role == ChatRole::kUser) {
      const auto content = Trim(messages[index - 1].content);
      if (messages[index - 1].images.empty() &&
          content.starts_with("<tool_response>") &&
          content.ends_with("</tool_response>"))
        continue;
      last_user_index = index - 1;
      break;
    }
  }
  if (last_user_index == messages.size()) {
    if (error_msg != nullptr)
      *error_msg = "No user query found in messages.";
    return std::nullopt;
  }

  std::size_t image_count = 0;
  for (; message_index < messages.size(); ++message_index) {
    const auto& msg = messages[message_index];
    if (msg.role == ChatRole::kSystem || msg.role == ChatRole::kDeveloper) {
      if (error_msg != nullptr)
        *error_msg = "System message must be at the beginning.";
      return std::nullopt;
    }
    const bool tool_result = msg.role == ChatRole::kTool;
    if (tool_result) {
      output.append("<|im_start|>user\n");
      while (message_index < messages.size() &&
             messages[message_index].role == ChatRole::kTool) {
        const auto& tool_message = messages[message_index];
        output.append("<tool_response>\n");
        output.append(Trim(tool_message.content));
        output.append("\n</tool_response>");
        ++message_index;
        if (message_index < messages.size() &&
            messages[message_index].role == ChatRole::kTool) {
          output.push_back('\n');
        }
      }
      --message_index;
      output.append("<|im_end|>\n");
      continue;
    }
    const auto role_name = ToString(msg.role);
    output.append("<|im_start|>");
    output.append(role_name);
    output.push_back('\n');

    std::string image_content;
    std::vector<std::size_t> local_image_offsets;
    if (!msg.images.empty()) {
      std::size_t cursor = 0;
      for (const auto& image : msg.images) {
        image_content.append(std::string_view(msg.content)
                                 .substr(cursor, image.offset - cursor));
        ++image_count;
        if (options.add_vision_id)
          image_content.append("Picture " + std::to_string(image_count) + ": ");
        image_content.append("<|vision_start|>");
        local_image_offsets.push_back(image_content.size());
        image_content.append("<|image_pad|><|vision_end|>");
        cursor = image.offset;
      }
      image_content.append(std::string_view(msg.content).substr(cursor));
    }
    // Trim the fully rendered content, including image markers. Whitespace
    // between text and images remains significant; image offsets follow the
    // trim.
    const std::string_view untrimmed = msg.images.empty()
                                           ? std::string_view(msg.content)
                                           : std::string_view(image_content);
    std::string_view content = Trim(untrimmed);
    std::string_view thought = Trim(msg.thought);
    if (qwen35 && msg.role == ChatRole::kAssistant && !msg.has_thought &&
        msg.thought.empty()) {
      // Without reasoning_content, Qwen3.5 splits inline thinking out of the
      // content exactly as its template does.
      constexpr std::string_view kOpen = "<think>";
      constexpr std::string_view kClose = "</think>";
      const auto first_close = content.find(kClose);
      if (first_close != std::string_view::npos) {
        std::string_view reasoning = content.substr(0, first_close);
        while (!reasoning.empty() && reasoning.back() == '\n') {
          reasoning.remove_suffix(1);
        }
        if (const auto open = reasoning.rfind(kOpen);
            open != std::string_view::npos) {
          reasoning.remove_prefix(open + kOpen.size());
        }
        while (!reasoning.empty() && reasoning.front() == '\n') {
          reasoning.remove_prefix(1);
        }
        thought = Trim(reasoning);
        content.remove_prefix(content.rfind(kClose) + kClose.size());
        while (!content.empty() && content.front() == '\n') {
          content.remove_prefix(1);
        }
      }
    }
    const bool preserve_thinking = !qwen35 && options.preserve_thinking;
    if (msg.role == ChatRole::kAssistant &&
        (preserve_thinking || message_index > last_user_index)) {
      output.append("<think>\n");
      output.append(thought);
      output.append("\n</think>\n\n");
    }

    if (image_offsets != nullptr && !local_image_offsets.empty()) {
      const auto removed =
          static_cast<std::size_t>(content.data() - untrimmed.data());
      for (const auto offset : local_image_offsets)
        image_offsets->push_back(output.size() + offset - removed);
    }
    output.append(content);
    if (msg.role == ChatRole::kAssistant && !msg.tool_calls.empty()) {
      if (!Trim(content).empty()) {
        output.append("\n\n");
      }
      AppendToolCalls(output, msg.tool_calls, options.dialect);
    }
    output.append("<|im_end|>\n");

    if (output.size() > options.max_output_bytes) {
      if (error_msg != nullptr) {
        *error_msg =
            "Prompt exceeded max output bytes during message rendering";
      }
      return std::nullopt;
    }
  }

  if (options.add_generation_prompt)
    output.append(GenerationPrompt(options.enable_thinking));

  if (output.size() > options.max_output_bytes) {
    if (error_msg != nullptr) {
      *error_msg = "Prompt exceeded max output bytes after generation prompt";
    }
    return std::nullopt;
  }

  return output;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<|im_start|>assistant\n<think>\n"
                         : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    const ChatTemplateOptions& options, std::string* error_msg) {
  return RenderAndTokenize(tokenizer, messages, {}, options, error_msg);
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  if (std::any_of(messages.begin(), messages.end(), [](const auto& message) {
        return !message.images.empty();
      })) {
    if (error_msg != nullptr) {
      *error_msg =
          "image messages require the model's vision prompt preparation";
    }
    return std::nullopt;
  }
  const auto rendered = Render(messages, tools, options, error_msg);
  if (!rendered.has_value()) {
    return std::nullopt;
  }

  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;

  return tokenizer.Encode(*rendered, tok_opts);
}

}  // namespace gufo::tokenization

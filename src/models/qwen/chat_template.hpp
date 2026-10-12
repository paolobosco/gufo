#ifndef GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_
#define GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

/// Chat message roles supported by the Qwen chat template.
enum class ChatRole : std::uint8_t {
  kSystem = 0,
  kDeveloper = 1,
  kUser = 2,
  kAssistant = 3,
  kTool = 4,
};

[[nodiscard]] constexpr std::string_view ToString(ChatRole role) noexcept {
  switch (role) {
    case ChatRole::kSystem:
    case ChatRole::kDeveloper:
      return "system";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

/// A structured input message for chat formatting.
struct ChatMessage {
  ChatMessage() = default;
  ChatMessage(ChatRole message_role, std::string message_content,
              std::string message_name = {}, std::string message_thought = {})
      : role(message_role),
        content(std::move(message_content)),
        name(std::move(message_name)),
        thought(std::move(message_thought)) {}

  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string name;     ///< Optional function/tool name
  std::string thought;  ///< Optional thinking/reasoning prefix
  /// The client supplied reasoning, possibly empty. Qwen3.5 extracts
  /// reasoning from `</think>` in the content only when it did not.
  bool has_thought{false};
  std::string tool_call_id;
  struct ImagePart {
    /// Insert an image before this byte of content. Equal offsets preserve
    /// input order; image-only messages use offset zero.
    std::size_t offset{0};
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
  };
  std::vector<ImagePart> images;

  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
    /// Qwen3.5 rendering of a non-string value: Python `str()` for scalars,
    /// `json.dumps` for objects and arrays. Empty when unknown.
    std::string python_value{};
  };

  struct ToolCall {
    std::string id;
    std::string name;
    std::vector<ToolArgument> arguments;
  };

  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete validated HTTP tool object, preserving field order/extensions.
  std::string definition_json;
};

enum class QwenReasoningEffort : std::uint8_t {
  kLow,
  kMedium,
  kXHigh,
};

/// Template family whose rendering rules the compiled formatter follows.
enum class QwenTemplateDialect : std::uint8_t {
  /// Reasoning-effort instructions and optionally preserved thinking.
  kQwen38,
  /// No effort levels; historical thinking only after the last user query.
  kQwen35,
};

/// Defaults prescribed by the template recognized in a GGUF artifact.
struct QwenTemplateTraits {
  QwenTemplateDialect dialect{QwenTemplateDialect::kQwen38};
  /// Thinking state when a request leaves it unspecified.
  bool default_enable_thinking{true};
  /// Open a system turn for an empty leading system message without tools.
  bool emit_empty_system{false};
};

inline constexpr QwenTemplateTraits kQwen38TemplateTraits{};

/// Formatting options for rendering a conversation into a text prompt.
struct ChatTemplateOptions {
  bool add_generation_prompt{true};
  bool enable_thinking{true};
  QwenReasoningEffort reasoning_effort{QwenReasoningEffort::kXHigh};
  bool preserve_thinking{true};
  bool add_vision_id{false};
  bool require_tool_call{false};
  QwenTemplateDialect dialect{QwenTemplateDialect::kQwen38};
  bool emit_empty_system{false};
  std::size_t max_output_bytes{1024ULL * 1024ULL};  ///< 1 MiB upper bound
};

/// Suffix opened for a new assistant turn, outside the stable conversation.
[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Resolve CLI/API controls against the artifact template's defaults.
/// Qwen3.8 maps provider-neutral minimal/high/max to native low/xhigh/xhigh.
/// Qwen3.5 has no effort levels or preserved thinking, so both are ignored.
[[nodiscard]] ChatTemplateOptions ResolveQwenChatOptions(
    const QwenTemplateTraits& traits, const ReasoningOptions& reasoning,
    bool add_vision_id = false);

/// Qwen3.5 tool-argument text for one complete non-string JSON value: Python
/// `str()` for scalars and `json.dumps(..., ensure_ascii=False)` for objects
/// and arrays. Returns nullopt for invalid JSON or values Python cannot encode.
[[nodiscard]] std::optional<std::string> QwenPythonToolValue(
    std::string_view json_value);

/// Fills `python_value` for the non-string members of the JSON object text
/// `arguments_json`, which `arguments` decodes in member order. Returns false
/// and leaves `arguments` unchanged when the members do not line up.
bool AssignQwenPythonToolValues(std::string_view arguments_json,
                                std::span<ChatMessage::ToolArgument> arguments);

/// Template-dependent fields of the Qwen continuation-cache identity.
struct QwenTemplateCacheIdentity {
  std::string_view model_kind;
  std::string_view chat_template;
  std::string_view reference_sha256;
};

[[nodiscard]] QwenTemplateCacheIdentity QwenCacheIdentityFields(
    QwenTemplateDialect dialect) noexcept;

/// Deterministic, bounded Qwen ChatML formatter.
class QwenChatTemplate {
public:
  enum class Profile : std::uint8_t {
    kLegacyChatMl,
    kQwen38Reasoning,
    kQwen35,
  };

  ~QwenChatTemplate() = default;

  QwenChatTemplate(const QwenChatTemplate&) = delete;
  QwenChatTemplate& operator=(const QwenChatTemplate&) = delete;
  QwenChatTemplate(QwenChatTemplate&&) noexcept = default;
  QwenChatTemplate& operator=(QwenChatTemplate&&) noexcept = default;

  /// Creates a chat template by extracting the template string from GGUF
  /// metadata.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Validates that a Qwen3.8 artifact carries a recognized, pinned chat
  /// template. Other Qwen model revisions retain their legacy formatter.
  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Creates a default Qwen ChatML template formatter.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateDefault(
      std::string_view raw_template = "");

  [[nodiscard]] std::string_view GetTemplateString() const noexcept {
    return template_string_;
  }
  [[nodiscard]] std::string_view GetTemplateSha256() const noexcept {
    return template_sha256_;
  }
  [[nodiscard]] Profile GetProfile() const noexcept { return profile_; }
  [[nodiscard]] std::string_view GetTemplateId() const noexcept;
  [[nodiscard]] const QwenTemplateTraits& Traits() const noexcept {
    return traits_;
  }
  /// False when the rules come from a template that is not pinned by hash.
  [[nodiscard]] bool IsRecognized() const noexcept { return recognized_; }

  [[nodiscard]] static constexpr std::string_view
  OfficialTemplateSha256() noexcept {
    return "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
  }

  [[nodiscard]] static constexpr std::string_view
  UnslothArtifactTemplateSha256() noexcept {
    return "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";
  }

  /// Qwen/Qwen3.5-9B chat_template.jinja; thinking on by default.
  [[nodiscard]] static constexpr std::string_view
  Qwen35OfficialTemplateSha256() noexcept {
    return "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715";
  }

  /// unsloth/Qwen3.5-9B-GGUF embedded template; thinking off by default.
  [[nodiscard]] static constexpr std::string_view
  Qwen35UnslothGgufTemplateSha256() noexcept {
    return "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67";
  }

  /// unsloth/Qwen3.5-9B and unsloth/Qwen3.5-9B-MTP-GGUF template; merges
  /// leading system/developer messages and omits an empty system turn.
  [[nodiscard]] static constexpr std::string_view
  Qwen35UnslothTemplateSha256() noexcept {
    return "8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8";
  }

  /// Formats a list of messages into a deterministic UTF-8 prompt string.
  [[nodiscard]] static std::optional<std::string> Render(
      std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<std::string> Render(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options = {}, std::string* error_msg = nullptr,
      std::vector<std::size_t>* image_offsets = nullptr);

  /// Formats messages and tokenizes the rendered prompt with the given
  /// tokenizer.
  [[nodiscard]] static std::optional<std::vector<TokenId>> RenderAndTokenize(
      const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<std::vector<TokenId>> RenderAndTokenize(
      const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
      std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

private:
  QwenChatTemplate(std::string template_str, std::string template_sha256,
                   Profile profile, QwenTemplateTraits traits = {},
                   bool recognized = true)
      : template_string_(std::move(template_str)),
        template_sha256_(std::move(template_sha256)),
        profile_(profile),
        traits_(traits),
        recognized_(recognized) {}

  std::string template_string_;
  std::string template_sha256_;
  Profile profile_{Profile::kLegacyChatMl};
  QwenTemplateTraits traits_;
  bool recognized_{true};
};

}  // namespace gufo::tokenization

#endif  // GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

// Prints the identity of a Qwen3.6-35B-A3B GGUF without reading any weight
// payload: the metadata that selects this runtime, the tensor formats in
// total and per role, the embedded chat template hash and the stop tokens.
//
//   qwen36_35b_a3b_inventory --model MODEL.gguf

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_35b_a3b/weights.hpp"

namespace q36 = gufo::models::qwen36_35b_a3b;

namespace {

void PrintUint(const gufo::core::GgufReader& reader, std::string_view key) {
  std::cout << key << ": ";
  if (const auto value = reader.GetMetadataUint64(key)) {
    std::cout << *value << '\n';
  } else {
    std::cout << (reader.FindMetadata(key) != nullptr ? "not an integer"
                                                      : "absent")
              << '\n';
  }
}

/// The optional per-block schedule, one digit per block (1 = DeltaNet).
void PrintSchedule(const gufo::core::GgufReader& reader, std::string_view key) {
  std::cout << key << ": ";
  const auto* meta = reader.FindMetadata(key);
  if (meta == nullptr) {
    std::cout << "absent\n";
    return;
  }
  std::string digits;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    for (const auto v : *u)
      digits += v <= 9 ? static_cast<char>('0' + v) : '?';
  } else if (const auto* s =
                 std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    for (const auto v : *s)
      digits += v >= 0 && v <= 9 ? static_cast<char>('0' + v) : '?';
  } else {
    std::cout << "present, not an integer array\n";
    return;
  }
  std::cout << "present, " << digits.size() << " entries " << digits << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else {
      std::cerr << "usage: " << argv[0] << " --model MODEL.gguf\n";
      return 2;
    }
  }
  if (model_path.empty()) {
    std::cerr << "usage: " << argv[0] << " --model MODEL.gguf\n";
    return 2;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << model_path << ": " << error << '\n';
    return 1;
  }
  const std::string architecture(
      reader->GetMetadataString("general.architecture").value_or(""));
  const std::string prefix = architecture + ".";
  std::cout << "file: " << model_path << '\n'
            << "architecture: "
            << (architecture.empty() ? "absent" : architecture) << '\n';
  PrintUint(*reader, prefix + "block_count");
  PrintUint(*reader, prefix + "nextn_predict_layers");
  PrintSchedule(*reader, prefix + "attention.recurrent_layers");

  std::map<std::string_view, std::uint64_t> formats;
  for (const auto& tensor : reader->GetTensors())
    ++formats[gufo::core::ToString(tensor.type)];
  std::cout << "tensors: " << reader->GetTensorCount() << '\n' << "formats:";
  std::string_view separator = " ";
  for (const auto& [type, count] : formats) {
    std::cout << separator << type << " x" << count;
    separator = ", ";
  }
  std::cout << '\n';

  std::cout << "tokenizer.chat_template sha256: ";
  if (const auto chat = reader->GetMetadataString("tokenizer.chat_template")) {
    const std::span<const std::uint8_t> bytes(
        reinterpret_cast<const std::uint8_t*>(chat->data()), chat->size());
    std::cout << gufo::crypto::Sha256Hex(bytes) << " (" << chat->size()
              << " bytes)\n";
  } else {
    std::cout << "absent\n";
  }
  PrintUint(*reader, "tokenizer.ggml.eos_token_id");
  PrintUint(*reader, "tokenizer.ggml.eot_token_id");

  const auto weights = q36::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::cout << "binding: failed: " << error << '\n';
    return 1;
  }
  std::optional<q36::MtpWeights> mtp;
  if (weights->config.HasMtp()) {
    mtp = q36::MtpWeights::Bind(*reader, weights->config, &error);
    if (!mtp) {
      std::cout << "binding: MTP block failed: " << error << '\n';
      return 1;
    }
  }
  std::cout << "binding: trunk" << (mtp ? " and MTP block" : "") << '\n'
            << "formats per role:\n"
            << q36::TypeHistogram(*weights, mtp ? &*mtp : nullptr);
  return 0;
}

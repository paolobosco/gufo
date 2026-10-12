// MTP binding and quantized-oracle contract on synthetic GGUF files.
//
// The MTP block is stored after the target layers (blk.{num_layers}), whether
// num_layers comes from qwen35.nextn_predict_layers or from a trailing
// blk.{block_count-1}.nextn.* block. Every matrix format with a CPU dispatch
// except F16 is accepted, and the CPU oracle on quantized matrices matches the
// same oracle on their F32 dequantization.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufValueType;
using gufo::core::ModelConfig;

constexpr std::size_t kAlignment = 32;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "Assertion failed: " << message << '\n';
  std::exit(1);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

struct TensorSpec {
  std::string name;
  std::vector<std::uint64_t> dims;  // GGML order: dims[0] is the row length.
  GgmlType type{GgmlType::kF32};
  std::vector<std::uint8_t> payload;

  [[nodiscard]] std::size_t Elements() const {
    std::size_t count = 1;
    for (const auto dim : dims) {
      count *= dim;
    }
    return count;
  }
};

// GGUF v3 writer with complete, aligned tensor payloads.
class GgufBuilder {
public:
  void AddString(std::string_view key, std::string_view value) {
    AppendString(metadata_, key);
    AppendPod(metadata_, static_cast<std::uint32_t>(GgufValueType::kString));
    AppendString(metadata_, value);
    ++metadata_count_;
  }

  void AddUint32(std::string_view key, std::uint32_t value) {
    AppendString(metadata_, key);
    AppendPod(metadata_, static_cast<std::uint32_t>(GgufValueType::kUint32));
    AppendPod(metadata_, value);
    ++metadata_count_;
  }

  void AddFloat32(std::string_view key, float value) {
    AppendString(metadata_, key);
    AppendPod(metadata_, static_cast<std::uint32_t>(GgufValueType::kFloat32));
    AppendPod(metadata_, value);
    ++metadata_count_;
  }

  void AddTensor(TensorSpec tensor) {
    Expect(tensor.payload.size() ==
               gufo::quant::EncodedSizeBytes(tensor.type, tensor.Elements()),
           "synthetic payload matches its declared type and shape");
    tensors_.push_back(std::move(tensor));
  }

  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    std::vector<std::uint8_t> out;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    out.insert(out.end(), magic, magic + 4);
    AppendPod(out, std::uint32_t{3});
    AppendPod(out, static_cast<std::uint64_t>(tensors_.size()));
    AppendPod(out, metadata_count_);
    out.insert(out.end(), metadata_.begin(), metadata_.end());

    std::uint64_t offset = 0;
    std::vector<std::uint64_t> offsets;
    for (const auto& tensor : tensors_) {
      offsets.push_back(offset);
      AppendString(out, tensor.name);
      AppendPod(out, static_cast<std::uint32_t>(tensor.dims.size()));
      for (const auto dim : tensor.dims) {
        AppendPod(out, dim);
      }
      AppendPod(out, static_cast<std::uint32_t>(tensor.type));
      AppendPod(out, offset);
      offset = AlignUp(offset + tensor.payload.size());
    }
    out.resize(AlignUp(out.size()), 0);
    const std::size_t data_start = out.size();
    out.resize(data_start + offset, 0);
    for (std::size_t index = 0; index < tensors_.size(); ++index) {
      std::ranges::copy(tensors_[index].payload,
                        out.begin() + static_cast<std::ptrdiff_t>(
                                          data_start + offsets[index]));
    }
    return out;
  }

private:
  [[nodiscard]] static std::uint64_t AlignUp(std::uint64_t value) {
    return (value + kAlignment - 1) / kAlignment * kAlignment;
  }

  template<class T>
  static void AppendPod(std::vector<std::uint8_t>& out, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
  }

  static void AppendString(std::vector<std::uint8_t>& out,
                           std::string_view value) {
    AppendPod(out, static_cast<std::uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
  }

  std::vector<std::uint8_t> metadata_;
  std::uint64_t metadata_count_{0};
  std::vector<TensorSpec> tensors_;
};

// Owns the image so the reader's tensor pointers stay valid.
struct SyntheticGguf {
  std::vector<std::uint8_t> bytes;
  std::shared_ptr<const gufo::core::GgufReader> reader;
};

SyntheticGguf Open(const GgufBuilder& builder) {
  SyntheticGguf gguf{.bytes = builder.Build(), .reader = nullptr};
  std::string error;
  auto reader = gufo::core::GgufReader::OpenMemory(gguf.bytes.data(),
                                                   gguf.bytes.size(), &error);
  Expect(reader != nullptr, "synthetic GGUF opens: " + error);
  gguf.reader = std::move(reader);
  return gguf;
}

std::vector<std::uint8_t> FloatBytes(std::span<const float> values) {
  std::vector<std::uint8_t> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

// fp16 super-block scale for random payloads: every block format used here
// starts with it, and it bounds the decoded weights to roughly +-0.1.
std::uint16_t LeadingScale(GgmlType type) {
  switch (type) {
    case GgmlType::kQ8_0:
    case GgmlType::kQ4_0:
    case GgmlType::kIQ4_NL:
      return 0x1400;  // 2^-10
    case GgmlType::kIQ4_XS:
      return 0x0400;  // 2^-14
    case GgmlType::kIQ3_S:
      return 0x0C00;  // 2^-12
    default:
      Fail("no synthetic block layout for this type");
  }
}

std::vector<std::uint8_t> RandomPayload(GgmlType type, std::size_t elements,
                                        float scale, std::mt19937& rng) {
  if (type == GgmlType::kF32) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> values(elements);
    for (float& value : values) {
      value = dist(rng);
    }
    return FloatBytes(values);
  }
  if (type == GgmlType::kF16) {
    return std::vector<std::uint8_t>(elements * sizeof(std::uint16_t), 0);
  }
  const std::size_t block_elements = gufo::quant::QuantizedBlockElements(type);
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(type, block_elements);
  Expect(block_bytes != 0 && elements % block_elements == 0,
         "synthetic quantized tensor is block aligned");
  std::uniform_int_distribution<unsigned> byte(0, 255);
  std::vector<std::uint8_t> payload(elements / block_elements * block_bytes);
  for (auto& value : payload) {
    value = static_cast<std::uint8_t>(byte(rng));
  }
  const std::uint16_t d = LeadingScale(type);
  for (std::size_t offset = 0; offset < payload.size(); offset += block_bytes) {
    std::memcpy(payload.data() + offset, &d, sizeof(d));
  }
  return payload;
}

TensorSpec Norm(std::string name, std::size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.9F, 1.1F);
  std::vector<float> values(size);
  for (float& value : values) {
    value = dist(rng);
  }
  return {.name = std::move(name),
          .dims = {size},
          .type = GgmlType::kF32,
          .payload = FloatBytes(values)};
}

TensorSpec Matrix(std::string name, std::size_t rows, std::size_t columns,
                  GgmlType type, std::mt19937& rng) {
  const std::size_t elements = rows * columns;
  return {.name = std::move(name),
          .dims = {columns, rows},
          .type = type,
          .payload = RandomPayload(type, elements, 0.05F, rng)};
}

struct MatrixTypes {
  GgmlType embedding{GgmlType::kF32};  // token_embd and output
  GgmlType eh_proj{GgmlType::kF32};
  GgmlType attn_q{GgmlType::kF32};
  GgmlType attn_kv{GgmlType::kF32};
  GgmlType attn_v{GgmlType::kF32};
  GgmlType attn_output{GgmlType::kF32};
  GgmlType ffn_gate_up{GgmlType::kF32};
  GgmlType ffn_down{GgmlType::kF32};
};

// Tensors read by QwenMtpWeights::LoadFromGguf, with the MTP block at
// blk.{mtp_block}.
std::vector<TensorSpec> MtpTensors(const ModelConfig& config,
                                   std::uint32_t mtp_block,
                                   const MatrixTypes& types,
                                   std::uint32_t seed = 0x5EED) {
  std::mt19937 rng(seed);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::string block = "blk." + std::to_string(mtp_block) + ".";

  std::vector<TensorSpec> tensors;
  tensors.push_back(Matrix("token_embd.weight", config.vocab_size, hidden,
                           types.embedding, rng));
  tensors.push_back(
      Matrix("output.weight", config.vocab_size, hidden, types.embedding, rng));
  tensors.push_back(Norm(block + "nextn.enorm.weight", hidden, rng));
  tensors.push_back(Norm(block + "nextn.hnorm.weight", hidden, rng));
  tensors.push_back(Matrix(block + "nextn.eh_proj.weight", hidden, 2 * hidden,
                           types.eh_proj, rng));
  tensors.push_back(Norm(block + "nextn.shared_head_norm.weight", hidden, rng));
  tensors.push_back(Norm(block + "attn_norm.weight", hidden, rng));
  tensors.push_back(Matrix(block + "attn_q.weight", 2 * attention, hidden,
                           types.attn_q, rng));
  tensors.push_back(
      Matrix(block + "attn_k.weight", kv, hidden, types.attn_kv, rng));
  tensors.push_back(
      Matrix(block + "attn_v.weight", kv, hidden, types.attn_v, rng));
  tensors.push_back(Matrix(block + "attn_output.weight", hidden, attention,
                           types.attn_output, rng));
  tensors.push_back(Norm(block + "attn_q_norm.weight", config.head_dim, rng));
  tensors.push_back(Norm(block + "attn_k_norm.weight", config.head_dim, rng));
  tensors.push_back(Norm(block + "post_attention_norm.weight", hidden, rng));
  tensors.push_back(Matrix(block + "ffn_gate.weight", intermediate, hidden,
                           types.ffn_gate_up, rng));
  tensors.push_back(Matrix(block + "ffn_up.weight", intermediate, hidden,
                           types.ffn_gate_up, rng));
  tensors.push_back(Matrix(block + "ffn_down.weight", hidden, intermediate,
                           types.ffn_down, rng));
  return tensors;
}

// The same tensors with every non-F32 payload replaced by its dequantization.
std::vector<TensorSpec> DequantizedToF32(std::vector<TensorSpec> tensors) {
  for (auto& tensor : tensors) {
    if (tensor.type == GgmlType::kF32) {
      continue;
    }
    std::vector<float> values(tensor.Elements());
    gufo::quant::Dequantize(tensor.type, tensor.payload.data(), values.data(),
                            values.size());
    tensor.type = GgmlType::kF32;
    tensor.payload = FloatBytes(values);
  }
  return tensors;
}

GgufBuilder MetadataBuilder(const ModelConfig& config,
                            std::uint32_t block_count,
                            std::optional<std::uint32_t> nextn_layers) {
  GgufBuilder builder;
  builder.AddString("general.architecture", "qwen35");
  builder.AddUint32("qwen35.block_count", block_count);
  if (nextn_layers.has_value()) {
    builder.AddUint32("qwen35.nextn_predict_layers", *nextn_layers);
  }
  builder.AddUint32("qwen35.embedding_length", config.hidden_size);
  builder.AddUint32("qwen35.feed_forward_length", config.intermediate_size);
  builder.AddUint32("qwen35.attention.head_count", config.num_attention_heads);
  builder.AddUint32("qwen35.attention.head_count_kv",
                    config.num_key_value_heads);
  builder.AddUint32("qwen35.attention.key_length", config.head_dim);
  builder.AddUint32("qwen35.context_length", config.context_length);
  builder.AddUint32("qwen35.full_attention_interval",
                    config.full_attention_interval);
  builder.AddUint32("qwen35.ssm.conv_kernel", config.ssm_conv_kernel);
  builder.AddUint32("qwen35.ssm.state_size", config.ssm_state_size);
  builder.AddUint32("qwen35.ssm.group_count", config.ssm_group_count);
  builder.AddUint32("qwen35.ssm.time_step_rank", config.ssm_time_step_rank);
  builder.AddUint32("qwen35.ssm.inner_size", config.ssm_inner_size);
  builder.AddUint32("qwen35.rope.dimension_count", config.rotary_dim);
  builder.AddFloat32("qwen35.rope.freq_base", config.rope_theta);
  return builder;
}

SyntheticGguf MakeGguf(const ModelConfig& config, std::uint32_t block_count,
                       std::optional<std::uint32_t> nextn_layers,
                       std::vector<TensorSpec> tensors) {
  auto builder = MetadataBuilder(config, block_count, nextn_layers);
  for (auto& tensor : tensors) {
    builder.AddTensor(std::move(tensor));
  }
  return Open(builder);
}

void ExpectBound(const SyntheticGguf& gguf, std::uint32_t num_layers,
                 std::string_view label) {
  const std::string prefix = std::string(label) + ": ";
  std::string error;
  const auto weights =
      gufo::speculative::QwenMtpWeights::LoadFromGguf(*gguf.reader, &error);
  Expect(weights.has_value(), prefix + "MTP weights load: " + error);
  Expect(weights->config.num_layers == num_layers,
         prefix + "target layer count excludes the MTP block");
  Expect(weights->config.mtp_num_layers == 1, prefix + "one MTP layer");
  const std::string block = "blk." + std::to_string(num_layers) + ".";
  const auto bound_to = [&](const gufo::models::QwenTensorRef& tensor,
                            std::string_view suffix) {
    const auto* info = gguf.reader->FindTensor(block + std::string(suffix));
    return info != nullptr && tensor.data == info->data;
  };
  Expect(
      bound_to(weights->embedding_norm, "nextn.enorm.weight") &&
          bound_to(weights->hidden_norm, "nextn.hnorm.weight") &&
          bound_to(weights->fusion_projection, "nextn.eh_proj.weight") &&
          bound_to(weights->shared_head_norm, "nextn.shared_head_norm.weight"),
      prefix + "nextn tensors bind to blk.{num_layers}");
  Expect(bound_to(weights->layer.attn_q, "attn_q.weight") &&
             bound_to(weights->layer.ffn_norm, "post_attention_norm.weight") &&
             bound_to(weights->layer.ffn_down, "ffn_down.weight"),
         prefix + "layer tensors bind to blk.{num_layers}");
}

std::string LoadError(const SyntheticGguf& gguf) {
  std::string error;
  const auto weights =
      gufo::speculative::QwenMtpWeights::LoadFromGguf(*gguf.reader, &error);
  Expect(!weights.has_value(), "invalid MTP weights are rejected");
  return error;
}

// (a) block_count 5 + nextn_predict_layers 1: the MTP block is blk.4.
void TestExplicitNextnBindsAfterTargetLayers() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const auto gguf = MakeGguf(config, 5, 1, MtpTensors(config, 4, {}));
  ExpectBound(gguf, 4, "explicit nextn_predict_layers");
}

// (b) No nextn_predict_layers key: the trailing blk.4.nextn.* block is the
// MTP layer and the target keeps four layers.
void TestInferredNextnBindsAfterTargetLayers() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const auto gguf =
      MakeGguf(config, 5, std::nullopt, MtpTensors(config, 4, {}));
  ExpectBound(gguf, 4, "inferred nextn layer");
}

// (c) The Qwen3.8-27B layout (block_count 65, nextn 1) still binds blk.64.
void TestQwen38_27BLayoutStillBinds() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const auto gguf = MakeGguf(config, 65, 1, MtpTensors(config, 64, {}));
  ExpectBound(gguf, 64, "Qwen3.8-27B layout");
}

// (d) A four-layer target never falls back to the 27B block index.
void TestFixedBlockIndexIsNotAccepted() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const auto gguf = MakeGguf(config, 5, 1, MtpTensors(config, 64, {}));
  const std::string error = LoadError(gguf);
  Expect(error == "Missing Qwen MTP tensor: blk.4.nextn.enorm.weight",
         "missing blk.4 tensors are reported by name: " + error);
}

// (e) F16 matrices are rejected like in the target loader, and formats
// without a CPU dispatch stay rejected.
void TestUnsupportedMatrixTypesAreRejected() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  for (const auto type : {GgmlType::kF16, GgmlType::kQ4_0}) {
    const auto gguf =
        MakeGguf(config, 5, 1, MtpTensors(config, 4, {.eh_proj = type}));
    const std::string error = LoadError(gguf);
    const std::string expected =
        "Unsupported Qwen MTP tensor type for blk.4.nextn.eh_proj.weight: " +
        std::string(gufo::core::ToString(type));
    Expect(error == expected, "unsupported MTP matrix type: " + error);
  }
}

bool AllFinite(std::span<const float> values) {
  return std::ranges::all_of(values,
                             [](float value) { return std::isfinite(value); });
}

void ExpectClose(std::span<const float> actual, std::span<const float> expected,
                 float tolerance, std::string_view message) {
  Expect(actual.size() == expected.size(), message);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "index " << index << ": " << actual[index] << " vs "
                << expected[index] << '\n';
      Fail(message);
    }
  }
}

// (f) Q8_0, IQ4_NL, IQ4_XS and IQ3_S matrices load, and the CPU oracle on
// them matches the oracle on their F32 dequantization: embedding and output
// rows, fused projection, attention and FFN all address and decode correctly.
void TestQuantizedOracleMatchesDequantizedF32() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const MatrixTypes types{.embedding = GgmlType::kQ8_0,
                          .eh_proj = GgmlType::kIQ4_XS,
                          .attn_q = GgmlType::kQ8_0,
                          .attn_kv = GgmlType::kQ8_0,
                          .attn_v = GgmlType::kIQ4_NL,
                          .attn_output = GgmlType::kQ8_0,
                          .ffn_gate_up = GgmlType::kQ8_0,
                          .ffn_down = GgmlType::kIQ3_S};
  auto quantized_tensors = MtpTensors(config, 4, types);
  auto dense_tensors = DequantizedToF32(quantized_tensors);
  const auto quantized = MakeGguf(config, 5, 1, std::move(quantized_tensors));
  const auto dense = MakeGguf(config, 5, 1, std::move(dense_tensors));

  std::string error;
  auto quantized_reference =
      gufo::speculative::QwenMtpReference::Create(quantized.reader, 8, &error);
  Expect(quantized_reference != nullptr,
         "quantized MTP reference is created: " + error);
  auto dense_reference =
      gufo::speculative::QwenMtpReference::Create(dense.reader, 8, &error);
  Expect(dense_reference != nullptr, "F32 MTP reference is created: " + error);
  Expect(quantized_reference->GetWeights().fusion_projection.type ==
                 GgmlType::kIQ4_XS &&
             quantized_reference->GetWeights().output.type == GgmlType::kQ8_0,
         "quantized MTP matrices keep their storage type");

  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> target_hidden(config.hidden_size);
  for (float& value : target_hidden) {
    value = dist(rng);
  }

  constexpr float kTolerance = 1.0e-5F;
  std::vector<float> quantized_hidden(target_hidden);
  std::vector<float> dense_hidden(target_hidden);
  const std::uint32_t tokens[] = {17, 201};
  for (std::uint32_t position = 0; position < 2; ++position) {
    const auto quantized_out = quantized_reference->ForwardHidden(
        tokens[position], quantized_hidden, position);
    const auto dense_out = dense_reference->ForwardHidden(
        tokens[position], dense_hidden, position);
    Expect(quantized_out.size() == config.hidden_size,
           "quantized MTP step produces a hidden state");
    Expect(AllFinite(quantized_out), "quantized MTP hidden is finite");
    Expect(std::ranges::any_of(quantized_out,
                               [](float value) { return value != 0.0F; }),
           "quantized MTP hidden is nonzero");
    ExpectClose(quantized_out, dense_out, kTolerance,
                "quantized MTP hidden matches its F32 dequantization");
    quantized_hidden.assign(quantized_out.begin(), quantized_out.end());
    dense_hidden.assign(dense_out.begin(), dense_out.end());
  }

  std::vector<float> quantized_logits;
  std::vector<float> dense_logits;
  for (const std::uint32_t token : {0U, 1U, 37U, 128U, 255U}) {
    quantized_logits.push_back(quantized_reference->ComputeLogit(token));
    dense_logits.push_back(dense_reference->ComputeLogit(token));
  }
  ExpectClose(quantized_logits, dense_logits, kTolerance,
              "quantized MTP logits match their F32 dequantization");
  Expect(std::ranges::any_of(
             quantized_logits,
             [&](float logit) { return logit != quantized_logits.front(); }),
         "logits read distinct output rows");
}

}  // namespace

int main() {
  TestExplicitNextnBindsAfterTargetLayers();
  TestInferredNextnBindsAfterTargetLayers();
  TestQwen38_27BLayoutStillBinds();
  TestFixedBlockIndexIsNotAccepted();
  TestUnsupportedMatrixTypesAreRejected();
  TestQuantizedOracleMatchesDequantizedF32();
  std::cout << "Qwen MTP weight tests passed.\n";
  return 0;
}

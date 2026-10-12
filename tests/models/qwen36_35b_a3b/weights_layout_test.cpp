// Binds the tensor layout of Qwen3.6-35B-A3B-UD-Q6_K_XL (Unsloth, MTP block
// included) without its payload: the real names, shapes and formats over one
// anonymous mapping whose payload pages are never touched.

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_35b_a3b/weights.hpp"

namespace q36 = gufo::models::qwen36_35b_a3b;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type;
};

/// UD-Q6_K_XL formats: Q8_0 embedding, head, dense projections and down
/// experts; Q6_K routed gate/up except block 39 (Q8_0); F32 norms, DeltaNet
/// scalars and trunk routers; BF16 routers in the MTP block.
std::vector<Tensor> UdLayout() {
  constexpr std::uint64_t kHidden = 2048;
  constexpr std::uint64_t kVocab = 248320;
  constexpr std::uint64_t kExperts = 256;
  constexpr std::uint64_t kFf = 512;
  constexpr auto kF32 = GgmlType::kF32;
  constexpr auto kQ8 = GgmlType::kQ8_0;
  std::vector<Tensor> t;
  const auto add = [&](std::string name, std::vector<std::uint64_t> dims,
                       GgmlType type) {
    t.push_back({std::move(name), std::move(dims), type});
  };
  add("token_embd.weight", {kHidden, kVocab}, kQ8);
  add("output_norm.weight", {kHidden}, kF32);
  add("output.weight", {kHidden, kVocab}, kQ8);
  for (std::uint32_t il = 0; il < 41; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const bool mtp = il == 40;
    const bool linear = !mtp && (il + 1) % 4 != 0;
    add(p + "attn_norm.weight", {kHidden}, kF32);
    add(p + "post_attention_norm.weight", {kHidden}, kF32);
    if (linear) {
      add(p + "attn_qkv.weight", {kHidden, 8192}, kQ8);
      add(p + "attn_gate.weight", {kHidden, 4096}, kQ8);
      add(p + "ssm_conv1d.weight", {4, 8192}, kF32);
      add(p + "ssm_dt.bias", {32}, kF32);
      add(p + "ssm_a", {32}, kF32);
      add(p + "ssm_alpha.weight", {kHidden, 32}, kF32);
      add(p + "ssm_beta.weight", {kHidden, 32}, kF32);
      add(p + "ssm_norm.weight", {128}, kF32);
      add(p + "ssm_out.weight", {4096, kHidden}, kQ8);
    } else {
      add(p + "attn_q.weight", {kHidden, 8192}, kQ8);
      add(p + "attn_k.weight", {kHidden, 512}, kQ8);
      add(p + "attn_v.weight", {kHidden, 512}, kQ8);
      add(p + "attn_output.weight", {4096, kHidden}, kQ8);
      add(p + "attn_q_norm.weight", {256}, kF32);
      add(p + "attn_k_norm.weight", {256}, kF32);
    }
    const GgmlType router = mtp ? GgmlType::kBF16 : kF32;
    const GgmlType gate_up = il == 39 ? kQ8 : GgmlType::kQ6_K;
    add(p + "ffn_gate_inp.weight", {kHidden, kExperts}, router);
    add(p + "ffn_gate_exps.weight", {kHidden, kFf, kExperts}, gate_up);
    add(p + "ffn_up_exps.weight", {kHidden, kFf, kExperts}, gate_up);
    add(p + "ffn_down_exps.weight", {kFf, kHidden, kExperts}, kQ8);
    add(p + "ffn_gate_inp_shexp.weight", {kHidden}, router);
    add(p + "ffn_gate_shexp.weight", {kHidden, kFf}, kQ8);
    add(p + "ffn_up_shexp.weight", {kHidden, kFf}, kQ8);
    add(p + "ffn_down_shexp.weight", {kFf, kHidden}, kQ8);
    if (mtp) {
      add(p + "nextn.eh_proj.weight", {2 * kHidden, kHidden}, kQ8);
      add(p + "nextn.enorm.weight", {kHidden}, kF32);
      add(p + "nextn.hnorm.weight", {kHidden}, kF32);
      add(p + "nextn.shared_head_norm.weight", {kHidden}, kF32);
    }
  }
  return t;
}

/// Encoded bytes of one tensor in the formats used here.
std::uint64_t EncodedBytes(const Tensor& t) {
  std::uint64_t elements = 1;
  for (const auto d : t.dims)
    elements *= d;
  switch (t.type) {
    case GgmlType::kF32:
      return elements * 4;
    case GgmlType::kBF16:
      return elements * 2;
    case GgmlType::kQ4_0:
      return elements / 32 * 18;
    case GgmlType::kQ8_0:
      return elements / 32 * 34;
    case GgmlType::kQ6_K:
      return elements / 256 * 210;
    default:
      throw std::logic_error("unexpected format in the UD layout");
  }
}

template<class T>
void Pod(std::vector<std::uint8_t>& out, T value) {
  const auto at = out.size();
  out.resize(at + sizeof(value));
  std::memcpy(out.data() + at, &value, sizeof(value));
}

void Text(std::vector<std::uint8_t>& out, std::string_view value) {
  Pod(out, std::uint64_t{value.size()});
  out.insert(out.end(), value.begin(), value.end());
}

/// GGUF v3 header: the qwen35moe metadata Config::FromGguf reads and one
/// descriptor per tensor, every payload at offset 0.
std::vector<std::uint8_t> Header(const std::vector<Tensor>& tensors) {
  std::vector<std::uint8_t> kv;
  std::uint64_t kv_count = 0;
  const auto u32 = [&](std::string_view suffix, std::uint32_t value) {
    Text(kv, std::string("qwen35moe.").append(suffix));
    Pod(kv, std::uint32_t{4});
    Pod(kv, value);
    ++kv_count;
  };
  const auto f32 = [&](std::string_view suffix, float value) {
    Text(kv, std::string("qwen35moe.").append(suffix));
    Pod(kv, std::uint32_t{6});
    Pod(kv, value);
    ++kv_count;
  };
  Text(kv, "general.architecture");
  Pod(kv, std::uint32_t{8});
  Text(kv, "qwen35moe");
  ++kv_count;
  u32("block_count", 41);
  u32("nextn_predict_layers", 1);
  u32("context_length", 262144);
  u32("embedding_length", 2048);
  u32("full_attention_interval", 4);
  u32("attention.head_count", 16);
  u32("attention.head_count_kv", 2);
  u32("attention.key_length", 256);
  u32("attention.value_length", 256);
  u32("rope.dimension_count", 64);
  u32("ssm.conv_kernel", 4);
  u32("ssm.state_size", 128);
  u32("ssm.group_count", 16);
  u32("ssm.time_step_rank", 32);
  u32("ssm.inner_size", 4096);
  u32("expert_count", 256);
  u32("expert_used_count", 8);
  u32("expert_feed_forward_length", 512);
  u32("expert_shared_feed_forward_length", 512);
  f32("attention.layer_norm_rms_epsilon", 1e-6F);
  f32("rope.freq_base", 1e7F);
  // The converter writes the mRoPE sections as int32.
  Text(kv, "qwen35moe.rope.dimension_sections");
  Pod(kv, std::uint32_t{9});
  Pod(kv, std::uint32_t{5});
  Pod(kv, std::uint64_t{4});
  for (const std::int32_t section : {11, 11, 10, 0})
    Pod(kv, section);
  ++kv_count;

  std::vector<std::uint8_t> bytes;
  Pod(bytes, std::uint32_t{0x46554747});
  Pod(bytes, std::uint32_t{3});
  Pod(bytes, std::uint64_t{tensors.size()});
  Pod(bytes, kv_count);
  bytes.insert(bytes.end(), kv.begin(), kv.end());
  for (const auto& t : tensors) {
    Text(bytes, t.name);
    Pod(bytes, static_cast<std::uint32_t>(t.dims.size()));
    for (const auto d : t.dims)
      Pod(bytes, d);
    Pod(bytes, static_cast<std::uint32_t>(t.type));
    Pod(bytes, std::uint64_t{0});
  }
  return bytes;
}

/// A GGUF image in an anonymous, unreserved mapping. The reader checks only
/// each tensor's own extent, so overlapping payloads at offset 0 need room
/// for the header and the largest tensor; only header pages are written.
class Image {
public:
  explicit Image(const std::vector<Tensor>& tensors) {
    const auto header = Header(tensors);
    std::uint64_t largest = 0;
    for (const auto& t : tensors)
      largest = std::max(largest, EncodedBytes(t));
    size_ = (header.size() + 31) / 32 * 32 + largest;
    void* base = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    Require(base != MAP_FAILED, "anonymous mapping failed");
    base_ = base;
    std::memcpy(base_, header.data(), header.size());
  }
  ~Image() { (void)munmap(base_, size_); }
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  [[nodiscard]] std::unique_ptr<gufo::core::GgufReader> Open(
      std::string* error) const {
    return gufo::core::GgufReader::OpenMemory(base_, size_, error);
  }

private:
  void* base_{nullptr};
  std::size_t size_{0};
};

void CheckUdLayout() {
  const auto layout = UdLayout();
  Require(layout.size() == 753, "the UD layout has 753 tensors");
  std::uint64_t largest = 0;
  for (const auto& t : layout)
    largest = std::max(largest, EncodedBytes(t));
  // token_embd and output: 248320 x 2048 Q8_0.
  Require(largest == 540344320, "the largest UD tensor is the Q8_0 head");

  const Image image(layout);
  std::string error;
  const auto reader = image.Open(&error);
  Require(reader != nullptr && error.empty(),
          "the UD layout did not open: " + error);
  Require(reader->GetTensorCount() == 753, "the reader lost descriptors");
  const auto weights = q36::ModelWeights::Bind(*reader, &error);
  Require(weights.has_value(), "the UD trunk did not bind: " + error);
  const auto& c = weights->config;
  Require(c.num_layers == 40 && c.num_layers_all == 41 && c.HasMtp() &&
              c.vocab_size == 248320 && weights->layers.size() == 40,
          "the UD trunk geometry changed");
  const auto mtp = q36::MtpWeights::Bind(*reader, c, &error);
  Require(mtp.has_value(), "the UD MTP block did not bind: " + error);

  Require(weights->token_embd.type == GgmlType::kQ8_0 &&
              weights->output.type == GgmlType::kQ8_0 &&
              weights->output.name == "output.weight" &&
              weights->output_norm.type == GgmlType::kF32,
          "embedding, head or final norm format changed");
  const auto check_block = [&](const q36::LayerWeights& l, std::uint32_t il) {
    const std::string block = "block " + std::to_string(il);
    const bool trunk = il < c.num_layers;
    const GgmlType gate_up = il == 39 ? GgmlType::kQ8_0 : GgmlType::kQ6_K;
    const GgmlType router = trunk ? GgmlType::kF32 : GgmlType::kBF16;
    Require(l.linear == c.IsLinearLayer(il), block + " mixer changed");
    Require(l.ffn_gate_exps.type == gate_up && l.ffn_up_exps.type == gate_up &&
                l.ffn_down_exps.type == GgmlType::kQ8_0,
            block + " routed expert formats changed");
    Require(l.ffn_gate_exps.experts == 256 &&
                l.ffn_gate_exps.SizeBytes() ==
                    (il == 39 ? 285212672U : 220200960U) &&
                l.ffn_down_exps.SizeBytes() == 285212672U,
            block + " routed expert sizes changed");
    Require(l.router.type == router && l.shexp_gate_inp.type == l.router.type,
            block + " router and shared-expert gate must share " +
                std::string(gufo::core::ToString(router)));
    for (const auto* t : {&l.shexp_gate, &l.shexp_up, &l.shexp_down}) {
      Require(t->type == GgmlType::kQ8_0, block + " shared expert format");
    }
    const auto dense =
        l.linear ? std::vector{&l.ssm_qkv, &l.ssm_gate, &l.ssm_out}
                 : std::vector{&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out};
    for (const auto* t : dense) {
      Require(!t->empty() && t->type == GgmlType::kQ8_0,
              block + " dense projection format");
    }
    if (l.linear) {
      Require(l.ssm_alpha.type == GgmlType::kF32 &&
                  l.ssm_beta.type == GgmlType::kF32,
              block + " DeltaNet alpha/beta format");
    }
    Require(l.nextn_eh_proj.empty() == trunk, block + " nextn presence");
  };
  for (std::uint32_t il = 0; il < c.num_layers; ++il)
    check_block(weights->layers[il], il);
  check_block(mtp->block, c.num_layers);
  Require(mtp->block.nextn_eh_proj.type == GgmlType::kQ8_0 &&
              mtp->block.nextn_enorm.type == GgmlType::kF32 &&
              mtp->block.nextn_hnorm.type == GgmlType::kF32 &&
              mtp->block.nextn_shared_norm.type == GgmlType::kF32,
          "nextn formats changed");

  const std::string expected =
      "attn_gate: Q8_0 x30\n"
      "attn_k: Q8_0 x11\n"
      "attn_k_norm: F32 x11\n"
      "attn_norm: F32 x41\n"
      "attn_output: Q8_0 x11\n"
      "attn_q: Q8_0 x11\n"
      "attn_q_norm: F32 x11\n"
      "attn_qkv: Q8_0 x30\n"
      "attn_v: Q8_0 x11\n"
      "ffn_down_exps: Q8_0 x41\n"
      "ffn_down_shexp: Q8_0 x41\n"
      "ffn_gate_exps: Q6_K x40, Q8_0 x1\n"
      "ffn_gate_inp: BF16 x1, F32 x40\n"
      "ffn_gate_inp_shexp: BF16 x1, F32 x40\n"
      "ffn_gate_shexp: Q8_0 x41\n"
      "ffn_up_exps: Q6_K x40, Q8_0 x1\n"
      "ffn_up_shexp: Q8_0 x41\n"
      "nextn.eh_proj: Q8_0 x1\n"
      "nextn.enorm: F32 x1\n"
      "nextn.hnorm: F32 x1\n"
      "nextn.shared_head_norm: F32 x1\n"
      "output: Q8_0 x1\n"
      "output_norm: F32 x1\n"
      "post_attention_norm: F32 x41\n"
      "ssm_a: F32 x30\n"
      "ssm_alpha: F32 x30\n"
      "ssm_beta: F32 x30\n"
      "ssm_conv1d: F32 x30\n"
      "ssm_dt.bias: F32 x30\n"
      "ssm_norm: F32 x30\n"
      "ssm_out: Q8_0 x30\n"
      "token_embd: Q8_0 x1\n"
      "total: BF16 x2, F32 x368, Q6_K x80, Q8_0 x303 (753 tensors)\n";
  const auto histogram = q36::TypeHistogram(*weights, &*mtp);
  Require(histogram == expected, "UD type histogram changed:\n" + histogram);
  const std::string trunk_total =
      "total: F32 x361, Q6_K x78, Q8_0 x294 (733 tensors)\n";
  const auto trunk = q36::TypeHistogram(*weights);
  Require(trunk.ends_with("\n" + trunk_total),
          "trunk type histogram changed:\n" + trunk);
}

void CheckRejectedExpertFormat() {
  auto layout = UdLayout();
  const std::string name = "blk.0.ffn_gate_exps.weight";
  const auto it = std::find_if(layout.begin(), layout.end(),
                               [&](const Tensor& t) { return t.name == name; });
  Require(it != layout.end(), "missing " + name);
  // Q4_0 is a valid GGUF storage type that no routed kernel decodes.
  it->type = GgmlType::kQ4_0;
  const Image image(layout);
  std::string error;
  const auto reader = image.Open(&error);
  Require(reader != nullptr && error.empty(),
          "a Q4_0 expert must still open: " + error);
  Require(!q36::ModelWeights::Bind(*reader, &error).has_value(),
          "a Q4_0 routed expert was bound");
  Require(error.find(name) != std::string::npos &&
              error.find("Q4_0") != std::string::npos,
          "the rejection must name the tensor and its format: " + error);
}

}  // namespace

int main() {
  try {
    CheckUdLayout();
    CheckRejectedExpertFormat();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "Qwen3.6 UD-Q6_K_XL layout checks passed.\n";
}

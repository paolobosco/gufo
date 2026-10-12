#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

// Full-attention layers are 3, 7, ..., 31. Tap 3 follows the first attention
// block before later layers propagate a difference; tap 31 feeds the
// vocabulary projection.
constexpr std::array<std::uint32_t, 3> kCrossPathTaps{3, 7, 31};

// Provisional limits until calibrated on gfx1151; record the measured values
// with the limits. Full prefill and prefill plus one decode step cannot agree
// exactly: prefill quantizes projection activations to Q8_1 and accumulates
// attention in FP16 WMMA tiles, while decode keeps FP32 activations and split-K
// attention. Assumed rounding budget: about 1% relative L2 per tap and a KL
// near 1e-3 nats. The limits leave about 5x margin and grow with the layer
// index because differences propagate. A stale attention output replaces a
// whole attention block at each full-attention layer and should exceed them;
// the first hardware run must confirm this with a negative control.
constexpr std::array<double, kCrossPathTaps.size()> kMaxTapRelativeL2{
    0.05, 0.08, 0.15};
constexpr double kMaxCrossPathKl = 0.05;

// Provisional CPU/GPU limits with the same calibration status. The CPU keeps
// FP32 activations and KV; the first GPU row comes from Q8_1 prefill and later
// rows from decode over an FP16 KV cache.
constexpr double kMaxCpuReferenceKl = 0.05;
// A different top-1 is accepted only as a near-tie: the GPU choice lies within
// this many logits of the CPU maximum.
constexpr float kCpuReferenceTieMargin = 0.1F;
constexpr std::uint32_t kCpuReferenceContext = 64;

// The baseline batched-attention kernel holds (65536 - 1024) / 4 = 16128
// visible positions in LDS. Depth 15361 keeps the first 512-token chunk below
// that limit while the full 2059-token chunk crosses it; depth 16384 starts
// beyond it. Lengths of at least 1025 keep both cross-path prefills on the
// grouped WMMA branch (batch >= 1024).
constexpr std::uint32_t kLongContext = 32768;
constexpr std::uint32_t kChunkDepth = 15361;
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 4>
    kLongPrefillCases{{{0, 64}, {0, 1025}, {kChunkDepth, 2059}, {16384, 1025}}};

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Sub-checks that could not run on this artifact. They are reported with the
// final status line so a partial run is never mistaken for a full pass.
std::vector<std::string>& SkippedChecks() {
  static std::vector<std::string> skipped;
  return skipped;
}

// On the 9B every projection format below takes the same native W x A8 WMMA
// prefill route at any batch size (the FP16 low-bit route is 27B-only), so
// chunked and whole prefills must agree bit for bit.
bool HasChunkExactProjections(const gufo::models::QwenModelWeights& weights) {
  const auto exact = [](const gufo::models::QwenTensorRef& tensor) {
    switch (tensor.type) {
      case gufo::core::GgmlType::kQ8_0:
      case gufo::core::GgmlType::kQ3_K:
      case gufo::core::GgmlType::kQ4_K:
      case gufo::core::GgmlType::kQ5_K:
      case gufo::core::GgmlType::kQ6_K:
      case gufo::core::GgmlType::kIQ4_NL:
      case gufo::core::GgmlType::kIQ4_XS:
      case gufo::core::GgmlType::kIQ3_S:
        return true;
      default:
        return false;
    }
  };
  return std::ranges::all_of(weights.layers, [&](const auto& layer) {
    const bool mixer = layer.is_full_attention
                           ? exact(layer.attn_q) && exact(layer.attn_k) &&
                                 exact(layer.attn_v) && exact(layer.attn_output)
                           : exact(layer.attn_qkv) && exact(layer.attn_gate) &&
                                 exact(layer.ssm_alpha) &&
                                 exact(layer.ssm_beta) && exact(layer.ssm_out);
    return mixer && exact(layer.ffn_gate) && exact(layer.ffn_up) &&
           exact(layer.ffn_down);
  });
}

struct Case {
  std::vector<Token> tokens;
  std::vector<std::vector<float>> logits;
  std::size_t prompt_size;
};

// Teacher forcing keeps quantization comparisons on identical input prefixes.
// Free-running text would conflate different inputs with numerical differences.
constexpr std::array<const char*, 3> kTexts{
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs "
    "when the requested page is unavailable and the operating system must act.",
    "A train leaves the station at noon and travels at sixty kilometers per "
    "hour. "
    "Another train leaves one hour later at ninety kilometers per hour. To "
    "find "
    "when they meet, compare the distances each train has traveled.",
    "template <typename T> class RingBuffer { public: bool push(const T& "
    "value) "
    "{ if (count == capacity) return false; data[tail] = value; "
    "tail = (tail + 1) % capacity; ++count; return true; } };",
};

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  Expect(!view.empty() && std::ranges::all_of(
                              view, [](float v) { return std::isfinite(v); }),
         "target logits must be finite and nonempty");
  return {view.begin(), view.end()};
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::string Fingerprint(std::span<const float> values) {
  Expect(
      !values.empty() &&
          std::ranges::all_of(values, [](float v) { return std::isfinite(v); }),
      "fingerprinted values must be finite and nonempty");
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(values.data()),
       values.size_bytes()});
}

double LogNormalizer(std::span<const float> logits) {
  const double maximum = *std::ranges::max_element(logits);
  double sum = 0;
  for (const double value : logits)
    sum += std::exp(value - maximum);
  return maximum + std::log(sum);
}

struct Divergence {
  double kl{0};
  double tv{0};
};

// KL(P || Q) and total variation over the full vocabulary.
Divergence MeasureDivergence(std::span<const float> p,
                             std::span<const float> q) {
  Expect(p.size() == q.size(), "compared vocabularies differ");
  const double log_p = LogNormalizer(p), log_q = LogNormalizer(q);
  Divergence result;
  for (std::size_t t = 0; t < p.size(); ++t) {
    const double probability = std::exp(p[t] - log_p);
    result.kl += probability * ((p[t] - log_p) - (q[t] - log_q));
    result.tv += std::abs(probability - std::exp(q[t] - log_q)) * 0.5;
  }
  return result;
}

double RelativeL2(std::span<const float> reference,
                  std::span<const float> candidate) {
  Expect(reference.size() == candidate.size(), "compared widths differ");
  double difference = 0, norm = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double delta = static_cast<double>(reference[i]) - candidate[i];
    difference += delta * delta;
    norm += static_cast<double>(reference[i]) * reference[i];
  }
  return std::sqrt(difference / norm);
}

void CheckIdentity(const gufo::core::ModelConfig& config) {
  std::cout << "config hidden=" << config.hidden_size
            << " layers=" << config.num_layers
            << " ffn=" << config.intermediate_size
            << " heads=" << config.num_attention_heads << '/'
            << config.num_key_value_heads << " head_dim=" << config.head_dim
            << " rotary=" << config.rotary_dim
            << " ssm_groups=" << config.ssm_group_count
            << " ssm_rank=" << config.ssm_time_step_rank
            << " ssm_state=" << config.ssm_state_size
            << " ssm_inner=" << config.ssm_inner_size
            << " vocab=" << config.vocab_size << '\n';
  Expect(config.IsValidQwen() && config.hidden_size == 4096 &&
             config.num_layers == 32 && config.intermediate_size == 12288 &&
             config.num_attention_heads == 16 &&
             config.num_key_value_heads == 4 && config.head_dim == 256 &&
             config.rotary_dim == 64 && config.ssm_group_count == 16 &&
             config.ssm_time_step_rank == 32 && config.ssm_state_size == 128 &&
             config.ssm_inner_size == 4096 && config.vocab_size == 248320,
         "target checks require the Qwen3.5-9B shape");
}

void CheckVerificationFeatures(Executor& executor,
                               std::span<const std::vector<float>> expected) {
  const auto actual = executor.GetVerificationHiddenStates();
  const std::size_t width = expected.front().size();
  Expect(actual.size() == expected.size() * width,
         "verification feature dimensions changed");
  for (std::size_t row = 0; row < expected.size(); ++row) {
    Expect(ByteEqual(actual.subspan(row * width, width), expected[row]),
           "verification features differ from scalar decoding");
  }
  Expect(ByteEqual(executor.CopyLastHidden(), expected.back()),
         "last verification features differ from scalar decoding");
}

void CheckMixedContextBatch(const Executor& owner, bool replay) {
  for (const auto storage : {gufo::hip::QwenKvCacheStorage::kFp16,
                             gufo::hip::QwenKvCacheStorage::kFp32}) {
    auto policy = gufo::hip::QwenExecutionPolicy::Production();
    policy.kv_cache_storage = storage;
    Executor short_session(owner.GetSharedModel(), 32, policy);
    Executor long_session(owner.GetSharedModel(), 64, policy);
    Executor longest_session(owner.GetSharedModel(), 128, policy);
    const std::array<Executor*, 3> sessions{&short_session, &long_session,
                                            &longest_session};
    constexpr std::array<std::uint32_t, 3> prefix_sizes{5, 9, 13};
    constexpr std::uint32_t continuation = 4;
    std::array<std::vector<Token>, 3> tokens;
    std::array<std::vector<std::vector<float>>, 3> expected;
    for (std::size_t row = 0; row < sessions.size(); ++row) {
      auto& session = *sessions[row];
      tokens[row] = owner.GetTokenizer().Encode(kTexts[row]);
      Expect(tokens[row].size() >= prefix_sizes[row] + continuation,
             "mixed-context fixture has too few tokens");
      session.Reset();
      (void)session.ForwardPromptBatch(
          std::span(tokens[row]).first(prefix_sizes[row]));
      auto snapshot = session.SaveSnapshot(prefix_sizes[row]);
      for (std::uint32_t step = 0; step < continuation; ++step) {
        const auto position = prefix_sizes[row] + step;
        (void)session.ForwardToken(tokens[row][position], position);
        expected[row].push_back(Logits(session));
      }
      session.RestoreSnapshot(*snapshot);
      if (replay)
        session.SaveState(prefix_sizes[row]);
    }
    for (std::uint32_t step = 0; step < continuation; ++step) {
      std::array<gufo::hip::QwenGpuBatchItem, 3> items;
      for (std::size_t row = 0; row < sessions.size(); ++row) {
        const auto position = prefix_sizes[row] + step;
        items[row] = {sessions[row], tokens[row][position], position};
      }
      // Hand replayed state to each possible coordinator. The longest prefix
      // crosses the replay ring; all rows still use one scalar oracle.
      const auto coordinator = step % sessions.size();
      std::rotate(items.begin(), items.begin() + coordinator, items.end());
      const auto predictions = Executor::ForwardTokenBatch(items);
      Expect(predictions.size() == sessions.size(),
             "mixed-context batch returned the wrong number of rows");
      for (std::size_t row = 0; row < sessions.size(); ++row) {
        const auto actual = Logits(*sessions[row]);
        const bool exact = ByteEqual(actual, expected[row][step]);
        if (!exact) {
          const auto comparison =
              gufo::testing::CompareLogits(actual, expected[row][step]);
          std::cerr << "mixed-context storage=" << static_cast<int>(storage)
                    << " replay=" << replay << " row=" << row
                    << " step=" << step
                    << " max_abs=" << comparison.max_abs_diff << '\n';
        }
        Expect(exact, "mixed-context batching changed target logits");
        const auto& logits = expected[row][step];
        const auto next = static_cast<Token>(std::ranges::max_element(logits) -
                                             logits.begin());
        const std::size_t batch_row =
            (row + sessions.size() - coordinator) % sessions.size();
        Expect(predictions[batch_row] == next,
               "mixed-context batch returned a different token");
      }
      if (replay) {
        for (std::size_t row = 0; row < sessions.size(); ++row) {
          sessions[row]->RestoreState();
          sessions[row]->CommitVerificationChunk(
              std::span(tokens[row]).subspan(prefix_sizes[row], step + 1),
              prefix_sizes[row]);
        }
      }
    }
    std::cout << "mixed-context batch: storage=" << static_cast<int>(storage)
              << " replay=" << replay << " all 12 full-logit rows exact\n";
  }
}

void CheckConcurrencyWidths(const Executor& owner) {
  constexpr std::size_t count = 8;
  constexpr std::size_t cohort_limit = count * count;
  // The C8 coordinator reuses its FFN gate scratch for all cohort logits;
  // smaller coordinators exercise the bounded-allocation fallback. The FFN is
  // narrower than the vocabulary, so size that context from the model shape.
  const auto& config = owner.GetConfig();
  const auto coordinator_context = std::max<std::uint32_t>(
      1024,
      std::bit_ceil(static_cast<std::uint32_t>(
          (cohort_limit * config.vocab_size + config.intermediate_size - 1) /
          config.intermediate_size)));
  std::array<std::unique_ptr<Executor>, count> sessions;
  std::array<std::unique_ptr<gufo::hip::QwenGpuSnapshot>, count> snapshots;
  std::array<gufo::hip::QwenGpuBatchItem, count> items;
  std::array<std::vector<Token>, count> tokens;
  std::array<std::array<std::vector<float>, 9>, count> expected_logits;
  std::array<std::array<std::vector<float>, 9>, count> expected_features;
  for (std::size_t row = 0; row < count; ++row) {
    sessions[row] = std::make_unique<Executor>(
        owner.GetSharedModel(), row == count / 2 ? coordinator_context
                                : row % 2 == 0   ? 64U
                                                 : 128U);
    auto& session = *sessions[row];
    // MTP verification captures the final hidden state without layer taps.
    session.SetPromptHiddenCapture(true);
    const std::string text = kTexts[row % kTexts.size()];
    tokens[row] = owner.GetTokenizer().Encode(text + "\n" + text);
    const auto position = static_cast<std::uint32_t>(5 + 4 * row);
    Expect(tokens[row].size() >= position + expected_logits[row].size(),
           "concurrency fixture has too few tokens");
    (void)session.ForwardPromptBatch(std::span(tokens[row]).first(position));
    snapshots[row] = session.SaveSnapshot(position);
    items[row] = {&session, tokens[row][position], position};
    for (std::size_t step = 0; step < expected_logits[row].size(); ++step) {
      (void)session.ForwardToken(tokens[row][position + step], position + step);
      expected_logits[row][step] = Logits(session);
      const auto features = session.CopyLastHidden();
      expected_features[row][step].assign(features.begin(), features.end());
    }
  }
  // Reuse eight scalar oracles across the required serving widths. Unequal
  // prefixes and different prompts expose accidental sharing of request state.
  for (const std::size_t width : {2U, 4U, 6U, 8U}) {
    for (std::size_t row = 0; row < width; ++row)
      sessions[row]->RestoreSnapshot(*snapshots[row]);
    const auto predictions =
        Executor::ForwardTokenBatch(std::span(items).first(width));
    Expect(predictions.size() == width, "concurrency width changed");
    for (std::size_t row = 0; row < width; ++row) {
      Expect(ByteEqual(Logits(*sessions[row]), expected_logits[row][0]) &&
                 ByteEqual(sessions[row]->CopyLastHidden(),
                           expected_features[row][0]),
             "concurrent decoding changed target logits or hidden state");
      const auto& logits = expected_logits[row][0];
      Expect(predictions[row] ==
                 static_cast<Token>(std::ranges::max_element(logits) -
                                    logits.begin()),
             "concurrent decoding changed the target prediction");
    }
    std::cout << "concurrency width=" << width << " logits/hidden exact=1\n";

    for (const std::size_t cohort_rows :
         {0U,  9U,  10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U, 23U, 24U,
          25U, 28U, 31U, 32U, 33U, 35U, 36U, 42U, 48U, 56U, 64U}) {
      if ((cohort_rows > 16 && width != count) ||
          (cohort_rows > 0 && cohort_rows <= 16 && width != 2))
        continue;
      std::array<gufo::hip::QwenGpuVerificationItem, count> verification_items;
      for (std::size_t row = 0; row < width; ++row) {
        const auto position = items[row].position;
        const std::size_t length =
            cohort_rows != 0 ? cohort_rows / width + (row < cohort_rows % width)
                             : (3 * row + width - 2) % 8 + 1;
        sessions[row]->RestoreSnapshot(*snapshots[row]);
        sessions[row]->SaveState(position);
        verification_items[row] = {
            sessions[row].get(),
            std::span(tokens[row]).subspan(position, length), position, true};
      }
      // Rotate the coordinator, vary chunk lengths and cross eight total rows.
      // Attention capacity and recurrent replay still belong to each sequence.
      const std::size_t rotation = width / 2;
      auto batch = std::span(verification_items).first(width);
      std::rotate(batch.begin(), batch.begin() + rotation, batch.end());
      const auto memory_before = batch.front().executor->GetMemoryUsage();
      const auto verified = Executor::ForwardVerificationBatch(batch);
      const auto memory_after = batch.front().executor->GetMemoryUsage();
      if (width == count && cohort_rows == cohort_limit) {
        const auto retained_limit = 8 * config.vocab_size * sizeof(float);
        Expect(memory_after.TotalBytes() - memory_before.TotalBytes() <=
                   retained_limit,
               "a coordinator must not retain the whole cohort's logits");
      }
      Expect(verified.size() == width, "verification cohort width changed");
      for (std::size_t index = 0; index < width; ++index) {
        const std::size_t row = (index + rotation) % width;
        auto& session = *sessions[row];
        const auto& item = batch[index];
        Expect(verified[index].size() == item.tokens.size(),
               "verification changed a sequence length");
        CheckVerificationFeatures(
            session,
            std::span(expected_features[row]).first(item.tokens.size()));
        for (std::size_t step = 0; step < item.tokens.size(); ++step) {
          const auto& logits = expected_logits[row][step];
          Expect(ByteEqual(session.CopyVerificationLogits(step), logits) &&
                     verified[index][step] ==
                         static_cast<Token>(std::ranges::max_element(logits) -
                                            logits.begin()),
                 "concurrent verification changed logits or predictions");
        }
        const std::size_t committed = item.tokens.size() / 2 + 1;
        session.RestoreState();
        session.CommitVerificationChunk(item.tokens.first(committed),
                                        item.position);
        (void)session.ForwardToken(tokens[row][item.position + committed],
                                   item.position + committed);
        Expect(ByteEqual(Logits(session), expected_logits[row][committed]) &&
                   ByteEqual(session.CopyLastHidden(),
                             expected_features[row][committed]),
               "concurrent verification replay changed continuation state");
      }
      std::cout << "verification concurrency=" << width
                << " cohort_rows=" << cohort_rows
                << " logits/hidden/replay exact=1\n";
    }
  }
  for (std::size_t row = 0; row < count; ++row) {
    sessions[row]->RestoreSnapshot(*snapshots[row]);
    sessions[row]->SaveState(items[row].position);
  }
  // Append chunks without resetting the rollback snapshot. Requests leave
  // independently, including a final single-session continuation.
  for (std::size_t stage = 0; stage < 4; ++stage) {
    const auto active = count >> stage;
    const auto offset = stage * 2;
    std::vector<gufo::hip::QwenGpuVerificationItem> chunks;
    for (std::size_t row = 0; row < active; ++row) {
      const auto position =
          items[row].position + static_cast<std::uint32_t>(offset);
      chunks.push_back({sessions[row].get(),
                        std::span(tokens[row]).subspan(position, 2), position,
                        true});
    }
    const auto rotation = stage % active;
    std::rotate(chunks.begin(), chunks.begin() + rotation, chunks.end());
    const auto predictions = Executor::ForwardVerificationBatch(chunks);
    for (std::size_t index = 0; index < active; ++index) {
      const auto row = (index + rotation) % active;
      CheckVerificationFeatures(
          *sessions[row], std::span(expected_features[row]).subspan(offset, 2));
      for (std::size_t local = 0; local < 2; ++local) {
        const auto& expected = expected_logits[row][offset + local];
        Expect(
            ByteEqual(sessions[row]->CopyVerificationLogits(local), expected) &&
                predictions[index][local] ==
                    static_cast<Token>(std::ranges::max_element(expected) -
                                       expected.begin()),
            "appended verification chunk changed logits or hidden state");
      }
    }
    for (std::size_t row = active / 2; row < active; ++row) {
      const auto position = items[row].position;
      const auto committed = offset + 1;
      auto& session = *sessions[row];
      session.RestoreState();
      session.CommitVerificationChunk(
          std::span(tokens[row]).subspan(position, committed), position);
      (void)session.ForwardToken(tokens[row][position + committed],
                                 position + committed);
      const auto actual = Logits(session);
      if (!ByteEqual(actual, expected_logits[row][committed])) {
        const auto difference = gufo::testing::CompareLogits(
            actual, expected_logits[row][committed]);
        std::cerr << "chunk replay stage=" << stage << " row=" << row
                  << " committed=" << committed
                  << " max_abs=" << difference.max_abs_diff << '\n';
      }
      Expect(ByteEqual(actual, expected_logits[row][committed]) &&
                 ByteEqual(session.CopyLastHidden(),
                           expected_features[row][committed]),
             "rollback across verification chunks changed continuation state");
    }
  }
  std::cout << "appended verification: C8/4/2/1 logits/hidden/replay exact\n";
}

void CheckWideCache(Executor& reference) {
  // Exercise a full-length context without filling it.
  Executor wide(reference.GetSharedModel(), 262144);
  auto tokens = reference.GetTokenizer().Encode(kTexts[0]);
  constexpr std::size_t prefix = 24;
  constexpr std::size_t continuation = 3;
  Expect(tokens.size() >= prefix + continuation,
         "wide-cache fixture has too few tokens");
  reference.Reset();
  (void)reference.ForwardPromptBatch(std::span(tokens).first(prefix));
  auto expected_prefill = Logits(reference);
  std::array<std::vector<float>, continuation> expected;
  for (std::size_t step = 0; step < continuation; ++step) {
    (void)reference.ForwardToken(tokens[prefix + step], prefix + step);
    expected[step] = Logits(reference);
  }
  wide.Reset();
  (void)wide.ForwardPromptBatch(std::span(tokens).first(prefix));
  Expect(ByteEqual(Logits(wide), expected_prefill),
         "wide-cache prefill changed target logits");
  wide.SaveState(prefix);
  for (std::size_t step = 0; step < continuation; ++step) {
    (void)wide.ForwardToken(tokens[prefix + step], prefix + step);
    Expect(ByteEqual(Logits(wide), expected[step]),
           "wide-cache scalar decode changed target logits");
  }
  wide.RestoreState();
  const auto predictions = wide.ForwardVerificationChunk(
      std::span(tokens).subspan(prefix, continuation), prefix, true);
  Expect(predictions.size() == continuation,
         "wide-cache verifier returned the wrong number of rows");
  for (std::size_t step = 0; step < continuation; ++step) {
    Expect(ByteEqual(wide.CopyVerificationLogits(step), expected[step]),
           "wide-cache verification changed target logits");
  }
  std::cout
      << "context=262144: prefill, scalar and verification logits exact\n";
}

std::optional<gufo::core::GgmlType> FindCpuUnsupportedType(
    const gufo::models::QwenModelWeights& weights) {
  std::optional<gufo::core::GgmlType> unsupported;
  const auto inspect = [&](const gufo::models::QwenTensorRef& tensor) {
    if (!unsupported && !tensor.empty() &&
        !gufo::quant::IsSupported(tensor.type))
      unsupported = tensor.type;
  };
  inspect(weights.token_embd);
  inspect(weights.output_norm);
  inspect(weights.output);
  for (const auto& layer : weights.layers) {
    for (const auto* tensor :
         {&layer.attn_norm,   &layer.ffn_norm,    &layer.attn_q,
          &layer.attn_k,      &layer.attn_v,      &layer.attn_output,
          &layer.attn_q_norm, &layer.attn_k_norm, &layer.attn_qkv,
          &layer.attn_gate,   &layer.ssm_a,       &layer.ssm_conv1d,
          &layer.ssm_dt,      &layer.ssm_alpha,   &layer.ssm_beta,
          &layer.ssm_norm,    &layer.ssm_out,     &layer.ffn_gate,
          &layer.ffn_up,      &layer.ffn_down})
      inspect(*tensor);
  }
  return unsupported;
}

// Teacher forcing on the CPU checks consistency between two implementations
// of the same loader and tensor conventions. It is not an independent quality
// oracle; the reference mode and external references measure quality.
void CheckCpuReference(const gufo::core::GgufReader& reader,
                       std::span<const Case> cases) {
  std::string error;
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(reader, &error);
  Expect(weights.has_value(), error);
  if (const auto type = FindCpuUnsupportedType(*weights)) {
    std::cout << "CPU reference skipped: unsupported tensor format "
              << gufo::core::ToString(*type) << " (not a pass)\n";
    SkippedChecks().push_back("cpu-reference (" +
                              std::string(gufo::core::ToString(*type)) + ")");
    return;
  }
  const auto& config = weights->config;
  gufo::models::QwenKvCache kv_cache(config.FullAttentionLayerCount(),
                                     config.num_key_value_heads,
                                     kCpuReferenceContext, config.head_dim);
  gufo::models::QwenSsmCache ssm_cache(
      config.num_layers, config.SsmQkvSize(), config.ssm_conv_kernel,
      config.ssm_time_step_rank, config.ssm_state_size, config.SsmValueSize());
  gufo::models::QwenScratchArena arena(config);
  std::vector<float> logits(config.vocab_size);
  double total_kl = 0, max_kl = 0;
  std::size_t rows = 0, top1 = 0;
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const auto& history = cases[c];
    Expect(history.tokens.size() <= kCpuReferenceContext,
           "CPU reference history exceeds its cache");
    kv_cache.Reset();
    ssm_cache.Reset();
    for (std::size_t pos = 0; pos < history.tokens.size(); ++pos) {
      gufo::models::ForwardModel(history.tokens[pos],
                                 static_cast<std::uint32_t>(pos), *weights,
                                 kv_cache, ssm_cache, arena, logits);
      if (pos + 1 < history.prompt_size)
        continue;
      const std::size_t row = pos + 1 - history.prompt_size;
      Expect(row < history.logits.size(), "CPU reference row is missing");
      const auto& gpu = history.logits[row];
      const auto comparison = gufo::testing::CompareLogits(logits, gpu);
      const auto divergence = MeasureDivergence(logits, gpu);
      const bool near_tie =
          comparison.finite &&
          logits[comparison.candidate_argmax] >=
              logits[comparison.reference_argmax] - kCpuReferenceTieMargin;
      std::cout << "cpu reference case=" << c << " row=" << row
                << " top1_match=" << comparison.top1_match
                << " kl=" << divergence.kl << " tv=" << divergence.tv
                << " rmse=" << comparison.root_mean_square_error
                << " max_abs=" << comparison.max_abs_diff << '\n';
      Expect(comparison.finite && (comparison.top1_match || near_tie),
             "CPU and GPU target choices differ beyond a near-tie");
      Expect(divergence.kl <= kMaxCpuReferenceKl,
             "CPU and GPU logits exceed the provisional KL limit");
      total_kl += divergence.kl;
      max_kl = std::max(max_kl, divergence.kl);
      top1 += comparison.top1_match;
      ++rows;
    }
  }
  std::cout << "cpu reference rows=" << rows << " top1=" << top1
            << " mean_kl=" << total_kl / rows << " max_kl=" << max_kl << '\n';
}

void CheckChunkInvariance(Executor& executor,
                          const gufo::hip::QwenGpuSnapshot& prefix,
                          std::span<const Token> prompt, std::uint32_t depth,
                          std::span<const float> expected_logits,
                          const std::string& features) {
  const auto length = static_cast<std::uint32_t>(prompt.size());
  executor.RestoreSnapshot(prefix);
  (void)executor.ForwardPromptBatch(prompt, depth);
  Expect(ByteEqual(Logits(executor), expected_logits) &&
             Fingerprint(executor.GetPromptHiddenStates()) == features,
         "repeated long prefill changed logits or hidden taps");
  for (const std::uint32_t budget : {512U, 2048U}) {
    executor.RestoreSnapshot(prefix);
    std::vector<float> captured;
    for (std::uint32_t offset = 0; offset < length; offset += budget) {
      const auto count = std::min(budget, length - offset);
      (void)executor.ForwardPromptBatch(prompt.subspan(offset, count),
                                        depth + offset);
      const auto part = executor.GetPromptHiddenStates();
      captured.insert(captured.end(), part.begin(), part.end());
    }
    Expect(ByteEqual(Logits(executor), expected_logits) &&
               Fingerprint(captured) == features,
           "scheduled prefill chunks changed logits or hidden taps");
    std::cout << "prefill chunk depth=" << depth << " budget=" << budget
              << ": full logits and all tap rows exact\n";
  }
}

void CheckLongPrefill(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model) {
  std::string error;
  auto executor = Executor::Create(model, &error, kLongContext);
  Expect(executor != nullptr, error);
  const auto text = executor->GetTokenizer().Encode(
      std::string(kTexts[0]) + "\n" + kTexts[1] + "\n" + kTexts[2]);
  Expect(!text.empty(), "long-prefill fixture must tokenize");
  const std::size_t width = executor->GetConfig().hidden_size;
  const std::size_t row_width = kCrossPathTaps.size() * width;
  // Mirror the 27B qualification: chunk-exact prefill is established for Q8_0
  // FFN weights only. The 27B FP16 large-prefill route does not match this
  // shape, but other formats remain unqualified until measured.
  const bool chunk_exact = HasChunkExactProjections(model->GetWeights());
  for (const auto& [depth, length] : kLongPrefillCases) {
    const std::uint32_t end = depth + length;
    // A cyclic history lets top-1 follow a copied continuation even with stale
    // attention, so acceptance also bounds hidden taps and KL.
    std::vector<Token> tokens(end);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = text[i % text.size()];
    const auto prompt = std::span<const Token>(tokens).subspan(depth);
    executor->Reset();
    executor->SetPromptHiddenCapture(false);
    std::unique_ptr<gufo::hip::QwenGpuSnapshot> prefix;
    if (depth != 0) {
      (void)executor->ForwardPromptBatch(std::span(tokens).first(depth));
      prefix = executor->SaveSnapshot(depth);
      Expect(prefix != nullptr, "deep prefill snapshot");
    }
    executor->SetPromptHiddenCapture(true, kCrossPathTaps);

    // (A) One prefill over every prompt token.
    (void)executor->ForwardPromptBatch(prompt, depth);
    const auto full_logits = Logits(*executor);
    const auto captured = executor->GetPromptHiddenStates();
    Expect(captured.size() == length * row_width,
           "prefill must capture every requested tap row");
    const std::vector<float> full_last_row(captured.end() - row_width,
                                           captured.end());
    const auto features = Fingerprint(captured);
    std::cout << "prefill fingerprint depth=" << depth << " tokens=" << length
              << " logits=" << Fingerprint(full_logits)
              << " features=" << features << '\n';
    if (depth == kChunkDepth) {
      if (chunk_exact) {
        CheckChunkInvariance(*executor, *prefix, prompt, depth, full_logits,
                             features);
      } else {
        std::cout << "prefill chunk check skipped: a projection format "
                     "outside the native WMMA prefill route (not a pass)\n";
        SkippedChecks().push_back("chunk-invariance");
      }
    }

    // (B) Prefill all but the last token, then decode it.
    if (prefix)
      executor->RestoreSnapshot(*prefix);
    else
      executor->Reset();
    (void)executor->ForwardPromptBatch(prompt.first(length - 1), depth);
    (void)executor->ForwardToken(prompt.back(), end - 1);
    const auto decode_logits = Logits(*executor);
    const auto hidden = executor->CopyLastHidden();
    const std::vector<float> decode_row(hidden.begin(), hidden.end());
    Expect(decode_row.size() == row_width,
           "decode must capture every requested tap");

    std::cout << "cross-path depth=" << depth << " tokens=" << length;
    bool taps_within = true;
    for (std::size_t tap = 0; tap < kCrossPathTaps.size(); ++tap) {
      const double error =
          RelativeL2(std::span(full_last_row).subspan(tap * width, width),
                     std::span(decode_row).subspan(tap * width, width));
      taps_within = taps_within && error <= kMaxTapRelativeL2[tap];
      std::cout << " rel_l2[" << kCrossPathTaps[tap] << "]=" << error;
    }
    const auto divergence = MeasureDivergence(full_logits, decode_logits);
    const auto comparison =
        gufo::testing::CompareLogits(full_logits, decode_logits);
    std::cout << " kl=" << divergence.kl << " tv=" << divergence.tv
              << " top1_match=" << comparison.top1_match
              << " rmse=" << comparison.root_mean_square_error
              << " max_abs=" << comparison.max_abs_diff << '\n';
    Expect(comparison.finite && comparison.top1_match,
           "prefill/decode target choice differs");
    Expect(taps_within,
           "prefill/decode hidden taps exceed the provisional relative L2 "
           "limits");
    Expect(divergence.kl <= kMaxCrossPathKl,
           "prefill/decode logits exceed the provisional KL limit");
  }
}

enum class CheckMode { kAll, kPrefill, kConsistency, kReference };

std::vector<Case> Capture(const char* path, CheckMode mode) {
  std::string error;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, error);
  auto executor = Executor::CreateFromGguf(reader, &error, 128);
  Expect(executor != nullptr, error);
  CheckIdentity(executor->GetConfig());
  if (mode == CheckMode::kPrefill) {
    CheckLongPrefill(executor->GetSharedModel());
    return {};
  }
  const bool check = mode != CheckMode::kReference;
  // MTP verification captures the final hidden state without layer taps.
  if (check)
    executor->SetPromptHiddenCapture(true);
  std::vector<Case> cases;
  for (const auto* text : kTexts) {
    // The middle case crosses the 16-row replay ring after three tokens.
    Case row{.tokens = executor->GetTokenizer().Encode(text),
             .logits = {},
             .prompt_size = cases.size() == 1 ? 29U : 24U};
    const std::size_t prompt_size = row.prompt_size;
    // Exercise the widest verification block.
    constexpr std::size_t continuation = 8;
    Expect(row.tokens.size() >= prompt_size + continuation,
           "quality fixture has too few tokens");
    row.tokens.resize(prompt_size + continuation);
    executor->Reset();
    (void)executor->ForwardPromptBatch(
        std::span(row.tokens).first(prompt_size));
    row.logits.push_back(Logits(*executor));
    std::vector<std::vector<float>> expected_features;
    auto snapshot = executor->SaveSnapshot(prompt_size);
    for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
      (void)executor->ForwardToken(row.tokens[pos], pos);
      row.logits.push_back(Logits(*executor));
      if (check) {
        const auto features = executor->CopyLastHidden();
        Expect(
            features.size() == executor->GetConfig().hidden_size &&
                std::ranges::all_of(
                    features, [](float value) { return std::isfinite(value); }),
            "scalar target hidden state must be finite and complete");
        expected_features.emplace_back(features.begin(), features.end());
      }
    }
    if (check) {
      executor->RestoreSnapshot(*snapshot);
      const auto suffix = std::span(row.tokens).subspan(prompt_size);
      // Speculative decoding verifies every width. Reuse one scalar oracle
      // and snapshot instead of loading another model or adding a suite.
      if (cases.empty()) {
        for (std::size_t width = 2; width < suffix.size(); ++width) {
          executor->RestoreSnapshot(*snapshot);
          const auto predictions = executor->ForwardVerificationChunk(
              suffix.first(width), prompt_size, true);
          Expect(predictions.size() == width, "verification width mismatch");
          for (std::size_t index = 0; index < width; ++index) {
            Expect(ByteEqual(executor->CopyVerificationLogits(index),
                             row.logits[index + 1]),
                   "adaptive-width verification changed target logits");
          }
          CheckVerificationFeatures(*executor,
                                    std::span(expected_features).first(width));
          std::cout << "verification width=" << width << " exact=1\n";
        }
        executor->RestoreSnapshot(*snapshot);
      }
      executor->SaveState(prompt_size);
      const auto predictions =
          executor->ForwardVerificationChunk(suffix, prompt_size, true);
      Expect(predictions.size() == suffix.size(),
             "verification returned the wrong number of rows");
      for (std::size_t index = 0; index < suffix.size(); ++index) {
        const auto logits = executor->CopyVerificationLogits(index);
        const auto comparison =
            gufo::testing::CompareLogits(logits, row.logits[index + 1]);
        std::cout << "verification case=" << cases.size() << " row=" << index
                  << " exact=" << ByteEqual(logits, row.logits[index + 1])
                  << " max_abs=" << comparison.max_abs_diff << '\n';
        Expect(ByteEqual(logits, row.logits[index + 1]),
               "batched verification changed target logits");
      }
      CheckVerificationFeatures(*executor, expected_features);
      std::cout << "verification hidden case=" << cases.size()
                << " rows=" << suffix.size() << " exact=1\n";
      executor->RestoreState();
      constexpr std::size_t committed = 5;
      executor->CommitVerificationChunk(suffix.first(committed), prompt_size);
      for (std::size_t index = committed; index < suffix.size(); ++index) {
        (void)executor->ForwardToken(suffix[index], prompt_size + index);
        Expect(
            ByteEqual(Logits(*executor), row.logits[index + 1]) &&
                ByteEqual(executor->CopyLastHidden(), expected_features[index]),
            "rejected draft replay changed target logits or hidden state");
      }
      executor->RestoreSnapshot(*snapshot);
      for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos);
        Expect(ByteEqual(Logits(*executor), row.logits[pos - prompt_size + 1]),
               "snapshot continuation changed target logits");
      }
      executor->Reset();
      (void)executor->ForwardPromptBatch(
          std::span(row.tokens).first(prompt_size));
      Expect(ByteEqual(Logits(*executor), row.logits.front()),
             "repeated prefill changed target logits");
      executor->Reset();
      for (std::size_t pos = 0; pos < prompt_size; ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos,
                                     pos + 1 == prompt_size);
      }
      const auto scalar = Logits(*executor);
      const auto comparison =
          gufo::testing::CompareLogits(scalar, row.logits.front());
      std::cout << "prefill case=" << cases.size()
                << " top1_match=" << comparison.top1_match
                << " kl=" << MeasureDivergence(scalar, row.logits.front()).kl
                << " rmse=" << comparison.root_mean_square_error
                << " max_abs=" << comparison.max_abs_diff << '\n';
      Expect(comparison.finite && comparison.top1_match,
             "prefill/scalar target choice differs");
    }
    cases.push_back(std::move(row));
  }
  if (!check)
    return cases;
  CheckMixedContextBatch(*executor, false);
  CheckMixedContextBatch(*executor, true);
  CheckConcurrencyWidths(*executor);
  if (mode == CheckMode::kConsistency)
    return cases;
  CheckCpuReference(*reader, cases);
  CheckWideCache(*executor);
  CheckLongPrefill(executor->GetSharedModel());
  return cases;
}

void CompareReference(const std::vector<Case>& candidate,
                      const std::vector<Case>& reference) {
  Expect(candidate.size() == reference.size(), "reference case count differs");
  double total_kl = 0, total_tv = 0, total_nll_delta = 0;
  std::size_t rows = 0, labels = 0, top1 = 0;
  for (std::size_t c = 0; c < reference.size(); ++c) {
    Expect(candidate[c].tokens == reference[c].tokens,
           "reference and candidate tokenization differ");
    for (std::size_t r = 0; r < reference[c].logits.size(); ++r) {
      const auto& p = reference[c].logits[r];
      const auto& q = candidate[c].logits[r];
      const auto divergence = MeasureDivergence(p, q);
      const auto cmp = gufo::testing::CompareLogits(p, q);
      total_kl += divergence.kl;
      total_tv += divergence.tv;
      top1 += cmp.top1_match;
      ++rows;
      if (r + 1 < reference[c].logits.size()) {
        const auto label = reference[c].tokens[reference[c].prompt_size + r];
        total_nll_delta +=
            (LogNormalizer(q) - q[label]) - (LogNormalizer(p) - p[label]);
        ++labels;
      }
      std::cout << "reference case=" << c << " row=" << r
                << " top1_match=" << cmp.top1_match << " kl=" << divergence.kl
                << " tv=" << divergence.tv
                << " rmse=" << cmp.root_mean_square_error << '\n';
    }
  }
  // These are measurements, not a claim that different quantizations must
  // produce identical distributions or that the reference is independent truth.
  std::cout << "reference rows=" << rows << " top1=" << top1
            << " mean_kl=" << total_kl / rows << " mean_tv=" << total_tv / rows
            << " mean_nll_delta=" << total_nll_delta / labels << '\n';
}
}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN35_9B_MODEL");
    if (model == nullptr) {
      std::cout << "Set GUFO_QWEN35_9B_MODEL for the Qwen3.5-9B target "
                   "check.\n";
      return 77;
    }
    Expect(argc <= 3,
           "usage: qwen35_9b_target_test [model.gguf [--prefill-only | "
           "--consistency-only | reference.gguf]]");
    const std::string_view option = argc == 3 ? argv[2] : "";
    if (option == "--prefill-only") {
      (void)Capture(model, CheckMode::kPrefill);
    } else if (option == "--consistency-only") {
      (void)Capture(model, CheckMode::kConsistency);
    } else if (option.starts_with("--")) {
      throw std::invalid_argument("unknown option " + std::string(option));
    } else if (!option.empty()) {
      // Reference mode measures quantization on fixed histories. Load one
      // model at a time: a reference check must not require both targets to
      // remain resident in unified memory.
      const auto candidate = Capture(model, CheckMode::kReference);
      CompareReference(candidate, Capture(argv[2], CheckMode::kReference));
    } else {
      (void)Capture(model, CheckMode::kAll);
    }
    if (SkippedChecks().empty()) {
      std::cout << "Qwen3.5-9B target checks passed.\n";
    } else {
      std::cout << "Qwen3.5-9B target checks passed; skipped (not passes):";
      for (const auto& check : SkippedChecks())
        std::cout << ' ' << check;
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

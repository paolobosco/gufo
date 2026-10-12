#include "src/models/qwen/hip/detail/attention_policy.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

enum class Attempt {
  kTiled,
  kBaseline,
};

int failures{0};

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void TestDispatchThresholdAndFallbackOrder() {
  using gufo::hip::detail::DispatchPrefillAttention;
  for (const std::size_t length : {1023U, 1024U}) {
    for (const bool supported : {false, true}) {
      std::vector<Attempt> attempts;
      DispatchPrefillAttention(
          length,
          [&] {
            attempts.push_back(Attempt::kTiled);
            return supported;
          },
          [&] { attempts.push_back(Attempt::kBaseline); });
      const auto expected =
          length < 1024 ? std::vector{Attempt::kBaseline}
          : supported   ? std::vector{Attempt::kTiled}
                        : std::vector{Attempt::kTiled, Attempt::kBaseline};
      Check(attempts == expected, "threshold and unsupported-shape fallback");
    }
  }
}

void TestBackendSupportPredicates() {
  using gufo::hip::detail::AttentionSupportParams;
  using gufo::hip::detail::IsTiledAttentionSupported;
  const AttentionSupportParams valid{.batch_size = 1024,
                                     .start_pos = 0,
                                     .max_context = 4096,
                                     .num_heads = 24,
                                     .num_kv_heads = 4,
                                     .head_dim = 256,
                                     .has_k_cache_f16 = true,
                                     .has_v_cache_f16 = true};
  Check(IsTiledAttentionSupported(valid),
        "supported native attention geometry");
  auto params = valid;
  params.start_pos = 3072;
  Check(IsTiledAttentionSupported(params),
        "cached prefix reaches exact capacity");
  params.start_pos = 3073;
  Check(!IsTiledAttentionSupported(params), "cached prefix exceeds capacity");
  params = valid;
  params.num_heads = 16;
  Check(!IsTiledAttentionSupported(params), "unsupported query-head count");
  params = valid;
  params.num_kv_heads = 8;
  Check(!IsTiledAttentionSupported(params), "unsupported KV-head count");
  params = valid;
  params.head_dim = 128;
  Check(!IsTiledAttentionSupported(params), "unsupported head dimension");
  params = valid;
  params.batch_size = 0;
  Check(!IsTiledAttentionSupported(params), "empty batch is rejected");
  params = valid;
  params.batch_size = 4096;
  Check(IsTiledAttentionSupported(params), "full capacity is accepted");
  params.batch_size = 4097;
  Check(!IsTiledAttentionSupported(params), "overflowing batch is rejected");
  params = valid;
  params.has_k_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params), "missing K cache is rejected");
  params = valid;
  params.has_v_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params), "missing V cache is rejected");
}

void TestWmmaSupportPredicate() {
  using gufo::hip::detail::AttentionSupportParams;
  using gufo::hip::detail::IsTiledAttentionSupported;
  using gufo::hip::detail::IsWmmaAttentionSupported;
  const AttentionSupportParams valid{.batch_size = 1024,
                                     .start_pos = 0,
                                     .max_context = 4096,
                                     .num_heads = 24,
                                     .num_kv_heads = 4,
                                     .head_dim = 256,
                                     .has_k_cache_f16 = true,
                                     .has_v_cache_f16 = true};
  for (std::uint32_t heads = 0; heads <= 64; ++heads) {
    auto params = valid;
    params.num_heads = heads;
    Check(IsWmmaAttentionSupported(params) == (heads == 16 || heads == 24),
          "WMMA accepts exactly the 16- and 24-head instances");
    Check(IsTiledAttentionSupported(params) == (heads == 24),
          "the tiled kernel stays 24-head only");
  }
  for (const std::uint32_t heads : {16U, 24U}) {
    auto params = valid;
    params.num_heads = heads;
    params.start_pos = 3072;
    Check(IsWmmaAttentionSupported(params),
          "WMMA cached prefix reaches exact capacity");
    params.start_pos = 3073;
    Check(!IsWmmaAttentionSupported(params),
          "WMMA cached prefix exceeds capacity");
    params = valid;
    params.num_heads = heads;
    params.num_kv_heads = 8;
    Check(!IsWmmaAttentionSupported(params), "WMMA rejects 8 KV heads");
    params.num_kv_heads = 2;
    Check(!IsWmmaAttentionSupported(params), "WMMA rejects 2 KV heads");
    params = valid;
    params.num_heads = heads;
    params.head_dim = 128;
    Check(!IsWmmaAttentionSupported(params), "WMMA rejects head_dim 128");
    params = valid;
    params.num_heads = heads;
    params.batch_size = 0;
    Check(!IsWmmaAttentionSupported(params), "WMMA rejects an empty batch");
    params.batch_size = 1;
    Check(IsWmmaAttentionSupported(params),
          "WMMA has no batch threshold of its own");
    params = valid;
    params.num_heads = heads;
    params.has_k_cache_f16 = false;
    Check(!IsWmmaAttentionSupported(params), "WMMA requires the FP16 K cache");
    params = valid;
    params.num_heads = heads;
    params.has_v_cache_f16 = false;
    Check(!IsWmmaAttentionSupported(params), "WMMA requires the FP16 V cache");
  }
}

void TestBaselinePrefillLdsLimit() {
  using gufo::hip::detail::BaselinePrefillAttentionLdsBytes;
  using gufo::hip::detail::IsBaselinePrefillAttentionSupported;
  using gufo::hip::detail::kBatchedAttentionScratchFloats;
  using gufo::hip::detail::kGfx1151MaxWorkgroupLdsBytes;
  Check(kGfx1151MaxWorkgroupLdsBytes == 65536, "gfx1151 workgroup LDS");
  Check(BaselinePrefillAttentionLdsBytes(0) ==
            kBatchedAttentionScratchFloats * sizeof(float),
        "baseline static scratch is 1 KiB");
  Check(BaselinePrefillAttentionLdsBytes(16128) == 65536,
        "16128 visible scores fill the workgroup LDS exactly");
  Check(IsBaselinePrefillAttentionSupported(16128),
        "baseline accepts 16128 visible keys");
  Check(!IsBaselinePrefillAttentionSupported(16129),
        "baseline rejects 16129 visible keys");
  Check(IsBaselinePrefillAttentionSupported(1), "baseline accepts one key");
}

void TestPrefillAttentionGuard() {
  using gufo::hip::detail::AttentionSupportParams;
  using gufo::hip::detail::IsPrefillAttentionSupported;
  // A prompt ending at the given position, as ForwardPromptBatch checks it.
  const auto prompt = [](std::uint32_t heads, bool fp16_kv,
                         std::uint32_t start_pos, std::size_t tokens) {
    return AttentionSupportParams{.batch_size = tokens,
                                  .start_pos = start_pos,
                                  .max_context = 262144,
                                  .num_heads = heads,
                                  .num_kv_heads = 4,
                                  .head_dim = 256,
                                  .has_k_cache_f16 = fp16_kv,
                                  .has_v_cache_f16 = fp16_kv};
  };
  for (const std::uint32_t heads : {16U, 24U}) {
    // Production FP16 KV always takes WMMA, so no prompt length is rejected.
    for (const std::size_t end : {std::size_t{1}, std::size_t{16128},
                                  std::size_t{16129}, std::size_t{262144}}) {
      Check(IsPrefillAttentionSupported(prompt(heads, true, 0, end)),
            "FP16 KV prefill is never limited by baseline LDS");
      Check(IsPrefillAttentionSupported(
                prompt(heads, true, static_cast<std::uint32_t>(end - 1), 1)),
            "FP16 KV continuation is never limited by baseline LDS");
    }
    // FP32 KV reaches the baseline for every chunk.
    Check(IsPrefillAttentionSupported(prompt(heads, false, 0, 16128)),
          "FP32 KV prompt ending at 16128 fits the baseline");
    Check(!IsPrefillAttentionSupported(prompt(heads, false, 0, 16129)),
          "FP32 KV prompt ending at 16129 is rejected");
    Check(IsPrefillAttentionSupported(prompt(heads, false, 16000, 128)),
          "FP32 KV continuation ending at 16128 fits the baseline");
    Check(!IsPrefillAttentionSupported(prompt(heads, false, 16128, 1)),
          "FP32 KV continuation ending at 16129 is rejected");
  }
  // Unsupported geometry falls back to the baseline even with FP16 KV.
  Check(IsPrefillAttentionSupported(prompt(32, true, 0, 16128)),
        "unsupported heads fit the baseline up to 16128 keys");
  Check(!IsPrefillAttentionSupported(prompt(32, true, 0, 16129)),
        "unsupported heads are rejected beyond the baseline LDS");
}

void TestDecodeSplitPolicy() {
  using gufo::hip::detail::DecodeAttentionScratchElements;
  using gufo::hip::detail::IsFusedQkNormSupported;
  using gufo::hip::detail::IsSplitKDecodeAttentionSupported;
  using gufo::hip::detail::SelectDecodeAttentionSplitCount;

  Check(SelectDecodeAttentionSplitCount(1) == 1,
        "the first decode token uses one online-softmax partition");
  Check(SelectDecodeAttentionSplitCount(127) == 1,
        "decode stays on one partition below 128 tokens");
  Check(SelectDecodeAttentionSplitCount(128) == 32,
        "128-token decode uses parallel attention partitions");
  Check(SelectDecodeAttentionSplitCount(2048) == 32,
        "2K decode uses parallel attention partitions");
  Check(SelectDecodeAttentionSplitCount(8192) == 32,
        "8K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(16384) == 32,
        "16K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(32768) == 32,
        "32K decode caps the partition count");

  Check(IsSplitKDecodeAttentionSupported(4096, 24, 4, 256),
        "Qwen3.8 long-context decode shape supports split-K");
  Check(IsSplitKDecodeAttentionSupported(2048, 24, 4, 256),
        "Qwen3.8 shallow-context decode shape supports split-K");
  Check(!IsSplitKDecodeAttentionSupported(127, 24, 4, 256),
        "very short decode does not use split-K");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 0, 256),
        "split-K rejects zero KV heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 22, 4, 256),
        "split-K rejects non-divisible GQA heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 4, 128),
        "split-K rejects unsupported head dimensions");
  Check(DecodeAttentionScratchElements(24, 256) ==
            static_cast<std::size_t>(24 * 32 * 258),
        "split-K scratch layout covers stats and partial values");
  Check(IsFusedQkNormSupported(256),
        "fused Q/K norm accepts its shared-memory capacity");
  Check(!IsFusedQkNormSupported(257),
        "fused Q/K norm rejects larger head dimensions");
}

}  // namespace

int main() {
  TestDispatchThresholdAndFallbackOrder();
  TestBackendSupportPredicates();
  TestWmmaSupportPredicate();
  TestBaselinePrefillLdsLimit();
  TestPrefillAttentionGuard();
  TestDecodeSplitPolicy();
  if (failures != 0) {
    std::cerr << failures << " Qwen attention policy test(s) failed.\n";
    return 1;
  }
  std::cout << "Qwen attention policy tests passed.\n";
  return 0;
}

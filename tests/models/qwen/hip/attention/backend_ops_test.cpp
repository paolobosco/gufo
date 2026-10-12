#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

void TestBatchedAttentionEquivalence() {
  constexpr std::size_t batch = 2048;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 32;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;
  const std::size_t cache_size = 8 * num_kv_heads * max_context * head_dim * 2;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] = 0.08F * std::sin(static_cast<float>(index % 257) * 0.07F);
    h_gate[index] = 0.4F * std::cos(static_cast<float>(index % 193) * 0.05F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] = 0.07F * std::cos(static_cast<float>(index % 251) * 0.06F);
    h_v[index] = 0.2F * std::sin(static_cast<float>(index % 239) * 0.04F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_batch = nullptr,
        *d_cache_gemm = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr, *d_out_gemm = nullptr;
  float* d_scores = nullptr;
  hipblasHandle_t hipblas_handle = nullptr;

  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_batch, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_gemm, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_gemm, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scores, (num_heads / num_kv_heads) * batch *
                                     max_context * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_batch, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_gemm, 0, cache_size * sizeof(float)));

  const std::size_t total_k = 8 * num_kv_heads * max_context * head_dim;

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    gufo::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k, nullptr, nullptr,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  gufo::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache_batch,
                                    d_cache_batch + total_k, nullptr, nullptr,
                                    d_out_batch, 0, 0, batch, max_context,
                                    num_heads, num_kv_heads, head_dim);
  gufo::hip::LaunchBatchedAttentionGemm(
      hipblas_handle, d_q, d_k, d_v, d_gate, d_cache_gemm,
      d_cache_gemm + total_k, nullptr, nullptr, d_scores, d_out_gemm, 0, 0,
      batch, max_context, num_heads, num_kv_heads, head_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(q_size), res_batch(q_size), res_gemm(q_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_gemm.data(), d_out_gemm, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  float max_gemm_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
    const float gemm_diff = std::abs(res_seq[i] - res_gemm[i]);
    if (gemm_diff > max_gemm_diff)
      max_gemm_diff = gemm_diff;
  }
  std::cout << "Attention Seq vs Batch max diff: " << max_diff << "\n";
  std::cout << "Attention Seq vs GEMM max diff: " << max_gemm_diff << "\n";
  if (max_diff >= 1e-4F || max_gemm_diff >= 1e-4F) {
    std::cerr << "Batched attention mismatch\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_batch));
  HIP_CHECK(hipFree(d_cache_gemm));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_gemm));
  HIP_CHECK(hipFree(d_scores));
}

// The baseline keeps one FP32 score per visible key beside its static
// scratch, so 16128 keys fill the 64 KiB gfx1151 workgroup LDS exactly. That
// launch must run; one more key must be rejected before the KV write instead
// of being dropped by the launch and leaving stale output.
void TestBatchedAttentionLdsLimit() {
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 32;
  constexpr std::uint32_t limit = 16128;
  static_assert(gufo::hip::detail::IsBaselinePrefillAttentionSupported(limit));
  static_assert(
      !gufo::hip::detail::IsBaselinePrefillAttentionSupported(limit + 1));
  constexpr std::uint32_t max_context = limit + 1;
  constexpr std::size_t q_size = std::size_t{num_heads} * head_dim;
  constexpr std::size_t kv_size = std::size_t{num_kv_heads} * head_dim;
  constexpr std::size_t cache_elements =
      std::size_t{num_kv_heads} * max_context * head_dim;

  std::vector<float> h_q(q_size), h_gate(q_size), h_k(kv_size), h_v(kv_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.3F * std::sin(static_cast<float>(i + 1) * 0.37F);
    h_gate[i] = 0.5F * std::cos(static_cast<float>(i + 1) * 0.29F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.4F * std::cos(static_cast<float>(i + 1) * 0.31F);
    h_v[i] = 0.6F * std::sin(static_cast<float>(i + 1) * 0.23F);
  }
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  // Every cached key and value is the float with bytes 0x3C3C3C3C.
  constexpr std::uint8_t kCacheByte = 0x3C;
  const float cached = std::bit_cast<float>(0x3C3C3C3CU);
  HIP_CHECK(hipMemset(d_cache, kCacheByte, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out, 0xFF, q_size * sizeof(float)));

  // One query at position limit - 1 sees exactly `limit` keys.
  gufo::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache,
                                    d_cache + cache_elements, nullptr, nullptr,
                                    d_out, 0, limit - 1, 1, max_context,
                                    num_heads, num_kv_heads, head_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out(q_size);
  HIP_CHECK(hipMemcpy(out.data(), d_out, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const float* q = h_q.data() + (h * head_dim);
    double cached_dot = 0;
    double new_dot = 0;
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      cached_dot += double(q[d]) * cached;
      new_dot += double(q[d]) * h_k[d];
    }
    const double scale = 1.0 / std::sqrt(double(head_dim));
    const double top = std::max(cached_dot, new_dot) * scale;
    const double cached_weight =
        double(limit - 1) * std::exp((cached_dot * scale) - top);
    const double new_weight = std::exp((new_dot * scale) - top);
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      double value = ((cached_weight * cached) + (new_weight * h_v[d])) /
                     (cached_weight + new_weight);
      value /= 1 + std::exp(-double(h_gate[(h * head_dim) + d]));
      const float got = out[(h * head_dim) + d];
      max_diff =
          std::isfinite(got)
              ? std::max(max_diff, static_cast<float>(std::abs(got - value)))
              : std::numeric_limits<float>::infinity();
    }
  }
  std::cout << "Baseline attention at the LDS limit max diff: " << max_diff
            << "\n";
  if (!(max_diff < 1e-4F)) {
    std::cerr << "Baseline attention failed at exactly the LDS limit\n";
    std::abort();
  }

  // One more visible key: rejected before the KV write touches the cache.
  std::vector<float> before(2 * cache_elements);
  HIP_CHECK(hipMemcpy(before.data(), d_cache, before.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  bool rejected = false;
  try {
    gufo::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache,
                                      d_cache + cache_elements, nullptr,
                                      nullptr, d_out, 0, limit, 1, max_context,
                                      num_heads, num_kv_heads, head_dim);
  } catch (const std::length_error&) {
    rejected = true;
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> after(before.size());
  HIP_CHECK(hipMemcpy(after.data(), d_cache, after.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  if (!rejected || std::memcmp(before.data(), after.data(),
                               before.size() * sizeof(float)) != 0) {
    std::cerr << "Baseline attention beyond the LDS limit was not rejected "
                 "before the KV write\n";
    std::abort();
  }
  std::cout << "Baseline attention beyond the LDS limit: rejected, KV cache "
               "unchanged\n";

  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out})
    HIP_CHECK(hipFree(p));
}

void TestAttentionBackendEquivalence(std::uint32_t num_heads) {
  constexpr std::size_t batch = 128;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 128;
  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = batch * attention_width;
  const std::size_t kv_size = batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_tile = nullptr,
        *d_cache_wmma = nullptr;
  float *d_out_seq = nullptr, *d_out_tile = nullptr, *d_out_wmma = nullptr;
  void *d_cache_tile_f16 = nullptr, *d_cache_wmma_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_tile, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_wmma, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_cache_tile_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_cache_wmma_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_tile, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_wmma, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_wmma, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_cache_wmma_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));

  std::cout << "Attention backends with " << num_heads << " query heads\n";
  // The tiled kernel exists for 24 query heads only; WMMA covers 16 and 24.
  const bool has_tile = num_heads == 24;
  for (std::size_t token = 0; token < batch; ++token) {
    gufo::hip::LaunchAttention(d_q + token * attention_width,
                               d_k + token * kv_width, d_v + token * kv_width,
                               d_gate + token * attention_width, d_cache_seq,
                               d_cache_seq + cache_elements, nullptr, nullptr,
                               d_out_seq + token * attention_width, 0,
                               static_cast<std::uint32_t>(token), max_context,
                               num_heads, num_kv_heads, head_dim);
  }
  const bool tile_launched =
      !has_tile ||
      gufo::hip::LaunchBatchedAttentionTile(
          d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
          d_cache_tile_f16,
          static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
          d_out_tile, 0, 0, batch, max_context, num_heads, num_kv_heads,
          head_dim);
  if (!tile_launched) {
    std::cerr << "Tiled attention rejected the Qwen shape\n";
    std::abort();
  }
  const bool launched = gufo::hip::LaunchQwenWmmaAttention(
      d_q, d_k, d_v, d_gate, d_cache_wmma, d_cache_wmma + cache_elements,
      d_cache_wmma_f16,
      static_cast<std::uint16_t*>(d_cache_wmma_f16) + cache_elements,
      d_out_wmma, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!launched) {
    std::cerr << "Native WMMA attention rejected the Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> sequential(q_size), tiled(q_size), wmma(q_size);
  std::vector<float> sequential_cache(2 * cache_elements);
  std::vector<float> tile_cache(2 * cache_elements);
  std::vector<float> ck_cache(2 * cache_elements);
  HIP_CHECK(hipMemcpy(sequential.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wmma.data(), d_out_wmma, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sequential_cache.data(), d_cache_seq,
                      sequential_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck_cache.data(), d_cache_wmma,
                      ck_cache.size() * sizeof(float), hipMemcpyDeviceToHost));
  float max_tile_diff = 0.0F;
  float max_wmma_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_tile_diff =
        std::max(max_tile_diff, std::abs(sequential[index] - tiled[index]));
    max_wmma_diff =
        std::max(max_wmma_diff, std::abs(sequential[index] - wmma[index]));
  }
  float max_tile_cache_diff = 0.0F;
  float max_cache_diff = 0.0F;
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_tile_cache_diff =
        std::max(max_tile_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
    max_cache_diff = std::max(
        max_cache_diff, std::abs(sequential_cache[index] - ck_cache[index]));
  }
  if (has_tile) {
    std::cout << "Attention Seq vs tile max diff: " << max_tile_diff << "\n";
    std::cout << "Attention Seq vs tile cache max diff: " << max_tile_cache_diff
              << "\n";
  }
  std::cout << "Attention Seq vs WMMA max diff: " << max_wmma_diff << "\n";
  std::cout << "Attention Seq vs WMMA cache max diff: " << max_cache_diff
            << "\n";
  if (has_tile && (max_tile_diff >= 5e-3F || max_tile_cache_diff != 0.0F)) {
    std::cerr << "Tiled attention mismatch\n";
    std::abort();
  }
  if (max_wmma_diff >= 2e-3F || max_cache_diff != 0.0F) {
    std::cerr << "Native WMMA attention mismatch\n";
    std::abort();
  }

  if (!has_tile) {
    for (float* p : {d_q, d_k, d_v, d_gate, d_cache_seq, d_cache_tile,
                     d_cache_wmma, d_out_seq, d_out_tile, d_out_wmma})
      HIP_CHECK(hipFree(p));
    HIP_CHECK(hipFree(d_cache_tile_f16));
    HIP_CHECK(hipFree(d_cache_wmma_f16));
    return;
  }
  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_out_tile, 0, q_size * sizeof(float)));
  constexpr std::size_t chunk_size = batch / 2;
  const bool first_chunk_launched = gufo::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, chunk_size, max_context, num_heads, num_kv_heads,
      head_dim);
  const bool second_chunk_launched = gufo::hip::LaunchBatchedAttentionTile(
      d_q + chunk_size * attention_width, d_k + chunk_size * kv_width,
      d_v + chunk_size * kv_width, d_gate + chunk_size * attention_width,
      d_cache_tile, d_cache_tile + cache_elements, d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile + chunk_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk_size), chunk_size, max_context,
      num_heads, num_kv_heads, head_dim);
  if (!first_chunk_launched || !second_chunk_launched) {
    std::cerr << "Tiled attention rejected a chunked Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_chunked_diff = 0.0F;
  float max_chunked_cache_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_chunked_diff =
        std::max(max_chunked_diff, std::abs(sequential[index] - tiled[index]));
  }
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_chunked_cache_diff =
        std::max(max_chunked_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
  }
  std::cout << "Attention Seq vs chunked tile max diff: " << max_chunked_diff
            << "\n";
  std::cout << "Attention Seq vs chunked tile cache max diff: "
            << max_chunked_cache_diff << "\n";
  if (max_chunked_diff >= 5e-3F || max_chunked_cache_diff != 0.0F) {
    std::cerr << "Chunked tiled attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_tile));
  HIP_CHECK(hipFree(d_cache_wmma));
  HIP_CHECK(hipFree(d_cache_tile_f16));
  HIP_CHECK(hipFree(d_cache_wmma_f16));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_tile));
  HIP_CHECK(hipFree(d_out_wmma));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen attention backend ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedAttentionEquivalence();
  TestBatchedAttentionLdsLimit();
  TestAttentionBackendEquivalence(24);
  TestAttentionBackendEquivalence(16);
  std::cout << "Qwen attention backend ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention backend ops test.\n";
  return 77;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  if (prompt_tokens.empty()) {
    return 0;
  }
  // Validate the whole range before any state changes, so a rejected prompt
  // leaves the executor exactly as it was.
  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }
  const auto& config = weights_.config;
  if (!detail::IsPrefillAttentionSupported({
          .batch_size = prompt_tokens.size(),
          .start_pos = start_pos,
          .max_context = arena_.GetMaxContext(),
          .num_heads = config.num_attention_heads,
          .num_kv_heads = config.num_key_value_heads,
          .head_dim = config.head_dim,
          .has_k_cache_f16 = policy_.UsesFp16AttentionKv(),
          .has_v_cache_f16 = policy_.UsesFp16AttentionKv(),
      })) {
    throw std::length_error(
        "prompt exceeds the baseline prefill attention LDS limit");
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.clear();
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    h_prompt_hidden_.reserve(prompt_tokens.size() * captured_layers *
                             weights_.config.hidden_size);
  }

  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    next_token =
        ForwardPromptChunk(prompt_tokens.subspan(offset, chunk_size),
                           start_pos + static_cast<std::uint32_t>(offset),
                           compute_logits && is_last);
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

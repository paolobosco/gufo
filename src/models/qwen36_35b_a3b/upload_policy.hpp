#ifndef GUFO_MODELS_QWEN36_35B_A3B_UPLOAD_POLICY_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_UPLOAD_POLICY_HPP_

#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen36_35b_a3b {

/// The routed projections stacked per expert in one MoE block.
enum class ExpertRole : std::uint8_t { kGate, kUp, kDown };

/// Whether the uploader keeps a Q8_0 prefill copy of a routed expert stack.
/// Prefill swaps in the copy of a trunk `ffn_down_exps` only, and both the
/// routed WMMA and the MMQ routes then read it: a Q6_K down projection has no
/// routed WMMA decoder. Gate and up stacks are never swapped, and the MTP
/// block does not run on the prefill layers, so their copies are never read.
[[nodiscard]] constexpr bool NeedsPrefillQ8Copy(ExpertRole role,
                                                core::GgmlType type,
                                                bool is_mtp) noexcept {
  return role == ExpertRole::kDown && type == core::GgmlType::kQ6_K && !is_mtp;
}

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // GUFO_MODELS_QWEN36_35B_A3B_UPLOAD_POLICY_HPP_

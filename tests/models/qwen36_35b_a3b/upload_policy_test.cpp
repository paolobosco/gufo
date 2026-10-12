#include "src/models/qwen36_35b_a3b/upload_policy.hpp"

#include <iostream>

namespace q36 = gufo::models::qwen36_35b_a3b;
using gufo::core::GgmlType;

namespace {

// Prefill swaps in only the trunk down-projection copy, so that is the only
// routed expert copy the uploader may create (Q6_K has no routed WMMA
// decoder). UD-Q6_K_XL stores every down projection as Q8_0: no copies.
static_assert(q36::NeedsPrefillQ8Copy(q36::ExpertRole::kDown, GgmlType::kQ6_K,
                                      false));
static_assert(!q36::NeedsPrefillQ8Copy(q36::ExpertRole::kDown, GgmlType::kQ8_0,
                                       false));
static_assert(!q36::NeedsPrefillQ8Copy(q36::ExpertRole::kGate, GgmlType::kQ6_K,
                                       false));
static_assert(!q36::NeedsPrefillQ8Copy(q36::ExpertRole::kDown, GgmlType::kQ6_K,
                                       true));

const char* Name(q36::ExpertRole role) {
  switch (role) {
    case q36::ExpertRole::kGate:
      return "gate";
    case q36::ExpertRole::kUp:
      return "up";
    case q36::ExpertRole::kDown:
      return "down";
  }
  return "?";
}

}  // namespace

int main() {
  int failures = 0;
  int cases = 0;
  for (const auto role :
       {q36::ExpertRole::kGate, q36::ExpertRole::kUp, q36::ExpertRole::kDown}) {
    for (const auto type :
         {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ8_0}) {
      for (const bool mtp : {false, true}) {
        const bool expected =
            role == q36::ExpertRole::kDown && type == GgmlType::kQ6_K && !mtp;
        ++cases;
        if (q36::NeedsPrefillQ8Copy(role, type, mtp) != expected) {
          ++failures;
          std::cerr << "FAIL: " << Name(role) << ' '
                    << gufo::core::ToString(type) << (mtp ? " MTP" : " trunk")
                    << " copy should be " << (expected ? "kept" : "skipped")
                    << '\n';
        }
      }
    }
  }
  if (failures != 0) {
    return 1;
  }
  std::cout << "Qwen3.6 expert upload policy: " << cases
            << " role/type/block cases passed.\n";
}

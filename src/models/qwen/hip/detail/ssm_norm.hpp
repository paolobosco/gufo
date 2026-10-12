#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_SSM_NORM_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_SSM_NORM_HPP_

#include <hip/hip_runtime.h>

namespace gufo::hip::detail {

// Sum of x * x over a wave of 32 lanes, shared by every SSM post-norm epilogue
// whose output must match the FP32 epilogue bit for bit. The first butterfly
// step adds the partner's rounded square to the exact own square in one fma;
// later steps are plain adds. Spelling the fma out keeps the rounding
// independent of the compiler's contraction choices, which differ between
// kernels from LLVM 23 on.
__device__ __forceinline__ float WarpSquareSum(float x) {
  float sum = __builtin_fmaf(x, x, __shfl_xor(x * x, 16));
  for (int offset = 8; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
  return sum;
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_SSM_NORM_HPP_

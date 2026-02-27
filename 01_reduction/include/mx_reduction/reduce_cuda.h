#pragma once
// #include <cuda_runtime.h>
#include <cstdlib>
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"

namespace mx {

template<typename T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy);

} // namespace mx

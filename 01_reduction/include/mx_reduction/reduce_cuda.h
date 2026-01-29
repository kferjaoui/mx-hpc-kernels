#pragma once
#include <cuda_runtime.h>
#include "cuda_check.h"

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"

namespace mx {

template<typename T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy);

} // namespace mx

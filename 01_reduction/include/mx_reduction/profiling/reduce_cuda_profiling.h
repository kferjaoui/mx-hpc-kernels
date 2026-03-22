#pragma once

#include "mx/utils/policy.h"
#include "mx/utils/operations.h"

namespace mx::profile {

enum class ReduceKernel {
    Baseline,
    Interleaved,
    Sequential,
    WarpShuffle,
    TwoPass
};

template<typename T, class Op>
T reduce_cuda_profiled(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy,
                       ReduceKernel kernel, int warmup_iters, int iters);


} // namespace mx::profile

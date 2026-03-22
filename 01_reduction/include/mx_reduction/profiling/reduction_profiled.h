#pragma once
#include "CycleTimer.h"
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"

#include "mx_reduction/reduce_cpu.h"
#include "mx_reduction/profiling/reduce_cuda_profiling.h"

namespace mx::profile {

template <class> 
inline constexpr bool always_false_v = false;

// Main reduce function (profiled)
template<typename T, class Op, class Policy = CPU<>>
T reduce_profiled(const T* input, size_t size, T init, Op op, Policy policy = Policy{}){

    if constexpr (is_cpu_policy_v<Policy>)
    {   
        double start = CycleTimer::currentSeconds();
        T result = reduce_cpu(input, size, init, op, policy.threads);
        double end = CycleTimer::currentSeconds();
        printf("[CPU (%d threads)] Time: %f seconds\n", policy.threads, end - start);
        return result;
    }
    else if constexpr (is_cuda_policy_v<Policy>)
    {
        return mx::profile::reduce_cuda_profiled(input, size, init, op, policy, 
                                                mx::profile::ReduceKernel::TwoPass,
                                                5, 10);
    }
    else
    {
        static_assert(always_false_v<Policy>, "mx::reduce: unsupported Policy type");
    }
}

} // namespace mx::profile
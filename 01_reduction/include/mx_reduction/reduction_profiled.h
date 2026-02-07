#pragma once
#include "CycleTimer.h"
#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduce_cpu.h"
#include "mx_reduction/reduce_cuda_profiling.h"

namespace mx::profile {

template <class> 
inline constexpr bool always_false_v = false;

// Main reduce function (profiled)
template<typename T, class Op, class Policy>
T reduce_profiled(const T* input, size_t size, T init, Op op, Policy policy){

    if constexpr (std::is_same_v<Policy, CPU>) 
    {   
        double start = CycleTimer::currentSeconds();
        T result = reduce_cpu(input, size, init, op, policy.threads);
        double end = CycleTimer::currentSeconds();
        printf("[CPU (%d threads)] Time: %f seconds\n", policy.threads, end - start);
        return result;
    }
    else if constexpr (std::is_same_v<Policy, CUDA>)
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


// Default overload: CPU policy
template <typename T, class Op>
T reduce_profiled(const T* input, size_t size, T init, Op op){
    CPU policy{};
    return profile::reduce_profiled(input, size, init, op, policy); // Calls the main reduce function ''reduce<T, Op, Policy>''
}

} // namespace mx::profile
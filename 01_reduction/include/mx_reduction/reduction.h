#pragma once

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"

#include "mx_reduction/reduce_cpu.h"

#ifdef __CUDACC__
    #include "mx_reduction/reduce_cuda.cuh"
#endif

namespace mx{

template <class> 
inline constexpr bool always_false_v = false;

// Main reduce function
template<typename T, class Op, class Policy>
T reduce(const T* input, size_t size, T init, Op op, Policy policy){

    if constexpr (std::is_same_v<Policy, CPU>) 
    {
        return reduce_cpu(input, size, init, op, policy.threads);
    }
    #ifdef __CUDACC__
    else if constexpr (std::is_same_v<Policy, CUDA>)
    {
        return reduce_cuda(input, size, init, op, policy);
    }
    #endif
    else
    {
        static_assert(always_false_v<Policy>, "mx::reduce: unsupported Policy type");
    }
}

// Default overload: CPU policy
template <typename T, class Op>
T reduce(const T* input, size_t size, T init, Op op){
    CPU policy{};
    return reduce(input, size, init, op, policy); // Calls the main reduce function ''reduce<T, Op, Policy>''
}

} // namespace mx
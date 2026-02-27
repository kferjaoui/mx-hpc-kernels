#pragma once
#include "mx/utils/meta.h"
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_reduction/reduce_cpu.h"
#include "mx_reduction/reduce_cuda.h"

namespace mx{

// Main reduce function
template<typename T, class Op, class Policy>
T reduce(const T* input, size_t size, T init, Op op, Policy policy){

    if constexpr (std::is_same_v<Policy, CPU>) 
    {
        return reduce_cpu(input, size, init, op, policy.threads);
    }
    else if constexpr (std::is_same_v<Policy, CUDA>)
    {
        return reduce_cuda(input, size, init, op, policy);
    }
    else
    {
        static_assert(always_false_v<Policy>, "mx::reduce: unsupported Policy type");
    }
}

// Default overload: CPU policy
template <typename T, class Op>
T reduce(const T* input, size_t size, T init, Op op){
    CPU serial_policy{};
    return reduce(input, size, init, op, serial_policy); // Calls the main reduce function ''reduce<T, Op, Policy>''
}

} // namespace mx
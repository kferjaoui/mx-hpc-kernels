#pragma once
#include "mx/utils/meta.h"
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_reduction/reduce_cpu.h"
#include "mx_reduction/reduce_cuda.h"

namespace mx{

// Main reduce function
template<typename T, class Op, class Policy = CPU<>>
T reduce(const T* input, size_t size, T init, Op op, Policy policy = Policy{}){

    if constexpr (is_cpu_policy_v<Policy>)
    {
        return reduce_cpu(input, size, init, op, policy.threads);
    }
    else if constexpr (is_cuda_policy_v<Policy>)
    {
        return reduce_cuda(input, size, init, op, policy);
    }
    else
    {
        static_assert(always_false_v<Policy>, "mx::reduce: unsupported Policy type");
    }
}

} // namespace mx
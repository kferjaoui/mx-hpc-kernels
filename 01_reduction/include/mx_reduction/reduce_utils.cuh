#pragma once
#include <cuda_runtime.h>
#include <type_traits>

#include "mx_reduction/operations.h"
#include "mx_reduction/atomics.cuh"

namespace mx{

template <typename T, class Op>
__device__ inline void atomicOp(T* result, T reducedV, const Op& op)
{
    if constexpr (std::is_same_v<Op, mx::Sum<T>>)
    {
        atomicAdd(result, reducedV);
    } 
    else if constexpr (std::is_same_v<Op, mx::Multiply<T>>)
    {
        atomicMul(result, reducedV);
    } 
    else if constexpr (std::is_same_v<Op, mx::Max<T>>)
    {
        atomicMax_fp(result, reducedV);
    } 
    else if constexpr (std::is_same_v<Op, mx::Min<T>>)
    {
        atomicMin_fp(result, reducedV);
    } else {
        static_assert(!sizeof(Op), "mx::atomicOp: unsupported Op type");
    }
}

} // namespace mx
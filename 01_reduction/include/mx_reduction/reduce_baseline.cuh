#pragma once
#include <cuda_runtime.h>
#include <type_traits>

#include "mx_reduction/operations.h"
#include "mx_reduction/atomics.cuh"

namespace mx{

template <typename T, class Op>
__global__ void reduce_baseline(const T* __restrict__ input, // 2 regiters for double* (8 bytes)
                                T* __restrict__ result,      // 2 register for double* (8 bytes)
                                const size_t n,              // 2 registers for size_t (8 bytes)
                                const Op op)                 // Not sure how many registers for 'op'?
{
    T reducedV = op.identity();                         // 2 registers for T = double
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x; // 2 registers for 'tid' if x64 (+ 3 special registers for built-in functions)
    size_t stride = blockDim.x * gridDim.x;             // 2 registers for 'stride' if x64 (+ 2 special registers for built-in functions)
    for (size_t idx=tid; idx < n; idx += stride)        // 2 registers for 'idx' if x64
    {
        reducedV = op(reducedV, input[idx]);            // At least 3 registers (temporary to store result, offset calculation, address calculation)
    }

    if constexpr (std::is_same_v<Op, mx::Sum<T>>)       // Not sure whether there is predicate register for this?
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
    }
}

} // namespace mx
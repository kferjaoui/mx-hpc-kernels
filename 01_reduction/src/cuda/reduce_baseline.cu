#include <cuda_runtime.h>
#include <type_traits>

#include "mx_reduction/operations.h"
#include "mx_reduction/atomics.cuh"

namespace mx{

template <typename T, class Op>
__global__ void reduce_baseline(const T* __restrict__ input,
                                T* __restrict__ result,
                                const size_t n,
                                const Op op)
{
    T reducedV = op.identity();
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx=tid; idx < n; idx += stride)
    {
        reducedV = op(reducedV, input[idx]);
    }

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
    }
}

// Explicit instantiations:
// Sum instantiations
template __global__ void reduce_baseline<double, mx::Sum<double>>(const double* __restrict__ input,
                                                            double* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Sum<double> op);

template __global__ void reduce_baseline<float, mx::Sum<float>>(const float* __restrict__ input,
                                                            float* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Sum<float> op);

// Multiply instantiations
template __global__ void reduce_baseline<double, mx::Multiply<double>>(const double* __restrict__ input,
                                                            double* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Multiply<double> op);

template __global__ void reduce_baseline<float, mx::Multiply<float>>(const float* __restrict__ input,
                                                            float* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Multiply<float> op);

// Max instantiations
template __global__ void reduce_baseline<double, mx::Max<double>>(const double* __restrict__ input,
                                                            double* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Max<double> op);

template __global__ void reduce_baseline<float, mx::Max<float>>(const float* __restrict__ input,
                                                            float* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Max<float> op);

// Min instantiations
template __global__ void reduce_baseline<double, mx::Min<double>>(const double* __restrict__ input,
                                                            double* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Min<double> op);

template __global__ void reduce_baseline<float, mx::Min<float>>(const float* __restrict__ input,
                                                            float* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Min<float> op);

} // namespace mx
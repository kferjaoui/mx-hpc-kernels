#include <cuda_runtime.h>
#include <type_traits>

#include "mx_reduction/operations.h"

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
    for (size_t idx=tid; idx < n; idx += stride){
        reducedV = op(reducedV, input[idx]);
    }
    if constexpr (std::is_same_v<Op, mx::Sum<T>>){
        atomicAdd(result, reducedV);
    } 
    // else if constexpr (std::is_same_v<Op, mx::Multiply<T>>){
    //     atomicMul(result, reducedV);
    // } else if constexpr (std::is_same_v<Op, mx::Max<T>>){
    //     atomicMax(result, reducedV);
    // } else if constexpr (std::is_same_v<Op, mx::Min<T>>){
    //     atomicMin(result, reducedV);
    // }
}

// Explicit instantiation(s):
template __global__ void reduce_baseline<double, Sum<double>>(const double* __restrict__ input,
                                                            double* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Sum<double> op);

template __global__ void reduce_baseline<float, Sum<float>>(const float* __restrict__ input,
                                                            float* __restrict__ result,
                                                            const size_t n,
                                                            const mx::Sum<float> op);
                                                            
// template __global__ void reduce_baseline<double, Multiply<double>>(const double* __restrict__ input,
//                                                             double* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Multiply<double> op);

// template __global__ void reduce_baseline<float, Multiply<float>>(const float* __restrict__ input,
//                                                             float* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Multiply<float> op);

// template __global__ void reduce_baseline<double, Max<double>>(const double* __restrict__ input,
//                                                             double* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Max<double> op);

// template __global__ void reduce_baseline<float, Max<float>>(const float* __restrict__ input,
//                                                             float* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Max<float> op);

// template __global__ void reduce_baseline<double, Min<double>>(const double* __restrict__ input,
//                                                             double* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Min<double> op);

// template __global__ void reduce_baseline<float, Min<float>>(const float* __restrict__ input,
//                                                             float* __restrict__ result,
//                                                             const size_t n,
//                                                             const mx::Min<float> op);

} // namespace mx
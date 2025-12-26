#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include "dot_kernels.cuh"

template<typename T>
__global__ void dotproduct_singleblock_warp_downsweep(const T* __restrict__ x,
                                                      const T* __restrict__ y,
                                                      const size_t n, 
                                                      T* __restrict__ result)
{
    // NOTE: this kernel assumes a single block launch (gridDim.x == 1).
    // It also assumes that blockDim.x is a power of two and at least 64.

    // Per-thread sums
    T sum{};
    for (size_t idx = threadIdx.x; idx<n; idx+=blockDim.x){
        // address seperatly cases of single/double precison T and otherwise
        if constexpr (std::is_same_v<T, float>) {
            sum = __fmaf_rn(x[idx], y[idx], sum);  // CUDA intrinsic for float
        } else if constexpr (std::is_same_v<T, double>) {
            sum = __fma_rn(x[idx], y[idx], sum);  // CUDA intrinsic for double
        } else {
            sum = x[idx] * y[idx] + sum;       // fallback for other types
        }
    }

    extern __shared__ T sh[];
    sh[threadIdx.x] = sum;
    __syncthreads();

    // block-level reduction
    for(size_t stride=blockDim.x >> 1; stride>= warpSize; stride>>= 1){       //warpSize = 32
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride]; 
        __syncthreads();
    }

    // warp-level reduction
    T value = sh[threadIdx.x]; // value in threads's register, not in shared memory

    if(threadIdx.x < warpSize){    // threads with indices: 0..31
        unsigned mask = __activemask();  // lanes currently active at this instruction
        #pragma unroll
        for(size_t offset=warpSize/2; offset>0; offset>>= 1 ){ //offset starts as 16 then 8, 4, 2 and 1 
            value += __shfl_down_sync(mask, value, offset) ; 
        }
    }
    if (threadIdx.x == 0) *result = value;
}

template<typename T>
__global__ void dotproduct_multiblock_warp_downsweep(const T* __restrict__ x,
                                                     const T* __restrict__ y,
                                                     const size_t n,
                                                     T* __restrict__ result)
{   
    T sum{};
    for (size_t idx=threadIdx.x + blockIdx.x * blockDim.x; idx<n; idx += blockDim.x * gridDim.x)
    {
        if constexpr (std::is_same_v<T, float>) {
            sum = __fmaf_rn(x[idx], y[idx], sum);
        } else if constexpr (std::is_same_v<T, double>) {
            sum = __fma_rn(x[idx], y[idx], sum);
        } else {
            sum = x[idx] * y[idx] + sum;
        }
    }
    
    extern __shared__ T sh[];
    sh[threadIdx.x] = sum;
    __syncthreads();

    for(size_t stride=blockDim.x >> 1; stride>= warpSize; stride>>= 1){
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride]; 
        __syncthreads();
    }

    T value = sh[threadIdx.x];
    if (threadIdx.x < warpSize){
        auto mask = __activemask();
        for (size_t off= warpSize>>1; off>0; off>>=1){
            value += __shfl_down_sync(mask, value, off);
        }
    }
    if (threadIdx.x == 0) atomicAdd(result, value);
}

template __global__ void dotproduct_singleblock_warp_downsweep<double>(
    const double* __restrict__, const double* __restrict__, size_t, double* __restrict__);

template __global__ void dotproduct_multiblock_warp_downsweep<double>(
    const double* __restrict__, const double* __restrict__, size_t, double* __restrict__);

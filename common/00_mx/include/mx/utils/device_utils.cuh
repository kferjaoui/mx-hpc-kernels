#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <type_traits>

namespace mx::device {

template <typename T>
__device__ __forceinline__ T* shared_mem_ptr()
{
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    return reinterpret_cast<T*>(smem);
}

template <typename T>
__global__ void set_scalar(T* out, T value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = value;
}

template <typename T>
__global__ void fill_array(T* __restrict__ in, T value, size_t N) {
    size_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_tid < N) in[global_tid] = value;
} 


template <typename T>
__global__ void copy_array(const T* __restrict__ in, 
                           T* __restrict__ out, size_t N) {
    size_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_tid < N) out[global_tid] = in[global_tid];
}


} // namespace mx::device

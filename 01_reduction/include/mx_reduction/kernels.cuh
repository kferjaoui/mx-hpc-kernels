#pragma once
#include <cuda_runtime.h>

namespace mx::cuda_kernels {

template <typename T>
__global__ void set_scalar(T* out, T value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = value;
}

} // namespace mx::cuda_kernels

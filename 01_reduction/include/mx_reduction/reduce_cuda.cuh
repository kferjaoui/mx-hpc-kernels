#pragma once
#include <cuda_runtime.h>
#include "cuda_check.h"

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduce_baseline.cuh"

namespace mx {

template<typename T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy){
    
    T result{};
    T* device_input = nullptr;
    T* device_output= nullptr;

    CUDA_CHECK( cudaMalloc( &device_input, size * sizeof(T) ) );
    CUDA_CHECK( cudaMalloc( &device_output, sizeof(T) ) );

    CUDA_CHECK( cudaMemcpy( device_input, input, size * sizeof(T), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( device_output, &init, sizeof(T), cudaMemcpyHostToDevice ) );

    reduce_baseline<<<cuda_policy.grid, cuda_policy.block>>>(device_input, device_output, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(device_input);
    cudaFree(device_output);
    
    return result;
}

} // namespace mx

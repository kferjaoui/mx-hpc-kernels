#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mx_reduction/reduce_cuda.h"
#include "mx_reduction/reduce_baseline.cuh"
#include "mx_reduction/reduce_interleaved.cuh"
#include "mx_reduction/reduce_sequential.cuh"

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

    int block_size = cuda_policy.block;
    dim3 grid{cuda_policy.grid_x, cuda_policy.grid_y, cuda_policy.grid_z};
    
    // reduce_baseline<<<grid, block_size>>>(device_input, device_output, size, op);
    
    const size_t shmemBytes = block_size * sizeof(T);
    // reduce_block_shmem_interleaved_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
    reduce_block_shmem_sequential_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(device_input);
    cudaFree(device_output);
    
    return result;
}

template double reduce_cuda<double, Sum<double>>(
    const double*, size_t, double, const Sum<double>, const CUDA&);
    
template double reduce_cuda<double, Multiply<double>>(
    const double*, size_t, double, const Multiply<double>, const CUDA&);

template double reduce_cuda<double, Min<double>>(
    const double*, size_t, double, const Min<double>, const CUDA&);

template double reduce_cuda<double, Max<double>>(
    const double*, size_t, double, const Max<double>, const CUDA&);

} // namespace mx

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "CycleTimer.h"
#include "mx_reduction/reduce_cuda.h"
#include "mx/utils/device_utils.cuh"
#include "mx_reduction/detail/reduce_baseline.cuh"
#include "mx_reduction/detail/reduce_interleaved.cuh"
#include "mx_reduction/detail/reduce_sequential.cuh"
#include "mx_reduction/detail/reduce_warp_shuffle.cuh"
#include "mx_reduction/detail/reduce_two_pass.cuh"

#include <numeric>
#include <vector>

namespace mx {

template<typename T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy){
    
    T result{};

    // Kernel launch parameters
    int block_size    = cuda_policy.block;
    std::uint32_t grid_size_1d  = cuda_policy.grid_x * cuda_policy.grid_y * cuda_policy.grid_z; // Flatten the grid dimensions into 1D
    dim3 grid{grid_size_1d, 1, 1};
    const size_t shmemBytes = block_size * sizeof(T);

    // host pointers to device memory
    T* device_input = nullptr;
    T* device_output = nullptr;
    T* partials = nullptr;
    std::vector<T> init_partials(grid_size_1d, init);

    CUDA_CHECK( cudaMalloc( &device_input,  size * sizeof(T) ) );
    CUDA_CHECK( cudaMalloc( &partials,      grid_size_1d * sizeof(T) ) );
    CUDA_CHECK( cudaMalloc( &device_output, sizeof(T) ) );
    
    CUDA_CHECK( cudaMemcpy( device_input,   input,                size * sizeof(T),         cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( partials,       init_partials.data(), grid_size_1d * sizeof(T), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( device_output,  &init,                sizeof(T),                cudaMemcpyHostToDevice ) );
    
    // First pass: reduce input into per-lock partial results
    detail::reduce_multiblock_first_pass<<<grid, block_size, shmemBytes>>>(device_input, partials, size, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Second pass: reduce block partials into final result
    dim3 second_pass_grid{1, 1, 1};
    detail::reduce_monoblock_second_pass<<<second_pass_grid, block_size, shmemBytes>>>(partials, device_output, grid_size_1d, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(device_input);
    cudaFree(partials);
    cudaFree(device_output);
    
    return result;
}

// template<typename T, class Op>
// T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy){
    
//     T result{};
//     T* device_input = nullptr;
//     T* device_output= nullptr;

//     CUDA_CHECK( cudaMalloc( &device_input, size * sizeof(T) ) );
//     CUDA_CHECK( cudaMalloc( &device_output, sizeof(T) ) );

//     CUDA_CHECK( cudaMemcpy( device_input, input, size * sizeof(T), cudaMemcpyHostToDevice ) );
//     CUDA_CHECK( cudaMemcpy( device_output, &init, sizeof(T), cudaMemcpyHostToDevice ) );

//     int block_size = cuda_policy.block;
//     dim3 grid{cuda_policy.grid_x, cuda_policy.grid_y, cuda_policy.grid_z};
//     const size_t shmemBytes = block_size * sizeof(T);
    
//     // Debug info
//     printf("Launching kernel with grid (%u, %u, %u) and block size %d\n", grid.x, grid.y, grid.z, block_size);

//     reduce_warp_shuffle<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
    
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

//     cudaFree(device_input);
//     cudaFree(device_output);
    
//     return result;
// }

// instantiate for double
template double reduce_cuda<double, Sum<double>>(
    const double*, size_t, double, const Sum<double>, const CUDA&);
    
template double reduce_cuda<double, Multiply<double>>(
    const double*, size_t, double, const Multiply<double>, const CUDA&);

template double reduce_cuda<double, Min<double>>(
    const double*, size_t, double, const Min<double>, const CUDA&);

template double reduce_cuda<double, Max<double>>(
    const double*, size_t, double, const Max<double>, const CUDA&);

// instantiate for int
template int reduce_cuda<int, Sum<int>>(
const int*, size_t, int, const Sum<int>, const CUDA&);
    
template int reduce_cuda<int, Multiply<int>>(
    const int*, size_t, int, const Multiply<int>, const CUDA&);

template int reduce_cuda<int, Min<int>>(
    const int*, size_t, int, const Min<int>, const CUDA&);

template int reduce_cuda<int, Max<int>>(
    const int*, size_t, int, const Max<int>, const CUDA&);

} // namespace mx

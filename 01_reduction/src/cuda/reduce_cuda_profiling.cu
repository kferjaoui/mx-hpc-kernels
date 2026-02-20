#include <cuda_runtime.h>
#include <vector>
#include "cuda_check.h"

#include "mx/utils/device_utils.cuh"
#include "mx_reduction/detail/reduce_baseline.cuh"
#include "mx_reduction/detail/reduce_interleaved.cuh"
#include "mx_reduction/detail/reduce_sequential.cuh"
#include "mx_reduction/detail/reduce_warp_shuffle.cuh"
#include "mx_reduction/detail/reduce_two_pass.cuh"
#include "mx_reduction/profiling/reduce_cuda_profiling.h"

namespace mx::profile {

// Helper to convert enum to string for logging
const char* reduce_kernel_name(ReduceKernel kernel) {
    switch(kernel) {
        case ReduceKernel::Baseline:     return "Baseline";
        case ReduceKernel::Interleaved:  return "Interleaved";
        case ReduceKernel::Sequential:   return "Sequential";
        case ReduceKernel::WarpShuffle:  return "WarpShuffle";
        case ReduceKernel::TwoPass:      return "TwoPass";
        default:                         return "Unknown";
    }
}

template<typename T, class Op>
T reduce_cuda_profiled(const T* input,
                       size_t size,
                       T init,
                       Op op,
                       const CUDA& cuda_policy,
                       ReduceKernel kernel,
                       int warmup_iters = 5,
                       int iters = 100)
{
    // Kernel launch parameters
    int block_size = cuda_policy.block;
    std::uint32_t grid_size_1d  = cuda_policy.grid_x * cuda_policy.grid_y * cuda_policy.grid_z; // Flatten the grid dimensions into 1D
    dim3 grid{grid_size_1d, 1, 1};
    const size_t shmemBytes = block_size * sizeof(T);

    // host pointers to device memory
    T result{};
    T* device_input = nullptr;
    T* device_output = nullptr;
    // partials buffer for multi-block reduction (only used by TwoPass kernel, but we allocate it regardless for simplicity)
    T* partials = nullptr;
    std::vector<T> init_partials(grid_size_1d, init);

    CUDA_CHECK(cudaMalloc(&device_input,  size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(T)));
    CUDA_CHECK(cudaMalloc(&partials,      grid_size_1d * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(device_input,  input, size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output, &init, sizeof(T),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(partials, init_partials.data(), grid_size_1d * sizeof(T), cudaMemcpyHostToDevice));

    auto launch = [&](){
        switch (kernel) {
            case ReduceKernel::Baseline:
                ::mx::detail::reduce_baseline<<<grid, block_size>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::Interleaved:
                ::mx::detail::reduce_interleaved_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::Sequential:
                ::mx::detail::reduce_sequential_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::WarpShuffle:
                ::mx::detail::reduce_warp_shuffle<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::TwoPass:
                ::mx::detail::reduce_multiblock_first_pass<<<grid, block_size, shmemBytes>>>(device_input, partials, size, op);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                ::mx::detail::reduce_monoblock_second_pass<<<dim3{1, 1, 1}, block_size, shmemBytes>>>(partials, device_output, grid_size_1d, op);
                break;
        }
    };

    // Warmup (also “pays” one-time costs like clock ramp, caching, JIT if any)
    for (int i = 0; i < warmup_iters; ++i) {
        ::mx::device::set_scalar<T><<<1,1>>>(device_output, init);
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Events timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark
    float total_ms = 0.0f;
    float best_ms  = std::numeric_limits<float>::infinity();

    for (int i = 0; i < iters; ++i) {
        ::mx::device::set_scalar<T><<<1,1>>>(device_output, init);

        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float avg_ms = total_ms / iters;
    printf("[CUDA kernel: %s] avg: %.4f ms | best: %.4f ms | iters: %d\n", reduce_kernel_name(kernel), avg_ms, best_ms, iters);

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(partials));

    return result;
}


// instantiate for double
template double reduce_cuda_profiled<double, Sum<double>>(
    const double*, size_t, double, const Sum<double>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);
    
template double reduce_cuda_profiled<double, Multiply<double>>(
    const double*, size_t, double, const Multiply<double>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);

template double reduce_cuda_profiled<double, Min<double>>(
    const double*, size_t, double, const Min<double>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);

template double reduce_cuda_profiled<double, Max<double>>(
    const double*, size_t, double, const Max<double>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);


// instantiate for int
template int reduce_cuda_profiled<int, Sum<int>>(
const int*, size_t, int, const Sum<int>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);
    
template int reduce_cuda_profiled<int, Multiply<int>>(
    const int*, size_t, int, const Multiply<int>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);

template int reduce_cuda_profiled<int, Min<int>>(
    const int*, size_t, int, const Min<int>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);

template int reduce_cuda_profiled<int, Max<int>>(
    const int*, size_t, int, const Max<int>, const CUDA&, ReduceKernel kernel, int warmup_iters, int iters);

} // namespace mx::profile

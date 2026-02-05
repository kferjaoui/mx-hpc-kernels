#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mx_reduction/reduce_cuda_profiling.h"

#include "mx_reduction/kernels.cuh"
#include "mx_reduction/reduce_baseline.cuh"
#include "mx_reduction/reduce_interleaved.cuh"
#include "mx_reduction/reduce_sequential.cuh"
#include "mx_reduction/reduce_warp_shuffle.cuh"

namespace mx::profile {

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
    T result{};
    T* device_input = nullptr;
    T* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(&device_input,  size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(T)));

    CUDA_CHECK(cudaMemcpy(device_input,  input, size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output, &init, sizeof(T),         cudaMemcpyHostToDevice));

    int block_size = cuda_policy.block;
    dim3 grid{cuda_policy.grid_x, cuda_policy.grid_y, cuda_policy.grid_z};
    const size_t shmemBytes = block_size * sizeof(T);

    auto launch = [&](){
        switch (kernel) {
            case ReduceKernel::Baseline:
                reduce_baseline<<<grid, block_size>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::Interleaved:
                reduce_interleaved_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::Sequential:
                reduce_sequential_addressing<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
            case ReduceKernel::WarpShuffle:
                reduce_warp_shuffle<<<grid, block_size, shmemBytes>>>(device_input, device_output, size, op);
                break;
        }
    };

    // Warmup (also “pays” one-time costs like clock ramp, caching, JIT if any)
    for (int i = 0; i < warmup_iters; ++i) {
        mx::cuda_kernels::set_scalar<T><<<1,1>>>(device_output, init);
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
        mx::cuda_kernels::set_scalar<T><<<1,1>>>(device_output, init);

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
    printf("[CUDA kernel] avg: %.4f ms | best: %.4f ms | iters: %d\n", avg_ms, best_ms, iters);

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

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

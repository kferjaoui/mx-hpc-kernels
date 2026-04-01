#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/layout.h"
#include "mx/types.h"
#include "mx/utils/policy.h"
#include "mx_gemm/detail/device/gemm_naive.cuh"
#include "mx_gemm/detail/device/gemm_shmem_tiling.cuh"
#include "cuda_check.h"


namespace {
    using Layout = mx::RowMajor;
    using Mat    = mx::Dense<float, Layout>;
}

struct BenchStats {
    double min_s    = 0.0;
    double median_s = 0.0;
};

// Helpers
template <class MatT>
void fill_random(MatT& X, float low, float high, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);

    for (mx::index_t i = 0; i < X.rows(); ++i)
        for (mx::index_t j = 0; j < X.cols(); ++j)
            X(i, j) = dist(gen);
}

template <class MatT>
bool allclose(const MatT& A, const MatT& B, float rtol = 1e-4f, float atol = 1e-4f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;

    for (mx::index_t i = 0; i < A.rows(); ++i)
        for (mx::index_t j = 0; j < A.cols(); ++j) {
            const float a    = A(i, j);
            const float b    = B(i, j);
            const float diff = std::abs(a - b);
            const float tol  = atol + rtol * std::abs(b);
            if (diff > tol) return false;
        }
    return true;
}

double gflops(std::size_t N, std::size_t K, std::size_t M, double seconds) {
    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(K) * static_cast<double>(M);
    return flops / seconds / 1.0e9;
}

// GPU timer using CUDA events (measures device time)
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    void record_stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
    }
    // Returns elapsed time in seconds
    double elapsed_seconds() {
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return static_cast<double>(ms) / 1000.0;
    }
};

// Profiler function
template <class Fn>
BenchStats run_benchmark_gpu(std::size_t warmup, std::size_t attempts,
                             cudaStream_t stream, Fn&& fn, 
                             std::function<void()> reset_output_buffer)
{
    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) {
        reset_output_buffer();
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed runs
    std::vector<double> times;
    times.reserve(attempts);

    GpuTimer timer;
    for (std::size_t i = 0; i < attempts; ++i) {
        reset_output_buffer();
        timer.record_start(stream);
        fn();
        timer.record_stop(stream);
        times.push_back(timer.elapsed_seconds());
    }

    std::sort(times.begin(), times.end());

    BenchStats out;
    out.min_s = times.front();
    if (times.size() % 2 == 1) {
        out.median_s = times[times.size() / 2];
    } else {
        const std::size_t i = times.size() / 2;
        out.median_s = 0.5 * (times[i - 1] + times[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// cuBLAS reference GEMM (row-major via column-major trick)
//
// Row-major C = alpha * A * B + beta * C  is equivalent to
// Col-major C^T = alpha * B^T * A^T + beta * C^T
//
// So we pass (B, A) in that order with leading dimensions = number of columns
// in the row-major source matrix.
// ---------------------------------------------------------------------------
void cublas_gemm_rowmajor(cublasHandle_t handle,
                          float alpha,
                          const float* d_A, // row-major, N x K
                          const float* d_B, // row-major, K x M
                          float beta,
                          float* d_C,       // row-major, N x M
                          int N, int K, int M)
{
    // Col-major: C^T(M x N) = alpha * B^T(M x K) * A^T(K x N) + beta * C^T(M x N)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,               // m, n, k in col-major sense
                &alpha,
                d_B, M,                // B^T with lda = M
                d_A, K,                // A^T with ldb = K
                &beta,
                d_C, M);               // C^T with ldc = M
}

void print_result(const char* label,
                  const BenchStats& stats,
                  std::size_t N, std::size_t K, std::size_t M,
                  double baseline_median_s,
                  bool ok)
{
    if (!ok) {
        std::printf("[%-30s] : \033[31m MISMATCH vs cuBLAS reference\033[0m\n", label);
        return;
    }

    const double median_ms = stats.median_s * 1000.0;
    const double min_ms    = stats.min_s    * 1000.0;
    const double perf      = gflops(N, K, M, stats.median_s);
    const double speedup   = baseline_median_s / stats.median_s;

    std::printf("[%-30s] : median %9.3f ms | min %9.3f ms | %8.2f GF/s | x %6.2f\n",
                label, median_ms, min_ms, perf, speedup);
}

void print_case_header(std::size_t N, std::size_t K, std::size_t M) {
    std::printf("\n==============================================================\n");
    std::printf("Case: A(%zu x %zu), B(%zu x %zu), C(%zu x %zu)\n",
                N, K, K, M, N, M);
    std::printf("==============================================================\n");
}

// --- Main ----------------------------------------------------------------
int main() {
    constexpr std::size_t WARMUP     = 3;
    constexpr std::size_t N_ATTEMPTS = 10;
    constexpr float ALPHA            = 1.0f;
    constexpr float BETA             = 2.0f;

    // Print device info
    int device = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::printf("Device        : %s\n", props.name);
    std::printf("SMs           : %d\n", props.multiProcessorCount);
    std::printf("Clock         : %d MHz\n", props.clockRate / 1000);
    std::printf("Global mem    : %.0f MB\n", props.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("Shared mem/SM : %zu KB\n", props.sharedMemPerMultiprocessor / 1024);
    std::printf("Regs/SM       : %d\n", props.regsPerMultiprocessor);
    std::printf("Max threads/SM: %d\n", props.maxThreadsPerMultiProcessor);
    std::printf("Warp size     : %d\n", props.warpSize);
    std::printf("Compute cap   : %d.%d\n\n", props.major, props.minor);

    std::printf("Warmup runs   : %zu\n", WARMUP);
    std::printf("Timed runs    : %zu\n\n", N_ATTEMPTS);

    // Create stream and cuBLAS handle
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);

    // Attach stream and timing events to CUDA policy for my kernels
    mx::CUDA cuda_policy;
    cuda_policy.stream = reinterpret_cast<std::uintptr_t>(stream);

    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> cases = {
        {257,  131,   79},
        {1024, 768,  1536},
        {2000, 2000, 2000},
        {4096, 4096, 4096}
    };

    for (std::size_t case_id = 0; case_id < cases.size(); ++case_id) {
        const auto [N, K, M] = cases[case_id];
        print_case_header(N, K, M);

        // Host matrices (row-major)
        Mat A(N, K);
        Mat B(K, M);
        Mat C0(N, M);

        fill_random(A,  -1.0f, 1.0f, 1000u + 10u * static_cast<unsigned>(case_id) + 1u);
        fill_random(B,  -1.0f, 1.0f, 1000u + 10u * static_cast<unsigned>(case_id) + 2u);
        fill_random(C0, -1.0f, 1.0f, 1000u + 10u * static_cast<unsigned>(case_id) + 3u);

        // Allocate device memory
        const std::size_t size_A = N * K * sizeof(float);
        const std::size_t size_B = K * M * sizeof(float);
        const std::size_t size_C = N * M * sizeof(float);

        float *d_A, *d_B, *d_C, *d_C0, *d_C_ref;
        CUDA_CHECK(cudaMalloc(&d_A,     size_A));
        CUDA_CHECK(cudaMalloc(&d_B,     size_B));
        CUDA_CHECK(cudaMalloc(&d_C,     size_C));
        CUDA_CHECK(cudaMalloc(&d_C0,    size_C));
        CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));

        // Upload A and B once (constant across all kernels)
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), size_A, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), size_B, cudaMemcpyHostToDevice, stream));
       
        CUDA_CHECK(cudaMemcpyAsync(d_C0, C0.data(), size_C, cudaMemcpyHostToDevice, stream));

        // Reference: cuBLAS
        std::printf("\nReference\n\n");

        Mat C_ref(N, M);
        auto cublas_stats = run_benchmark_gpu(WARMUP, N_ATTEMPTS, stream,
                                [&]() {
                                    cublas_gemm_rowmajor(cublas_handle, ALPHA, d_A, d_B, BETA, d_C_ref,
                                                        static_cast<int>(N), static_cast<int>(K), static_cast<int>(M));
                                },
                                [&]() {
                                    CUDA_CHECK(cudaMemcpyAsync(d_C_ref, d_C0, size_C, cudaMemcpyDeviceToDevice, stream));
                                });

        // Download cuBLAS result for correctness checks
        CUDA_CHECK(cudaMemcpyAsync(C_ref.data(), d_C_ref, size_C, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::printf("[%-30s] : median %9.3f ms | min %9.3f ms | %8.2f GF/s\n",
                    "cuBLAS",
                    cublas_stats.median_s * 1000.0,
                    cublas_stats.min_s    * 1000.0,
                    gflops(N, K, M, cublas_stats.median_s));

        // My CUDA kernels
        std::printf("\nCUDA kernels\n\n");

        // Naive
        {
            Mat C_out = C0;
            const auto blockDim = static_cast<size_t>(cuda_policy.block);
            const auto gridDim  = static_cast<size_t>((N * M + blockDim - 1) / blockDim);

            auto stats = run_benchmark_gpu(WARMUP, N_ATTEMPTS, stream,
                            [&]() {
                                ::mx::detail::gemm_naive_1d_kernel<<<gridDim, blockDim, 0, stream>>>(
                                    ALPHA, d_A, d_B, BETA, d_C, N, K, M);
                            },
                            [&]() {
                                CUDA_CHECK(cudaMemcpyAsync(d_C, d_C0, size_C, cudaMemcpyDeviceToDevice, stream));
                            });

            CUDA_CHECK(cudaMemcpyAsync(C_out.data(), d_C, size_C, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const bool ok = allclose(C_out, C_ref);
            print_result("CUDA Naive", stats, N, K, M, cublas_stats.median_s, ok);
        }

        // Shared memory Tiling
        {
            Mat C_out = C0;

            auto stats = run_benchmark_gpu(WARMUP, N_ATTEMPTS, stream,
                            [&]() {
                                ::mx::detail::call_gemm_shmem_tiled(ALPHA, d_A, d_B, BETA, d_C, N, M, K, cuda_policy);
                            },
                            [&]() {
                                CUDA_CHECK(cudaMemcpyAsync(d_C, d_C0, size_C, cudaMemcpyDeviceToDevice, stream));
                            });

            CUDA_CHECK(cudaMemcpyAsync(C_out.data(), d_C, size_C, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const bool ok = allclose(C_out, C_ref);
            print_result("CUDA Shared Memory Tiling", stats, N, K, M, cublas_stats.median_s, ok);
        }

        // Cleanup per-case device memory
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_C0));
        CUDA_CHECK(cudaFree(d_C_ref));
    }

    // Cleanup globals
    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}

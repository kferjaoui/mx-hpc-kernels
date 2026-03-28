#include <type_traits>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mx/utils/meta.h"
#include "mx_gemm/gemm_cuda.h"

namespace mx {

template<detail::CudaGemmAlgorithm gemm_algo, typename T, class Layout>
void gemm_cuda(const T alpha, 
            DenseView<const T, Layout> A, 
            DenseView<const T, Layout> B,
            const T beta, 
            DenseView<T, Layout> C,
            const CUDA& cuda_policy) 
{
    if constexpr (is_same_v<Layout, RowMajor>){

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_policy.stream);
    
        auto N = A.rows();
        auto K = A.cols();
        auto M = B.cols();
    
        // Allocate device memory
        T *dA, *dB, *dC;
        size_t size_A = static_cast<size_t>(N * K * sizeof(T));
        size_t size_B = static_cast<size_t>(K * M * sizeof(T));
        size_t size_C = static_cast<size_t>(N * M * sizeof(T));
    
        CUDA_CHECK(cudaMalloc(&dA, size_A));
        CUDA_CHECK(cudaMalloc(&dB, size_B));
        CUDA_CHECK(cudaMalloc(&dC, size_C));
    
        CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), size_A, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), size_B, cudaMemcpyHostToDevice, stream));
        if (beta != static_cast<T>(0)) {
            CUDA_CHECK(cudaMemcpyAsync(dC, C.data(), size_C, cudaMemcpyHostToDevice, stream));
        }
        // launch the kernel based on the selected algorithm
        const auto blockDim = static_cast<size_t>(cuda_policy.block);
        const auto gridDim  = static_cast<size_t>((N * M + blockDim - 1) / blockDim);
        
        detail::gemm_naive_1d_kernel<<<gridDim, blockDim, 0, stream>>>(
                                                alpha, dA, dB, beta, dC, N, K, M);
        
        CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, size_C, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    
        CUDA_CHECK(cudaFree(dA)),
        CUDA_CHECK(cudaFree(dB)),
        CUDA_CHECK(cudaFree(dC)),
    } else {
        static_assert(always_false_v<Layout>, "Only supports RowMajor layouts.");
    }
    
}

// TODO; Still needs explicit instantiation of each type T={int, float, double}, 
// algorithm type and Layout

} // namespace mx
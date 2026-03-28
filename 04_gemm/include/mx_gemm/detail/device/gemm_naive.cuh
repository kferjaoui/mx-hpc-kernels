#pragma once 


namespace mx::detail {


// Note: Assumes RowMajor layout
template<typename T>
__global__ void gemm_naive_1d_kernel(const T alpha, const T* __restrict__ dA, const T* __restrict__ dB,
                                const T beta, T* __restrict__ dC, 
                                const index_t N, const index_t K, const index_t M) 
{
    auto g_tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (g_tid < N * M){
        T acc_ij{};
        // Each thread of global index (g_tid) maps to the C(i,j) element where:
        auto i = g_tid / M;
        auto j = g_tid - i * M; // g_tid % M
        for (unsigned int k = 0; k < K; ++k) {
            acc_ij += dA[k + K*i] * dB[j + M*k]; // A(i,k) * B(k,j)
        }

        dC[g_tid] = beta * dC[g_tid] + alpha * acc_ij;
    }
}

}
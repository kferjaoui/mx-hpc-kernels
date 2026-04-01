#pragma once
#include <cuda_runtime.h>
#include "mx/utils/device_utils.cuh"
#include "mx/utils/policy.h"

namespace mx::detail {

template <typename T, size_t Nr, size_t Mr>
__global__ void gemm_register_tiled(const T alpha, const T* __restrict__ A, const T* __restrict__ B,
                                    const T beta, T* __restrict__ C,
                                    const size_t N, const size_t M, const size_t K,
                                    const size_t N_TILE, const size_t M_TILE, const size_t K_TILE) 
{
    // Calculate local thread ID
    auto nThreadsperBlock = blockDim.x * blockDim.y; // also equal to ceil(N_TILE/Nr) * ceil(M_TILE/Mr)

    T* shmem = device::shared_mem_ptr<T>();

    // Global thread Id that map to C(ir,jr) element; 
    // where (ir, jr) is the top-left element of the subtile computed by this thread
    auto gtid_y = Nr * threadIdx.y + N_TILE * blockIdx.y; // ir; first row (global)index of the subtile
    auto gtid_x = Mr * threadIdx.x + M_TILE * blockIdx.x; // jr; first col (global) index of the subtile

    auto local_tid_1d = threadIdx.y * blockDim.x + threadIdx.x; // local thread id in the block
                                                                // example: for 256 threads per block, it ranges from 0 to 255

    auto size_A_tile = N_TILE * K_TILE;

    T C_subtile[Nr*Mr]{0};

    for(int start_K_TILE=0; start_K_TILE<K; start_K_TILE += K_TILE){
        auto actual_K_TILE = std::min(K-start_K_TILE, K_TILE);

        // load A_tile in the first (N_Tile * K_TILE) elements in row-major order
        auto offset_A_tile = start_K_TILE + K * (N_TILE * blockIdx.y);
        for(size_t idx = 0; idx < size_A_tile; idx += nThreadsperBlock){
            auto col_shift = (local_tid_1d + idx) % actual_K_TILE;
            auto row_shift = (local_tid_1d + idx) / actual_K_TILE;

            if ( (local_tid_1d + idx) < size_A_tile ) shmem[local_tid_1d + idx] = A[offset_A_tile + col_shift + K * row_shift];
        }

        auto size_B_tile = actual_K_TILE*M_TILE;
        auto start_B_tile = size_A_tile;
        
        // load B_tile in the next (K_Tile * M_TILE) elements in row-major order
        auto offset_B_tile = (M_TILE * blockIdx.x) + start_K_TILE * M;
        for(size_t idx = 0; idx < size_B_tile; idx += nThreadsperBlock){
            auto col_shift = (local_tid_1d + idx) % M_TILE;
            auto row_shift = (local_tid_1d + idx) / M_TILE;
            if ( (local_tid_1d + idx) < size_B_tile ) shmem[start_B_tile + local_tid_1d + idx] = B[offset_B_tile + col_shift + M * row_shift];
        }

        __syncthreads(); // make sure the tiles are loaded before we start computing
    
        for(size_t k=0; k < actual_K_TILE; ++k){
            auto local_start_row_i_of_A = (threadIdx.y * Nr) * actual_K_TILE;
            auto local_start_col_j_of_B = (threadIdx.x * Mr);
            // Each thread conputes its subtile in row-major layout i.e. one row after the other
            for(size_t ir=0; ir<Nr; ++ir)
                for(size_t jr=0; jr<Mr; ++jr){
                    C_subtile[jr + ir * Mr] += alpha * shmem[local_start_row_i_of_A + k + (ir * actual_K_TILE)] * shmem[start_B_tile + local_start_col_j_of_B + jr + k * M_TILE]; 
                }
        }

        __syncthreads(); // wait for all threads to finish before overwriting the shared memory with the next tile
    }

    if (gtid_x < M && gtid_y < N) { // guard againt OOB access of C
        auto block_offset = blockIdx.x * M_TILE + M * (N_TILE * blockIdx.y);
        auto thread_offset_in_subtile = threadIdx.x * Mr + (threadIdx.y * Nr) * M;
        auto offset_C_subtile = block_offset + thread_offset_in_subtile;
        for(size_t ir = 0; ir<Nr; ++ir){
            for(size_t jr = 0; jr<Mr; ++jr){
                if (gtid_x + jr < M && gtid_y + ir < N) {
                    C[offset_C_subtile + jr + (ir * M)] = beta * C[offset_C_subtile + jr + (ir * M)] + C_subtile[jr + ir * Mr];
                }
            }
    }
    }

}

template <typename T>
__host__ void call_gemm_register_tiled(const T alpha, const T* __restrict__ dA, const T* __restrict__ dB,
                    const T beta, T* __restrict__ dC, const size_t N, const size_t M, const size_t K,
                    const CUDA& cuda_policy){

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_policy.stream);

    constexpr size_t K_TILE = 32;
    
    constexpr size_t N_TILE = 64;
    constexpr size_t M_TILE = 64;

    constexpr size_t N_SUBTILE = 4;
    constexpr size_t M_SUBTILE = 4;
    
    size_t nThreadsPerBlock_x = (M_TILE + M_SUBTILE - 1) / M_SUBTILE;
    size_t nThreadsPerBlock_y = (N_TILE + N_SUBTILE - 1) / N_SUBTILE;
    dim3 threadBlock(nThreadsPerBlock_x, nThreadsPerBlock_y, 1);

    auto grid_x = (M + M_TILE -1) / M_TILE;
    auto grid_y = (N + N_TILE -1) / N_TILE;
    dim3 grid(grid_x, grid_y, 1);
    const size_t shmemBytes = sizeof(T) * ( K_TILE * (N_TILE + M_TILE) ); // still needs to hold one A_tile and one B_tile

    gemm_register_tiled<T, N_SUBTILE, M_SUBTILE><<<grid, threadBlock, shmemBytes, stream>>>(
                                alpha, dA, dB, beta, dC, N, M, K, N_TILE, M_TILE, K_TILE);
}

} // namespace mx::detail

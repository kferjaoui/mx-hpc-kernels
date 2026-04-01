#pragma once
#include <cuda_runtime.h>
#include "mx/utils/device_utils.cuh"
#include "mx/utils/policy.h"

namespace mx::detail {

template <typename T>
__global__ void gemm_shmem_tiled(const T alpha, const T* __restrict__ A, const T* __restrict__ B,
                                    const T beta, T* __restrict__ C,
                                    const size_t N, const size_t M, const size_t K,
                                    const size_t N_TILE, const size_t M_TILE, const size_t K_TILE) 
{
    // Calculate local thread ID
    auto TILE_SIZE = N_TILE * M_TILE;

    T* shmem = device::shared_mem_ptr<T>();

    // Global thread Id that map to C(i,j)
    auto gtid_y = threadIdx.y + N_TILE * blockIdx.y; // i; also blockDim.y = N_TILE
    auto gtid_x = threadIdx.x + M_TILE * blockIdx.x; // j; also blockDim.x = M_TILE

    auto local_tid_1d = threadIdx.y * M_TILE + threadIdx.x; // local thread id in the block (flattened i.e. 1D)

    auto size_A_tile = N_TILE * K_TILE;

    T C_ij{0};

    for(int start_K_TILE=0; start_K_TILE<K; start_K_TILE += K_TILE){
        auto actual_K_TILE = std::min(K-start_K_TILE, K_TILE);

        // load A_tile in the first (N_Tile * K_TILE) elements in row-major order
        auto offset_A_tile = start_K_TILE + K * (N_TILE * blockIdx.y);
        for(size_t idx = 0; idx < size_A_tile; idx += TILE_SIZE){
            auto col_shift = (local_tid_1d + idx) % actual_K_TILE;
            auto row_shift = (local_tid_1d + idx) / actual_K_TILE;

            if ( (local_tid_1d + idx) < size_A_tile ) shmem[local_tid_1d + idx] = A[offset_A_tile + col_shift + K * row_shift];
        }

        auto size_B_tile = actual_K_TILE*M_TILE;
        auto start_B_tile = size_A_tile;
        
        // load B_tile in the next (K_Tile * M_TILE) elements in row-major order
        auto offset_B_tile = (M_TILE * blockIdx.x) + start_K_TILE * M;
        for(size_t idx = 0; idx < size_B_tile; idx += TILE_SIZE){
            auto col_shift = (local_tid_1d + idx) % M_TILE;
            auto row_shift = (local_tid_1d + idx) / M_TILE;
            if ( (local_tid_1d + idx) < size_B_tile  ) shmem[start_B_tile + local_tid_1d + idx] = B[offset_B_tile + col_shift + M * row_shift];
        }

        __syncthreads(); // make sure the tiles are loaded before we start computing
    
        for(size_t k=0; k < actual_K_TILE; ++k){
            auto local_start_row_i_of_A = threadIdx.y * actual_K_TILE;
            auto local_start_col_j_of_B = threadIdx.x;
            // access contiguous elements in row of A_tile but strided elements in col of B_tile
            C_ij += alpha * shmem[local_start_row_i_of_A + k] * shmem[start_B_tile + local_start_col_j_of_B + k * M_TILE]; 
        }

        __syncthreads(); // wait for all threads to finish before overwriting the shared memory with the next tile
    }

    if (gtid_x < M && gtid_y < N) { // guard againt OOB access of C
        C[gtid_y * M + gtid_x] = beta * C[gtid_y * M + gtid_x] + C_ij;
    }      
}

template <typename T>
__host__ void call_gemm_shmem_tiled(const T alpha, const T* __restrict__ dA, const T* __restrict__ dB,
                    const T beta, T* __restrict__ dC, const size_t N, const size_t M, const size_t K,
                    const CUDA& cuda_policy){

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_policy.stream);

    constexpr size_t K_TILE = 32;
    constexpr size_t N_TILE = 16;
    constexpr size_t M_TILE = 16;
    auto grid_x = (M + M_TILE -1) / M_TILE;
    auto grid_y = (N + N_TILE -1) / N_TILE;
    dim3 grid(grid_x, grid_y, 1);
    dim3 threadBlock(M_TILE, N_TILE, 1);
    const size_t shmemBytes = sizeof(T) * ( K_TILE * (N_TILE + M_TILE) );

    gemm_shmem_tiled<<<grid, threadBlock, shmemBytes, stream>>>(
                                alpha, dA, dB, beta, dC, N, M, K, N_TILE, M_TILE, K_TILE);
}

} // namespace mx::detail

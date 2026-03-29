#pragma once
#include <cuda_runtime.h>
#include "mx/utils/device_utils.cuh"
#include "mx/utils/policy.h"

namespace mx::detail {

template <typename T>
__global__ void gemm_shmem_tile_partial(const T alpha, const T* __restrict__ A, const T* __restrict__ B,
                                    const T beta, T* __restrict__ C,
                                    const size_t N, const size_t M, const size_t K,
                                    const size_t N_TILE, const size_t M_TILE, const size_t K_TILE,
                                    int K_TILE_START) 
{
    // Calculate local thread ID
    auto TILE_SIZE = N_TILE * M_TILE;

    T* shmem = device::shared_mem_ptr<T>();

    // Global thread Id that map to C(i,j)
    auto gtid_y = threadIdx.y + N_TILE * blockIdx.y; // i; also blockDim.y = N_TILE
    auto gtid_x = threadIdx.x + M_TILE * blockIdx.x; // j; also blockDim.x = M_TILE

    auto local_tid_1d = threadIdx.y * M_TILE + threadIdx.x; // local thread id in the block (flattened i.e. 1D)

    auto size_A_tile = N_TILE * K_TILE;

    // load A_tile in the first (N_Tile * K_TILE) elements in row-major order
    auto offset_A_tile = K_TILE_START + K * (N_TILE * blockIdx.y);
    for(size_t idx = 0; idx < size_A_tile; idx += TILE_SIZE){
        auto col_shift = (local_tid_1d + idx) % K_TILE;
        auto row_shift = (local_tid_1d + idx) / K_TILE;

        if ( (local_tid_1d + idx) < size_A_tile ) shmem[local_tid_1d + idx] = A[offset_A_tile + col_shift + K * row_shift];
    }

    auto size_B_tile = K_TILE*M_TILE;
    auto start_B_tile = size_A_tile;
    auto end_B_tile = size_A_tile + size_B_tile;
    
    // load B_tile in the next (K_Tile * M_TILE) elements in row-major order
    auto offset_B_tile = (M_TILE * blockIdx.x) + K_TILE_START * M;
    for(size_t idx = 0; idx < size_B_tile; idx += TILE_SIZE){
        auto col_shift = (local_tid_1d + idx) % M_TILE;
        auto row_shift = (local_tid_1d + idx) / M_TILE;
        if ( (local_tid_1d + idx) < size_B_tile  ) shmem[start_B_tile + local_tid_1d + idx] = B[offset_B_tile + col_shift + M * row_shift];
    }

    // load C_tile in the next (N_TILE * M_TILE) elements
    auto size_C_tile = N_TILE * M_TILE; 
    auto end_C_tile = end_B_tile + size_C_tile;
    
    if (gtid_x < M && gtid_y < N) { // guard againt OOB access of C
        shmem[end_B_tile + local_tid_1d] = C[gtid_y * M + gtid_x]; // each thread loads one element of C_tile
    }  

    __syncthreads(); // make sure the tiles are loaded before we start computing
  
    // first K_TILE, we need to scale C by beta
    T C_ij = shmem[end_B_tile + local_tid_1d];
    if (K_TILE_START == 0) {
        // we know there is exactly one element of C_tile per thread (no need to loop OR check bounds) -> TODO: generalize?
        C_ij *= beta;
    }

    for(size_t k=0; k < K_TILE; ++k){
        auto local_start_row_i_of_A = threadIdx.y * K_TILE;
        auto local_start_col_j_of_B = threadIdx.x;
        // access contiguous elements in row of A_tile but strided elements in col of B_tile
        C_ij += alpha * shmem[local_start_row_i_of_A + k] * shmem[start_B_tile + local_start_col_j_of_B + k * M_TILE]; 
    }
    __syncthreads();

    if (gtid_x < M && gtid_y < N) { // guard againt OOB access of C
        C[gtid_y * M + gtid_x] = C_ij;
    }      
}

template <typename T>
__host__ void gemm_shmem_tiled(const T alpha, const T* __restrict__ dA, const T* __restrict__ dB,
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
    const size_t shmemBytes = sizeof(T) * ( N_TILE * M_TILE + K_TILE * (N_TILE + M_TILE) );

    for(int start_K_TILE=0; start_K_TILE<K; start_K_TILE += K_TILE){
        auto actual_K_TILE = std::min(K-start_K_TILE, K_TILE);
        gemm_shmem_tile_partial<<<grid, threadBlock, shmemBytes, stream>>>(
                                    alpha, dA, dB, beta, dC, N, M, K, N_TILE, M_TILE, actual_K_TILE, start_K_TILE);
    }
}

} // namespace mx::detail

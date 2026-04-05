#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include "mx/utils/device_utils.cuh"
#include "mx/utils/policy.h"
#include "mx_gemm/detail/algorithms.h"

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

    T C_subtile[Nr*Mr]{0};

    for(int start_K_TILE=0; start_K_TILE<K; start_K_TILE += K_TILE){
        auto actual_K_TILE = std::min(K-start_K_TILE, K_TILE);
        auto size_A_tile = N_TILE * actual_K_TILE;

        // load A_tile in the first (N_Tile * K_TILE) elements in row-major order
        auto offset_A_tile = start_K_TILE + K * (N_TILE * blockIdx.y);
        for(size_t idx = 0; idx < size_A_tile; idx += nThreadsperBlock){
            auto col_shift = (local_tid_1d + idx) % actual_K_TILE;
            auto row_shift = (local_tid_1d + idx) / actual_K_TILE;

            if ( (local_tid_1d + idx) < size_A_tile ) shmem[local_tid_1d + idx] = A[offset_A_tile + col_shift + K * row_shift];
        }

        auto size_B_tile = actual_K_TILE * M_TILE;
        auto start_B_tile = N_TILE * K_TILE; 
        // NOTE: Always start loading B_tile at (N_TILE * K_TILE) despite the actual_K_TILE, 
        // hence, avoiding any overlaps with the A_tile in shared memory
        
        // load B_tile in the next (actual_K_TILE * M_TILE) elements in row-major order
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

// ------
// NOTES:
// ------
// - The kernel does not deal with tails when K % (sizeof(float4)/sizeof(T)) != 0 -> Use scalar kernel for those cases
// - This kernel is designed to work with a transposed B matrix (BT) that is stored in row-major order.
//
// -----
// TODO:
// -----
// - Templating is kind of useless here as the below kernel does not support double precision anyway 
//   due to the use of float4 for vectorized loads; 
// - It can be extended to support double by using double2 and adjusting the indexing accordingly.
// -> make vectorization type-dependent and sue a strcut such as Vec128<T> that maps to float4 for float and double2 for double:
//
// template <typename T>
// struct Vec128;

// template <>
// struct Vec128<float> {
//     using type = float4;
//     static constexpr size_t width = 4;
// };

// template <>
// struct Vec128<double> {
//     using type = double2;
//     static constexpr size_t width = 2;
// };

template <typename T, size_t N_TILE, size_t M_TILE, size_t K_TILE, size_t Nr, size_t Mr>
__global__ void gemm_register_tiled_vectorized(const T alpha,
                                               const T* __restrict__ A,
                                               const T* __restrict__ BT,
                                               const T beta,
                                               T* __restrict__ C,
                                               const size_t N,
                                               const size_t M,
                                               const size_t K)
{
    static_assert(std::is_same_v<T, float>,
                  "This vectorized kernel currently supports float only.");
    static_assert(K_TILE % (sizeof(float4) / sizeof(T)) == 0,
                  "K_TILE must be divisible by the vector width.");

    constexpr size_t VEC_WIDTH   = sizeof(float4) / sizeof(T); // 4 for float
    constexpr size_t K_TILE_F4   = K_TILE / VEC_WIDTH;
    constexpr size_t K_TILE_PAD  = K_TILE + 1;

    T* shmem = device::shared_mem_ptr<T>();
    auto* shmem_f4 = reinterpret_cast<float4*>(shmem);

    const auto nThreadsPerBlock = blockDim.x * blockDim.y;
    const auto local_tid_1d     = threadIdx.y * blockDim.x + threadIdx.x;

    const auto gtid_y = Nr * threadIdx.y + N_TILE * blockIdx.y;
    const auto gtid_x = Mr * threadIdx.x + M_TILE * blockIdx.x;

    const auto* A_f4  = reinterpret_cast<const float4*>(A);
    const auto* BT_f4 = reinterpret_cast<const float4*>(BT);

    const size_t K_f4          = K / VEC_WIDTH;
    const size_t start_BT_tile = N_TILE * K_TILE; // A tile is unpadded, BT tile starts right after it

    T C_subtile[Nr * Mr]{0};

    for (size_t start_K = 0; start_K < K; start_K += K_TILE) {
        const size_t start_K_f4 = start_K / VEC_WIDTH;

        // -----------------------------
        // Load A tile: [N_TILE x K_TILE]
        // -----------------------------
        const size_t size_A_tile_f4   = N_TILE * K_TILE_F4;
        const size_t offset_A_tile_f4 = start_K_f4 + K_f4 * (N_TILE * blockIdx.y);

        for (size_t idx_A4 = local_tid_1d; idx_A4 < size_A_tile_f4; idx_A4 += nThreadsPerBlock) {
            const size_t col_shift_f4 = idx_A4 % K_TILE_F4;
            const size_t row_shift    = idx_A4 / K_TILE_F4;

            shmem_f4[idx_A4] = A_f4[offset_A_tile_f4 + col_shift_f4 + K_f4 * row_shift];
        }

        // --------------------------------------------
        // Load BT tile: [M_TILE x K_TILE], padded in SMEM
        // --------------------------------------------
        const size_t size_BT_tile_f4   = M_TILE * K_TILE_F4;
        const size_t offset_BT_tile_f4 = start_K_f4 + (blockIdx.x * M_TILE) * K_f4;

        for (size_t idx_B4 = local_tid_1d; idx_B4 < size_BT_tile_f4; idx_B4 += nThreadsPerBlock) {
            const size_t col_shift_f4 = idx_B4 % K_TILE_F4;
            const size_t row_shift    = idx_B4 / K_TILE_F4;

            const float4 val = BT_f4[offset_BT_tile_f4 + col_shift_f4 + K_f4 * row_shift];

            const size_t base_T = start_BT_tile
                                + row_shift * K_TILE_PAD
                                + col_shift_f4 * VEC_WIDTH;

            const T* elems = reinterpret_cast<const T*>(&val);
            #pragma unroll
            for (size_t i = 0; i < VEC_WIDTH; ++i) {
                shmem[base_T + i] = elems[i];
            }
        }

        __syncthreads();

        // Thread-local bases for the compute phase (compute outside the hot k-loop for reuse)
        const size_t a_base  = (threadIdx.y * Nr) * K_TILE;
        const size_t bt_base = start_BT_tile + (threadIdx.x * Mr) * K_TILE_PAD;

        #pragma unroll
        for (size_t k_inner = 0; k_inner < K_TILE; ++k_inner) {
            // use registers to maximize reuse and minimize shmem access latency 
            T a_reg[Nr];
            T b_reg[Mr];

            #pragma unroll
            for (size_t ir = 0; ir < Nr; ++ir) {
                a_reg[ir] = alpha * shmem[a_base + ir * K_TILE + k_inner];
            }

            #pragma unroll
            for (size_t jr = 0; jr < Mr; ++jr) {
                b_reg[jr] = shmem[bt_base + jr * K_TILE_PAD + k_inner];
            }

            #pragma unroll
            for (size_t ir = 0; ir < Nr; ++ir) {
                #pragma unroll
                for (size_t jr = 0; jr < Mr; ++jr) {
                    C_subtile[jr + ir * Mr] += a_reg[ir] * b_reg[jr];
                }
            }
        }

        __syncthreads();
    }

    if (gtid_x < M && gtid_y < N) {
        const auto block_offset             = blockIdx.x * M_TILE + M * (N_TILE * blockIdx.y);
        const auto thread_offset_in_subtile = threadIdx.x * Mr + (threadIdx.y * Nr) * M;
        const auto offset_C_subtile         = block_offset + thread_offset_in_subtile;

        for (size_t ir = 0; ir < Nr; ++ir) {
            for (size_t jr = 0; jr < Mr; ++jr) {
                if (gtid_x + jr < M && gtid_y + ir < N) {
                    C[offset_C_subtile + jr + ir * M] =
                        beta * C[offset_C_subtile + jr + ir * M] + C_subtile[jr + ir * Mr];
                }
            }
        }
    }
}

// // TODO: Correct the kernel to support different data types T 

// template <typename T, size_t N_TILE, size_t M_TILE, size_t K_TILE, size_t Nr, size_t Mr>
// __global__ void gemm_register_tiled_vectorized(const T alpha, const T* __restrict__ A, const T* __restrict__ BT, // BT is the transposed B matrix; we will load it in row-major order to make use of vectorized loads
//                                             const T beta, T* __restrict__ C,
//                                             const size_t N, const size_t M, const size_t K) 
// {
//     // Calculate local thread ID
//     auto nThreadsperBlock = blockDim.x * blockDim.y; // also equal to ceil(N_TILE/Nr) * ceil(M_TILE/Mr)

//     T* shmem = device::shared_mem_ptr<T>();
//     auto shmem_f4 = reinterpret_cast<float4*>(shmem);

//     // Global thread Id that map to C(ir,jr) element; 
//     // where (ir=0, jr=0) is the top-left element of the subtile computed by this thread
//     auto gtid_y = Nr * threadIdx.y + N_TILE * blockIdx.y; // ir; first row (global)index of the subtile
//     auto gtid_x = Mr * threadIdx.x + M_TILE * blockIdx.x; // jr; first col (global) index of the subtile

//     auto local_tid_1d = threadIdx.y * blockDim.x + threadIdx.x; // local thread id in the block
//                                                                 // example: for 256 threads per block, it ranges from 0 to 255

//     auto A_f4 = reinterpret_cast<const float4*>(A);
//     auto BT_f4 = reinterpret_cast<const float4*>(BT);
//     auto jump = sizeof(float4) / sizeof(T); // for T = double, jump = 2 = (128/64)
    
//     auto K_f4 = K / jump;
//     auto K_TILE_f4 = K_TILE / jump;

//     const size_t K_TILE_PAD = K_TILE + 1; // 33 for K_TILE = 32
//     auto start_BT_tile = N_TILE * K_TILE;  // T-space offset (A tile is unpadded)

//     T C_subtile[Nr*Mr]{0};

//     for(int start_K_TILE_f4=0; start_K_TILE_f4 < K_f4; start_K_TILE_f4 += K_TILE_f4){
//         auto actual_K_TILE = std::min(K - (start_K_TILE_f4 * jump), K_TILE);
//         auto actual_K_TILE_f4 = std::min(K_f4 - start_K_TILE_f4, K_TILE_f4);
//         auto size_A_tile_f4 = N_TILE * actual_K_TILE_f4; // number of f4 elements in A_tile

//         // Vectorized loads of A_tile in the first (N_Tile * K_TILE) T-typed elements i.e. (N_Tile * K_TILE_f4) f4-typed elements
//         auto offset_A_tile_f4 = start_K_TILE_f4 + K_f4 * (N_TILE * blockIdx.y);
        
//         size_t idx_A4 = 0;
//         for(; idx_A4 < size_A_tile_f4; idx_A4 += nThreadsperBlock){
//             auto col_shift_f4 = (local_tid_1d + idx_A4) % actual_K_TILE_f4;
//             auto row_shift_f4 = (local_tid_1d + idx_A4) / actual_K_TILE_f4;

//             if ( (local_tid_1d + idx_A4) < size_A_tile_f4 ) shmem_f4[local_tid_1d + idx_A4] =
//                                                                          A_f4[offset_A_tile_f4 + col_shift_f4 + K_f4 * row_shift_f4];  
//                                                                          // A[offset_A_tile + col_shift + K * row_shift];
//         }

//         // Vectorized loads of BT_tile from a Row-major BT i.e. Use B_transposed in Row-major or B in Column-major
//         auto size_BT_tile_f4 = actual_K_TILE_f4 * M_TILE;

//         // load BT_tile in the next (M_TILE * K_Tile) elements in row-major order
//         auto offset_BT_tile_f4 = start_K_TILE_f4 + (blockIdx.x * M_TILE) * K_f4;
//         // IMPORTANT: Use `blockIdx.x` for the BT offset computation (above) as the BT row is determined by 
//         // where this block sits along the M dimension (i.e. the column direction of C), which is blockIdx.x.

//         size_t idx_B4 = 0;
//         for(; idx_B4 < size_BT_tile_f4; idx_B4 += nThreadsperBlock){
//             auto linear_tid = local_tid_1d + idx_B4;
//             auto col_shift_f4 = linear_tid % actual_K_TILE_f4; // narrower columns with f4 i.e. for T=double, col_shift_f4 = col_shift_T / 2 = linear_tid % 16 for K_TILE = 32
//             auto row_shift_f4 = linear_tid / actual_K_TILE_f4; // moves faster across the rows i.e. row_shift_f4 = row_shift_T * jump 
//             if ( linear_tid < size_BT_tile_f4 ) {
//                 // Load the f4 element from the row-major BT tile in global memory
//                 float4 val = BT_f4[offset_BT_tile_f4 + col_shift_f4 + K_f4 * row_shift_f4];
//                 // Compute the base index in shared memory for this f4 element
//                 auto base_T = start_BT_tile + (col_shift_f4 * jump) + row_shift_f4 * K_TILE_PAD; // padded with K_TILE_PAD

//                 // Store the f4 element in shared memory as T elements; this is where the padding of K_TILE_PAD comes into play to avoid bank conflicts
//                 T* element = reinterpret_cast<T*>(&val);
//                 for(size_t i=0; i<jump; ++i){
//                     shmem[base_T + i] = element[i];
//                 }

//                 // For example, for K_TILE = 32 and jump = 2 (double):
//                 // - Thread with linear_tid = 0 will load the first f4 element of the BT tile (col_shift_f4=0, row_shift_f4=0) and store it at base_T = start_BT_tile + 0 + 0 * 33
//                 // - Thread with linear_tid = 1 will load the second f4 element of the BT tile (col_shift_f4=1, row_shift_f4=0) and store it at base_T = start_BT_tile + 2 + 0 * 33
//                 // - Thread with linear_tid = 15 will load the 16th f4 element of the BT tile (col_shift_f4=15, row_shift_f4=0) and store it at base_T = start_BT_tile + 30 + 0 * 33
//                 // - Thread with linear_tid = 16 will load the 17th f4 element of the BT tile (col_shift_f4=0, row_shift_f4=1) and store it at base_T = start_BT_tile + 0 + 1 * 33
//                 // --------------------------------------------------------------------------------------------
//                 // - /!\ shmem[32], shmem[65], shmem[98], ... are not used to store the BT tile elements, which creates the padding.
//                 // --------------------------------------------------------------------------------------------
//             }
//         }

//         __syncthreads(); // make sure the tiles are loaded before we start computing
        
//         for(size_t k=0; k < actual_K_TILE; ++k){
//             auto a_base  = (threadIdx.y * Nr) * actual_K_TILE;
//             auto bt_base  = start_BT_tile + (threadIdx.x * Mr) * K_TILE_PAD; // has to be multiplied by K_TILE_PAD to account for the padding in shared memory
            

//             #pragma unroll
//             for(size_t ir=0; ir<Nr; ++ir) a_reg[ir] = alpha * shmem[a_base + k + (ir * actual_K_TILE)];
            
//             #pragma unroll
//             for(size_t jr=0; jr<Mr; ++jr) b_reg[jr] = shmem[bt_base + k + (jr * K_TILE_PAD)];

//             for(size_t ir=0; ir<Nr; ++ir)
//                 for(size_t jr=0; jr<Mr; ++jr){
//                     C_subtile[jr + ir * Mr] += a_reg[ir] * b_reg[jr];
//                 }
//         }

//         __syncthreads(); // wait for all threads to finish before overwriting the shared memory with the next tile
//     }

//     if (gtid_x < M && gtid_y < N) { // guard againt OOB access of C
//         auto block_offset = blockIdx.x * M_TILE + M * (N_TILE * blockIdx.y);
//         auto thread_offset_in_subtile = threadIdx.x * Mr + (threadIdx.y * Nr) * M;
//         auto offset_C_subtile = block_offset + thread_offset_in_subtile;
//         for(size_t ir = 0; ir<Nr; ++ir){
//             for(size_t jr = 0; jr<Mr; ++jr){
//                 if (gtid_x + jr < M && gtid_y + ir < N) {
//                     C[offset_C_subtile + jr + (ir * M)] = beta * C[offset_C_subtile + jr + (ir * M)] + C_subtile[jr + ir * Mr];
//                 }
//             }
//     }
//     }

// }


template <typename T, mx::detail::CudaGemmAlgorithm algo>
__host__ void call_gemm_register_tiled(const T alpha,
                                       const T* __restrict__ dA,
                                       const T* __restrict__ dB,
                                       const T beta,
                                       T* __restrict__ dC,
                                       const size_t N,
                                       const size_t M,
                                       const size_t K,
                                       const CUDA& cuda_policy)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_policy.stream);

    constexpr size_t K_TILE = 16;
    constexpr size_t N_TILE = 128;
    constexpr size_t M_TILE = 128;
    constexpr size_t N_SUBTILE = 8;
    constexpr size_t M_SUBTILE = 4;

    const size_t nThreadsPerBlock_x = (M_TILE + M_SUBTILE - 1) / M_SUBTILE; // 32
    const size_t nThreadsPerBlock_y = (N_TILE + N_SUBTILE - 1) / N_SUBTILE; // 16

    dim3 threadBlock(nThreadsPerBlock_x, nThreadsPerBlock_y, 1); // 512 theads per block for the current setup
    dim3 grid((M + M_TILE - 1) / M_TILE,
              (N + N_TILE - 1) / N_TILE,
              1);

    if constexpr (algo == CudaGemmAlgorithm::RegisterTiling) {
        const size_t shmemBytes = sizeof(T) * (K_TILE * (N_TILE + M_TILE));

        gemm_register_tiled<T, N_SUBTILE, M_SUBTILE>
            <<<grid, threadBlock, shmemBytes, stream>>>(
                alpha, dA, dB, beta, dC, N, M, K, N_TILE, M_TILE, K_TILE);
    }
    else if constexpr (algo == CudaGemmAlgorithm::RegisterTilingVectorized) {
        static_assert(std::is_same_v<T, float>,
                      "Vectorized kernel currently supports float only.");

        constexpr size_t VEC_WIDTH  = sizeof(float4) / sizeof(T);
        constexpr size_t K_TILE_PAD = K_TILE + 1;

        const bool full_k_tiles = (K % K_TILE) == 0;
        const bool vec_aligned   = (K % VEC_WIDTH) == 0;

        if (full_k_tiles && vec_aligned) {
            const size_t shmemBytes =
                sizeof(T) * (N_TILE * K_TILE + M_TILE * K_TILE_PAD);

            gemm_register_tiled_vectorized<T, N_TILE, M_TILE, K_TILE, N_SUBTILE, M_SUBTILE>
                <<<grid, threadBlock, shmemBytes, stream>>>(
                    alpha, dA, dB, beta, dC, N, M, K);
        } else {
            // Fallback to the scalar register-tiled kernel for non-full-K cases
            const size_t shmemBytes = sizeof(T) * (K_TILE * (N_TILE + M_TILE));

            gemm_register_tiled<T, N_SUBTILE, M_SUBTILE>
                <<<grid, threadBlock, shmemBytes, stream>>>(
                    alpha, dA, dB, beta, dC, N, M, K, N_TILE, M_TILE, K_TILE);
        }
    }
}

} // namespace mx::detail

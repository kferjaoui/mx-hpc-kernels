#pragma once
#include <cuda_runtime.h>
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"
#include "mx/utils/device_utils.cuh"
#include "mx_scan/detail/device/blelloch_core.cuh"

namespace mx::detail {

template <typename T>
__global__ void dbg_print(const T* a, size_t n, int level, int k = 1)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int kk = (int)((n < (size_t)k) ? n : (size_t)k);
        printf("dbg level %d (n=%llu): ", level, (unsigned long long)n);
        for (int i = 0; i < kk; ++i) {
            printf("%.10g ", (double)a[i]);
        }
        printf("\n");
    }
}

template<ScanType scan_type,typename T, class Op>
__global__ void blelloch_multiblock_first_pass(T* input,
                                               T* blocksums_array,
                                               size_t size, const Op& op)
{
    int local_tid = threadIdx.x;
    size_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;

    T* shmem = ::mx::device::shared_mem_ptr<T>();
    
    T x = (global_tid < size) ? input[global_tid] : op.identity();
    shmem[local_tid] = x;
    __syncthreads();

    T block_sum = blelloch_core(shmem, blockDim.x, local_tid, op);

    if (local_tid == 0) blocksums_array[blockIdx.x] = block_sum;

    // In-place per-block scan (stores back into `input`)
    if (global_tid < size) {
        if constexpr (scan_type == ScanType::Exclusive) input[global_tid] = shmem[local_tid];
        else                                            input[global_tid] = op(shmem[local_tid], x);
    }

}

template<typename T, class Op>
__global__ void blelloch_multiblock_second_pass(T* __restrict__ scanned_blocksums,
                                                T* __restrict__ upper_level_blocksums,
                                                size_t N, const Op& op)
{
    size_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_tid >= N) return;
    if (blockIdx.x == 0) return;
    
    upper_level_blocksums[global_tid] = 
                    op(scanned_blocksums[blockIdx.x -1], upper_level_blocksums[global_tid]);

}

}
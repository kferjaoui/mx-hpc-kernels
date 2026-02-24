#pragma once
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/device/blelloch_core.cuh"

namespace mx::detail {

// *NOTE*
// `size` must be a power of 2 (Blelloch binary tree structure).
// Sweeps require at most (size/2) active threads (at stride=1),
// but the final shared->global copy requires `size` threads,
// so the kernel requires blockDim.x >= size.
template<ScanType scan_type, typename T, typename Op>
__global__ void blelloch_on_device_monoblock(const T* __restrict__ input, 
                                             T* __restrict__ output,
                                             int size, Op op)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // resolves to {threadIdx.x} if {blockDim.x = 1} 

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh = reinterpret_cast<T*>(smem);

    sh[tid] = (tid < size) ? input[tid] : op.identity();

    T total_sum = blelloch_core(sh, size, tid, op);

    if constexpr (scan_type == ScanType::Exclusive){
        if(tid < size) output[tid] = sh[tid];
    } else { 
        // Inclusive via shared-memory shift + register-cached `total_sum`
        // -> This avoids the reads from global mem in: output[tid] = op(input[tid], sh[tid]); 
        if (tid < (size-1)) output[tid] = sh[tid + 1];
        if (tid == (size - 1)) output[tid] = total_sum;
    }

}

}

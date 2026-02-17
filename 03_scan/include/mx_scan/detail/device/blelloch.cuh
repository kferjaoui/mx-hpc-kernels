#pragma once
#include <cuda_runtime.h>
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"

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
    int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh = reinterpret_cast<T*>(smem);

    sh[tid] = (tid < size) ? input[tid] : op.identity();

    // 1. Up-sweep (Reduction)
    for(int stride = 1; stride < size; stride <<= 1){
        int idx_right = (stride<<1) * (tid + 1) - 1;  // 2 * (tid+1)  * stride - 1
        if(idx_right < size) sh[idx_right] = op(sh[idx_right - stride], sh[idx_right]);
        __syncthreads();
    }

    // 2. Neutralize root (but also save it for ScanType::Inclusive)
    int total_sum = sh[size - 1];
    sh[size - 1] = op.identity();
    __syncthreads();

    //3. Down-Sweep
    for(int stride = (size>>1); stride > 0; stride >>= 1){
        int idx_right = (stride<<1) * (tid + 1) - 1;  // 2 * (tid+1)  * stride - 1
        if(idx_right < size){
            T tmp = sh[idx_right];                            // save right (prefix from above)
            sh[idx_right] = op(tmp, sh[idx_right - stride]);  // right = op(prefix, old_left)
            sh[idx_right - stride] = tmp;                     // left receives prefix from above
        }
        __syncthreads();
    }

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

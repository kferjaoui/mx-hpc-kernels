#pragma once
#include <cuda_runtime.h>
#include "mx/utils/operations.h"

namespace mx::detail {

template<typename T, typename Op>
__device__ __forceinline__ T blelloch_core(T* sh, int size, int tid, const Op& op)
{
    // 1. Up-sweep (Reduction)
    for(int stride = 1; stride < size; stride <<= 1){
        int idx_right = (stride<<1) * (tid + 1) - 1;  // 2 * (tid+1)  * stride - 1
        if(idx_right < size) sh[idx_right] = op(sh[idx_right - stride], sh[idx_right]);
        __syncthreads();
    }
    // if (tid == 0) {
    //     printf("ROOT after upsweep = %.10g\n", (double)sh[size - 1]);
    // }
    // __syncthreads();

    // 2. Save the total block sum (for ScanType::Inclusive) before neutralizinng the root (classic Blelloch)
    T total_sum = sh[size - 1];
    // /!\ CRITICAL: Below barrier is required !!
    // No execution ordering between warps after the previous __syncthreads(),
    // so warp `k` could write sh[size-1]=identity() before warp 0 reads it...
    __syncthreads(); 
    if (tid == 0) sh[size - 1] = op.identity();
    __syncthreads();

    // 3. Down-Sweep
    for(int stride = (size>>1); stride > 0; stride >>= 1){
        int idx_right = (stride<<1) * (tid + 1) - 1;  // 2 * (tid+1)  * stride - 1
        if(idx_right < size){
            T tmp = sh[idx_right];                            // save right (prefix from above)
            sh[idx_right] = op(tmp, sh[idx_right - stride]);  // right = op(prefix, old_left)
            sh[idx_right - stride] = tmp;                     // left receives prefix from above
        }
        __syncthreads();
    }
    return total_sum;
}


} //namespace mx::detail
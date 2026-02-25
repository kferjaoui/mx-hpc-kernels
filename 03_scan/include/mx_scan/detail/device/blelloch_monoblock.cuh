#pragma once
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/device/blelloch_core.cuh"
#include "mx/utils/device_utils.cuh"

namespace mx::detail {

template<ScanType scan_type, typename T, typename Op>
__global__ void blelloch_on_device_monoblock(const T* __restrict__ input, 
                                             T* __restrict__ output,
                                             int logical_n, int tree_n_pow2, const Op& op)
{
    int tid = threadIdx.x;

    T* sh = ::mx::device::shared_mem_ptr<T>();
    sh[tid] = (tid < logical_n) ? input[tid] : op.identity();
    __syncthreads();

    T total_sum = blelloch_core(sh, tree_n_pow2, tid, op);

    if(tid < logical_n) {
        if constexpr (scan_type == ScanType::Exclusive) output[tid] = sh[tid];
        else { 
            // Inclusive via shared-memory shift + register-cached `total_sum`
            // -> This avoids the reads from global mem in: output[tid] = op(input[tid], sh[tid]); 
            if (tid < (logical_n-1))    output[tid] = sh[tid + 1];
            else                        output[tid] = total_sum;
        }
    }

}

} // namespace mx::detail

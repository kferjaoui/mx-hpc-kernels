#pragma once
#include <cuda_runtime.h>
#include "mx/utils/operations.h"
#include "mx/utils/device_utils.cuh"
#include "mx_scan/scan_types.h"

namespace mx::detail {

template<ScanType scan_type, typename T, typename Op>
__global__ void hillis_steele_on_device_monoblock(const T* __restrict__ input,
                                                  T* __restrict__ output,
                                                  int logical_n, int n_pow2, const Op& op)
{
    int tid = threadIdx.x; // 1 block, 1D grid

    T* sh = ::mx::device::shared_mem_ptr<T>();

    // ping-pong buffer indices for the double buffering in shared memory
    int pin  = 0;
    int pout = 1;

    if constexpr (scan_type == ScanType::Exclusive) { 
        // For exclusive scan, we need to shift the input to the right by one 
        // and set the first element to the identity
        sh[tid + pin * n_pow2] = (tid == 0) ? op.identity() 
                                : (tid <= logical_n) ? input[tid - 1] : op.identity();
    } else {
        // For inclusive scan, we can directly load the input into shared memory
        sh[tid + pin * n_pow2] = (tid < logical_n) ? input[tid]: op.identity();
    }
    __syncthreads();

    for (int stride = 1; stride < n_pow2; stride <<= 1) {
        T val = sh[tid + pin * n_pow2];
        if (tid >= stride) {
            val = op(sh[(tid - stride) + pin * n_pow2], val);
        }
        sh[tid + pout * n_pow2] = val;

        __syncthreads();
        int tmp = pin; pin = pout; pout = tmp; // swap ping-pong buffer indices

        // IMPORTANT: At the end of the loop, it is 'pin' that maps to the buffer with the final results of the scan, not 'pout'!
    }

    if (tid < logical_n) output[tid] = sh[tid + pin * n_pow2];
}

}
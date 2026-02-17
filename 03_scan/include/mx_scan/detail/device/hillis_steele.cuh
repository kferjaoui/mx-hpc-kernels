#pragma once
#include <cuda_runtime.h>
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"

namespace mx::detail {

// Only works for pow2 problem sizes that fit within a thread block i.e. <= 1024
template<ScanType scan_type, typename T, typename Op>
__global__ void hillis_steele_on_device_monoblock(const T* __restrict__ input,
                                                  T* __restrict__ output,
                                                  int size, Op op)
{
    int tid = threadIdx.x; // 1 block, 1D grid

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh = reinterpret_cast<T*>(smem);

    // ping-pong buffer indices for the double buffering in shared memory
    int pin  = 0;
    int pout = 1;

    if constexpr (scan_type == ScanType::Exclusive) { 
        // For exclusive scan, we need to shift the input to the right by one and set the first element to the identity
        sh[tid + pin * size] = (tid > 0) ? input[tid - 1] : op.identity();
    } else {
        // For inclusive scan, we can directly load the input into shared memory
        sh[tid + pin * size] = input[tid];
    }
    __syncthreads();

    for (int stride = 1; stride < size; stride <<= 1) {
        T val = sh[tid + pin * size];
        if (tid >= stride) {
            val = op(sh[(tid - stride) + pin * size], val);
        }
        sh[tid + pout * size] = val;

        __syncthreads();
        int tmp = pin; pin = pout; pout = tmp; // swap ping-pong buffer indices

        // IMPORTANT: At the end of the loop, it is 'pin' that maps to the buffer with the final results of the scan, not 'pout'!
    }

    output[tid] = sh[tid + pin * size];
}

}
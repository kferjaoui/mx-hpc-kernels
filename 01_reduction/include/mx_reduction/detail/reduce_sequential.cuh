#pragma once
#include "mx/utils/atomic_ops.cuh"
#include "mx/utils/device_utils.cuh"

namespace mx::detail {

// Shared memory with sequential addressing
// ********************************************************
// Note: Works only for block sizes that are POWER OF TWO
// ********************************************************
template <typename T, class Op>
__global__ void reduce_sequential_addressing(const T* __restrict__ input,
                                            T* __restrict__ result,
                                            const size_t N_elements,
                                            const Op op)
{
    unsigned int local_tid = threadIdx.x;                            // local (per-block) index of the thread
    unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x; // global index of the thread

    T thread_partial = op.identity();
    int stride = blockDim.x * gridDim.x;

    // Per-thread grid-strided reduction
    for(size_t idx = global_tid; idx<N_elements; idx += stride){
        thread_partial = op(thread_partial, input[idx]);
    }
    
    T* sh = ::mx::device::shared_mem_ptr<T>();

    sh[local_tid] = thread_partial;

    __syncthreads();
    
    // Per-block reduction (sequential addressing)
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (local_tid < s){ // only lane-level divergence at warp level
            sh[local_tid] = op(sh[local_tid], sh[local_tid + s]);
        }
        __syncthreads();
    }

    // Reduction across all blocks 
    if (local_tid == 0) {
        // sh[0] holds the per-block reduced value
        ::mx::device::atomicOp(result, sh[0], op);
    }
}

} // namespace mx
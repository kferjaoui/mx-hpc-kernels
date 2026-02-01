#pragma once
#include "mx_reduction/reduce_utils.cuh"

namespace mx{

// Shared memory interleaved addressing reduction kernel WITH divergent branching
template <typename T, class Op>
__global__ void reduce_block_shmem_interleaved_addressing(const T* __restrict__ input,
                                T* __restrict__ result,
                                const size_t n,
                                const Op op)
{
    unsigned int local_tid = threadIdx.x;                            // local (per-block) index of the thread
    unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x; // global index of the thread

    T thread_partial = op.identity();
    int stride = blockDim.x * gridDim.x;

    // Per-thread grid-strided reduction
    for(size_t idx = global_tid; idx<n; idx += stride){
        thread_partial = op(thread_partial, input[idx]);
    }
    
    extern __shared__ T sh[];
    sh[local_tid] = thread_partial;

    __syncthreads();
    
    // Per-block reduction (interleaved addressing)
    for (int d = 1; ( 1<<(d-1) ) < blockDim.x; ++d) {
        if ( (local_tid % (1<<d)) == 0 ){ // Problem: divergent branching within the warp
            sh[local_tid] = op(sh[local_tid], sh[ local_tid + (1<<(d - 1)) ]);
        }
        __syncthreads();
    }

    // Reduction across all blocks 
    if (local_tid == 0) {
        atomicOp(&result[0], sh[0], op);
    }
}

} // namespace mx
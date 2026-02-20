#pragma once
#include "mx/utils/atomic_ops.cuh"

namespace mx::detail {

// Shared memory interleaved addressing reduction kernel WITH divergent branching
// ********************************************************
// Note: Works only for block sizes that are POWER OF TWO
// ********************************************************
template <typename T, class Op>
__global__ void reduce_interleaved_addressing(const T* __restrict__ input,
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
    
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh = reinterpret_cast<T*>(smem);
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
        ::mx::device::atomicOp(&result[0], sh[0], op);
    }
}

} // namespace mx::detail
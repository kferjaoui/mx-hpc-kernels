#pragma once
#include "mx_reduction/reduce_utils.cuh"

namespace mx{

// Warp shuffle reduction with shared memory
template<typename T, class Op>
__global__ void reduce_warp_shuffle(const T* __restrict__ input,
                                    T* __restrict__ result,
                                    const size_t N_elements,
                                    const Op op)
{   
    unsigned int local_tid = threadIdx.x;                            // local (per-block) index of the thread
    unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x; // global index of the thread

    T thread_partial = op.identity();
    int grid_stride = blockDim.x * gridDim.x;

    for(size_t idx = global_tid; idx<N_elements; idx += grid_stride){
        thread_partial = op(thread_partial, input[idx]);
    }
    
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh = reinterpret_cast<T*>(smem);
    sh[local_tid] = thread_partial;

    __syncthreads();
    
    // Reduce in shared memory until we have <= 32 values left
    unsigned int active = blockDim.x;
    while (active > warpSize){
        unsigned half = (active + 1) / 2;
        if (local_tid < half) 
        {   
            unsigned mate = local_tid + half;
            if (mate < active) sh[local_tid] = op(sh[local_tid], sh[mate]); // IMPORTANT: out of bound check 
        }
        __syncthreads();
        active = half;
    }
    
    // In-warp reduce of the last elemnts using __shfl_down_sync() and avoids the heavy __syncthreads()
    // Only threads that have data participate 
    // IMPORTANT: 'active' can be less than warpSize (32)

    if (local_tid < active) {
        T value = sh[local_tid];

        // Stable mask: lanes with local_tid < active
        unsigned full = 0xFFFFFFFFu;
        unsigned mask = __ballot_sync(full, local_tid < active);

        #pragma unroll
        for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
            T other = __shfl_down_sync(mask, value, delta);
            if (local_tid + delta < active) {
                value = op(value, other);
            }
        }

        if (local_tid == 0) atomicOp(result, value, op);
    }

    // ********************************************************
    // INCORRECT IMPL; does not handle {blockDim.x < warpSize} correctly :(
    // Observed UB -> Tested with blockDim.x = 12, gives wrong results for op = Multiply<int>
    // if (local_tid < warpSize)
    // {
    //     T value = (local_tid < active) ? sh[local_tid] : op.identity();
    //     auto mask = __activemask(); // A mask of 32 bits where each bit represents whether the corresponding lane is active
    //     // #pragma unroll
    //     for (size_t delta=warpSize>>1; delta>0; delta>>=1) {
    //         value = op(value, __shfl_down_sync(mask, value, delta));
    //     }
    //     if (local_tid == 0) atomicOp(result, value, op);
    // }
    // ********************************************************

    // DEBUG version: Shared memory reduction (slower)
    // for(unsigned int s=16; s>0; s>>=1){
    //     if ((local_tid + s) < active ) sh[local_tid] = op(sh[local_tid], sh[local_tid + s]);
    //     __syncthreads();
    // }
    // if (local_tid == 0) atomicOp(result, sh[0], op);
}

} //namespace mx
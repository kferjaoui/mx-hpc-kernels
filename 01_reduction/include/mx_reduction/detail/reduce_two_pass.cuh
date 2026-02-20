#pragma once

namespace mx::detail {

template<typename T, class Op>
__global__ void reduce_multiblock_first_pass(const T* __restrict__ input,
                                            T* __restrict__ partials,
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
    T* sh1 = reinterpret_cast<T*>(smem);
    sh1[local_tid] = thread_partial;

    __syncthreads();
    
    // Reduce in shared memory until we have <= 32 values left
    unsigned int active = blockDim.x;
    while (active > warpSize){
        unsigned half = (active + 1) / 2;
        if (local_tid < half) 
        {   
            unsigned mate = local_tid + half;
            if (mate < active) sh1[local_tid] = op(sh1[local_tid], sh1[mate]); // IMPORTANT: out of bound check 
        }
        __syncthreads();
        active = half;
    }
    
    if (local_tid < active) {
        T value = sh1[local_tid];

        unsigned full = 0xFFFFFFFFu;
        unsigned mask = __ballot_sync(full, local_tid < active);

        #pragma unroll
        for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
            T other = __shfl_down_sync(mask, value, delta);
            if (local_tid + delta < active) {
                value = op(value, other);
            }
        }

        if (local_tid == 0) partials[blockIdx.x] = value;
    }
}

template<typename T, class Op>
__global__ void reduce_monoblock_second_pass( T* __restrict__ partials,
                                    T* __restrict__ result,
                                    const size_t grid_size_1st_pass,
                                    const Op op)
{   
    unsigned int tid = threadIdx.x; // local but also global since gridDim.x = 1 and blockIdx.x = 0

    T thread_partial = op.identity();
    int stride = blockDim.x; // monoblock is enough i.e. gridDim.x =1

    for(int index = tid; index < grid_size_1st_pass; index += stride){
        thread_partial = op(thread_partial, partials[index]);
    }

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* sh2 = reinterpret_cast<T*>(smem);
    sh2[tid] = thread_partial;

    __syncthreads();

    // Reduce in shared memory until we have <= 32 values left
    unsigned int active = blockDim.x;
    while (active > warpSize){
        unsigned half = (active + 1) / 2;
        if (tid < half) 
        {   
            unsigned mate = tid + half;
            if (mate < active) sh2[tid] = op(sh2[tid], sh2[mate]);
        }
        __syncthreads();
        active = half;
    }
    
    // Final warp reduction
    if (tid < active) {
        T value = sh2[tid];

        unsigned full = 0xFFFFFFFFu;
        unsigned mask = __ballot_sync(full, tid < active);

        #pragma unroll
        for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
            T other = __shfl_down_sync(mask, value, delta);
            if (tid + delta < active) {
                value = op(value, other);
            }
        }

        if (tid == 0) *result = op(*result, value);
    }
}

} //namespace mx::detail
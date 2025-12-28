#pragma once
#include <cstdlib>
#include <thread>
#include <vector>
#include <barrier>
#include <type_traits>

#include "mx_reduction/operations.h"
#include "mx_reduction/policy.h"

#ifdef __CUDACC__
    #include "cuda_check.h"
#endif

namespace mx{

// Forward declarations:
template <class T, class Op>
T reduce_cpu(const T* input, size_t size, T init, Op op, size_t nThreads);

template <class T, class Op>
T reduce_threads_impl(const T* input, size_t size, T init, Op op, size_t nThreads);

#ifdef __CUDACC__
template <class T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy);

template <typename T, class Op>
__global__ void reduce_baseline(const T* __restrict__ input,
                                T* __restrict__ result,
                                const size_t n,
                                const Op op);
#endif

// Main reduce function
template<typename T, class Op, class Policy>
T reduce(const T* input, size_t size, T init, Op op, Policy policy){

    if constexpr (std::is_same_v<Policy, CPU>) 
    {
        return reduce_cpu(input, size, init, op, policy.threads);
    }
    #ifdef __CUDACC__
    else if constexpr (std::is_same_v<Policy, CUDA>)
    {
        return reduce_cuda(input, size, init, op, policy);
    }
    #endif
    else
    {
        static_assert("mx::reduce: unsupported Policy type");
    }
}

// Default overload: CPU policy
template <typename T, class Op>
T reduce(const T* input, size_t size, T init, Op op){
    CPU policy{};
    return reduce(input, size, init, op, policy); // Calls the main reduce function ''reduce<T, Op, Policy>''
}

template<typename T, class Op>
T reduce_cpu(const T* input, size_t size, T init, Op op, size_t nThreads){
    T result{};
    if (nThreads <= 1) 
    {
        result = init;
        for(size_t i=0; i<size; i++)
            result = op(result, input[i]);
    }
    else 
    {
        // Use threads
        result = reduce_threads_impl(input, size, init, op, nThreads);
    }
    return result;
}

template<typename T, class Op>
T reduce_threads_impl(const T* input, size_t size, T init, Op op, size_t spawnThreads){

    std::vector<std::thread> threads;
    threads.reserve(spawnThreads);
    std::barrier<> sync_point(spawnThreads);
    std::vector<T> vResult(spawnThreads, init);

    auto threadWorker = [&](int threadIdx){
        for(size_t idx=threadIdx; idx<size; idx+=spawnThreads){
            vResult[threadIdx] = op(vResult[threadIdx], input[idx]);
        }

        sync_point.arrive_and_wait();

        for(size_t stride=spawnThreads>>1; stride>0; stride>>=1){
            if (threadIdx < stride) {
                vResult[threadIdx] = op(vResult[threadIdx], vResult[threadIdx+stride]); 
            }
            sync_point.arrive_and_wait();
        }
    };

    for (size_t idx=0; idx<spawnThreads; idx++) 
        threads.emplace_back(threadWorker, idx);

    for(auto& t: threads) t.join();

    return vResult[0];

}

#ifdef __CUDACC__
template<typename T, class Op>
T reduce_cuda(const T* input, size_t size, T init, Op op, const CUDA& cuda_policy){
    
    T result{};
    T* device_input = nullptr;
    T* device_output= nullptr;

    CUDA_CHECK( cudaMalloc( &device_input, size * sizeof(T) ) );
    CUDA_CHECK( cudaMalloc( &device_output, sizeof(T) ) );

    CUDA_CHECK( cudaMemcpy( device_input, input, size * sizeof(T), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( device_output, &init, sizeof(T), cudaMemcpyHostToDevice ) );

    reduce_baseline<<<cuda_policy.grid, cuda_policy.block, cuda_policy.block * sizeof(T)>>>(device_input, device_output, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(device_input);
    cudaFree(device_output);
    
    return result;
}
#endif

} // namespace mx
#pragma once
#include <cstdlib>
#include <thread>
#include <vector>
#include <barrier>
#include <type_traits>

#include "mx_reduction/operations.h"

namespace mx{

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

} // namespace mx
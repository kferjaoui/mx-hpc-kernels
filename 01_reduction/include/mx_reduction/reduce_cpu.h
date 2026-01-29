#pragma once
#include <cstdlib>
#include <thread>
#include <vector>
#include <barrier>
#include <type_traits>

#include "mx_reduction/operations.h"

namespace mx{

template <class T>
struct alignas(64) Padded {
    T value;
};

template<typename T, class Op>
T reduce_threads_impl_blocked_fast(const T* input, size_t size, Op op, size_t spawnThreads);

template<typename T, class Op>
T reduce_threads_impl_strided(const T* input, size_t size, Op op, size_t spawnThreads);

template<typename T, class Op>
T reduce_cpu(const T* input, size_t size, T init, Op op, size_t nThreads){
    T result{init}; // Initialize result with the provided initial value
    if (nThreads <= 1) 
    {
        T a0 = op.identity(), a1 = op.identity(),
          a2 = op.identity(), a3 = op.identity();

        size_t n4 = size & ~0x3; // highest multiple of 4 less than size
        size_t i = 0;
        // Unroll by 4 for vectorization
        for(; i<n4; i += 4){
            a0 = op(a0, input[i]);
            a1 = op(a1, input[i+1]);
            a2 = op(a2, input[i+2]);
            a3 = op(a3, input[i+3]);
        }
        result = op(result, op(op(a0, a1), op(a2, a3)));
        for(; i<size; i++)
            result = op(result, input[i]);
    }
    else 
    {
        // Use threads
        result = op(result, reduce_threads_impl_blocked_fast(input, size, op, nThreads));
    }
    return result;
}

template<typename T, class Op>
T reduce_threads_impl_strided(const T* input, size_t size, Op op, size_t spawnThreads){

    std::vector<std::thread> threads;
    threads.reserve(spawnThreads);
    std::vector<Padded<T>> vResult(spawnThreads, Padded<T>{op.identity()});

    auto threadWorker = [&](int threadIdx){
        T localResult{op.identity()};
        for(size_t idx=threadIdx; idx<size; idx+=spawnThreads){
            localResult = op(localResult, input[idx]);
        }
        vResult[threadIdx].value = localResult;
    };

    for (size_t idx=0; idx<spawnThreads; idx++) 
        threads.emplace_back(threadWorker, idx);

    for(auto& t: threads) t.join();

    // Final reduction on main thread
    T result = op.identity();
    for(size_t i=0; i<spawnThreads; i++){
        result = op(result, vResult[i].value);
    }

    return result;
}

template<typename T, class Op>
T reduce_threads_impl_blocked_fast(const T* input, size_t size, Op op, size_t spawnThreads){

    std::vector<std::thread> threads;
    threads.reserve(spawnThreads);
    std::vector<Padded<T>> vResults(spawnThreads, Padded<T>{op.identity()});

    auto threadWorker = [&](int threadIdx)
    {
        size_t fraction{size/spawnThreads};
        size_t remainder{size%spawnThreads};
        size_t workChunk = fraction + (threadIdx < remainder ? 1 : 0);
        size_t start = threadIdx * fraction + (threadIdx < remainder ? threadIdx : remainder);
        size_t end = start + workChunk;

        // unroll by 4 for each thread
        T l1{op.identity()}, l2{op.identity()},
          l3{op.identity()}, l4{op.identity()};

        size_t end4 = start + ((end - start) & ~0x3); // highest multiple of 4 less than end

        for(size_t i= start; i<end4; i+=4){
            l1 = op(l1, input[i]);
            l2 = op(l2, input[i+1]);
            l3 = op(l3, input[i+2]);
            l4 = op(l4, input[i+3]);
        }

        T localResult = op(op(l1, l2), op(l3, l4));
        for(size_t i=end4; i<end; i++){
            localResult = op(localResult, input[i]);
        }

        vResults[threadIdx].value = localResult;
    };

    for (size_t idx=0; idx<spawnThreads; idx++) 
        threads.emplace_back(threadWorker, idx);
        
    for(auto& t: threads) t.join();
    
    // Final reduction on main thread
    T result = op.identity();
    for(size_t i=0; i<spawnThreads; i++){
        result = op(result, vResults[i].value);
    }
    return result;
}

} // namespace mx
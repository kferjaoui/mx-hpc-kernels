#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <barrier>
#include <algorithm>
#include "mx/utils/operations.h"

namespace mx::detail {

// Core Blelloch: computes in-place the exclusive scan of `in`
// /!\ IMPORTANT: Assumes size is power of two
template<typename T, typename Op>
void scan_blelloch_core(T* in, size_t size, Op op, int nThreads)
{
    std::barrier<> sync_point(nThreads);

    auto workFunction = [&](int tid){
        // 1. up-sweep phase

        // // Easier (for me) to reason on the depth but log2() is not the cleanest way to get the depth...
        // for(int d=1; d<=(int)log2(size); ++d){ // stride is 2^(d-1) increasing with depth (d), where `d=1` is the first level and `d=log2(size)` the last.
        //     size_t offset = (static_cast<size_t>(tid) + 1) * (1ull << d) - 1; // 2^d * (Nthreads+1) - 1
        //     size_t step   = static_cast<size_t>(nThreads) * (1ull << d);      // 2^d *  Nthreads
            
        //     for(size_t idx = offset; idx < size; idx += step){
        //         in[idx] = op(in[idx], in[idx - (1ull << (d-1))]);
        //     }
            
        //     sync_point.arrive_and_wait();
        // }

        // Cleaner way
        for(size_t stride = 1; stride<size; stride <<=1){
            size_t offset = (stride << 1) * (size_t(tid) + 1) - 1; // 2^d * (tid + 1) - 1
            size_t step  = (stride << 1) * nThreads;     // 2^d *  Nthreads
            
            for(size_t idx = offset; idx < size; idx += step){
                in[idx] = op(in[idx - stride], in[idx]);
            }
            
            sync_point.arrive_and_wait();
        }

        // 2. The bridge -> set last element of up-sweep to identity
        if(tid == 0) in[size - 1] = op.identity(); 
        sync_point.arrive_and_wait();

        // 3. down-sweep phase
        for(size_t stride = (size>>1); stride >= 1; stride >>=1){ // decreasing strides: [(size/2), (size/4), ..., 2, 1]
            // sweep from left to right so keep identical (offset, step) pair than for up-sweep
            size_t offset = (stride << 1) * (size_t(tid) + 1) - 1;
            size_t step  = (stride << 1) * nThreads;
            
            for(size_t idx = offset; idx < size; idx += step){ 
                    T tmp = in[idx]; // tmp = right
                    // IMPORTANT: operand order matters unless op is commutative.
                    in[idx] = op(tmp, in[idx - stride]); // right = op(tmp, left)
                    in[idx - stride] = tmp;              // left = tmp
            }
            sync_point.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for(int tid=0; tid<nThreads; ++tid) threads.emplace_back(workFunction, tid);
    for(auto& t: threads) t.join();
}

// Exclusive Blelloch
template<typename T, typename Op>
void exclusive_scan_blelloch(const T* input, T* output, size_t size, Op op, int nThreads)
{   
    // Preserving input but opeation on output which starts as a copy of input's buffer
    std::copy(input, input + size, output);

    scan_blelloch_core(output, size, op, nThreads);

}

// Exclusive Blelloch
template<typename T, typename Op>
void inclusive_scan_blelloch(const T* input, T* output, size_t size, Op op, int nThreads)
{
    exclusive_scan_blelloch(input, output, size, op, nThreads);

    for(size_t i=0; i<size; ++i) output[i] = op(output[i], input[i]);

}

}

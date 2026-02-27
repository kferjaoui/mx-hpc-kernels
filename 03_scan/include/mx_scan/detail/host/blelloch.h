#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <barrier>
#include <algorithm>
#include "mx/utils/operations.h"
#include "mx/utils/next_pow2.h"

namespace mx::detail {

// Core Blelloch: computes in-place the exclusive scan of `in`
template<typename T, typename Op>
T scan_blelloch_core(T* in, size_t size_true, size_t size_padded, const Op& op, int nThreads)
{
    std::vector<T> padded_buf;
    T* in_pw2 = in;
    if(size_padded > size_true){
        padded_buf.resize(size_padded, op.identity());
        std::copy(in, in + size_true, padded_buf.data());
        in_pw2 = padded_buf.data();
    }

    T block_sum;
    std::barrier<> sync_point(nThreads);
    
    auto workFunction = [&](int tid){
        // 1. up-sweep phase
        // stride is 2^(d-1) increasing with depth (d), where `d=1` is the first level and `d=log2(size)` the last.
        for(size_t stride = 1; stride<size_padded; stride <<=1){ 
            size_t offset = (stride << 1) * (size_t(tid) + 1) - 1; // 2^d * (tid + 1) - 1
            size_t step  = (stride << 1) * nThreads;               // 2^d *  Nthreads
            for(size_t idx = offset; idx < size_padded; idx += step){
                in_pw2[idx] = op(in_pw2[idx - stride], in_pw2[idx]);
            }
            sync_point.arrive_and_wait();
        }

        // 2. The bridge -> set last element of up-sweep to identity
        if(tid == 0){
            block_sum = in_pw2[size_padded - 1];
            in_pw2[size_padded - 1] = op.identity();
        }
        sync_point.arrive_and_wait();

        // 3. down-sweep phase
        for(size_t stride = (size_padded>>1); stride >= 1; stride >>=1){ // decreasing strides: [(size/2), (size/4), ..., 2, 1]
            // sweep from left to right so keep identical (offset, step) pair than for up-sweep
            size_t offset = (stride << 1) * (size_t(tid) + 1) - 1;
            size_t step  = (stride << 1) * nThreads;
            
            for(size_t idx = offset; idx < size_padded; idx += step){ 
                    T tmp = in_pw2[idx]; // tmp = right
                    // IMPORTANT: operand order matters unless op is commutative.
                    in_pw2[idx] = op(tmp, in_pw2[idx - stride]); // right = op(tmp, left)
                    in_pw2[idx - stride] = tmp;  // left = tmp
            }
            sync_point.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for(int tid=0; tid<nThreads; ++tid) threads.emplace_back(workFunction, tid);
    for(auto& t: threads) t.join();

    // copy valid results of last block
    if(size_padded > size_true){
        std::copy(in_pw2, in_pw2 + size_true, in);
    }

    return block_sum;

}

// Exclusive Blelloch
template<typename T, typename Op>
void exclusive_scan_blelloch(const T* input, T* output, size_t true_size, const Op& op, int nThreads, size_t block_size = 1024)
// No actual hard limit of 1024 block_size on CPU (as it is the case on Nvidia GPU); just needs to be pow-of-2
{   
    // Preserving input by operating on output which starts as a copy of input's buffer
    std::copy(input, input + true_size, output);

    size_t nBlocks = (true_size + block_size - 1) / block_size;
    std::vector<T> block_sums(nBlocks);

    // 1. Per-block scan + save totals
    for(size_t ib = 0; ib < nBlocks; ++ib){
        size_t bstart = ib * block_size;
        size_t bend = std::min(bstart + block_size, true_size);
        size_t bsize_true = bend - bstart;
        size_t bsize_padded = (size_t)(next_pow2(bsize_true));
        
        // Blelloch scan (exclusive)
        block_sums[ib] = scan_blelloch_core(output + bstart, bsize_true, bsize_padded, op, nThreads);
        
        // // Debug
        // printf("[Block %zu] bsize_true = %zu; bsize_padded = %zu | ", ib, bsize_true, bsize_padded);
        // printf("block_sums[%zu] = %i\n", ib, block_sums[ib]);
    }

    // 2. Exclusive scan of block_sums (Serial)
    T running = op.identity();
    for(size_t ib = 0; ib < nBlocks; ++ib){
        T sum_ib = block_sums[ib];
        block_sums[ib] = running;
        running = op(running, sum_ib);
    }

    // // Debug
    // std::cout << "Exclusive scan of `block_sums` \n";
    // std::cout << "=============================` \n";
    // for(size_t ib = 0; ib < nBlocks; ++ib){
    //     printf("block_sums[%zu] = %i\n", ib, block_sums[ib]);
    // }

    // 3. Add the block's prefix to each block element
    for(size_t ib = 0; ib < nBlocks; ++ib){
        size_t bstart = ib * block_size;
        size_t bend = std::min(bstart + block_size, true_size);
        for(size_t i = bstart; i < bend; ++i){
            output[i] = op(block_sums[ib], output[i]);
        }
    }
}

// Exclusive Blelloch
template<typename T, typename Op>
void inclusive_scan_blelloch(const T* input, T* output, size_t size, const Op& op, int nThreads)
{
    exclusive_scan_blelloch(input, output, size, op, nThreads);

    for(size_t i=0; i<size; ++i) output[i] = op(output[i], input[i]);

}

}

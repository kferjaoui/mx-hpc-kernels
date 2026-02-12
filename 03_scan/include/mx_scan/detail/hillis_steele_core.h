#pragma once
#include <vector>
#include <thread>
#include <barrier>
#include <algorithm>

namespace mx::detail {

// Core Hillis–Steele: computes inclusive scan of `in` into `out`
template<typename T, typename Op>
void scan_hillis_steele_core(T* in, T* out, size_t size, Op op, int nThreads)
{
    T* original_out = out;

    std::barrier<> sync_point(nThreads);

    auto workFunction = [&](int tid){
        for(size_t stride = 1; stride < size; stride <<= 1){ // for depth (d), stride = 2^(d-1)
            for(size_t idx = static_cast<size_t>(tid); idx<size; idx += static_cast<size_t>(nThreads)) 
            {
                if(idx >= stride) out[idx] = op(in[idx], in[idx-stride]);
                else              out[idx] = in[idx];
            }

            sync_point.arrive_and_wait();
            if(tid == 0) std::swap(in, out); // 1 thread does the swapping
            sync_point.arrive_and_wait(); // make sure the swap has occured before scan at next depth
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for(int tid = 0; tid < nThreads; ++tid) threads.emplace_back(workFunction, tid);
    for(auto& t: threads) t.join();

    if(in != original_out){ // After the last swap, `in` should always hold the final scan. If `in` != `original_out`, then copy.
        std::copy(in, in + size, original_out);
    }
}

// Inclusive Hillis–Steele
template<typename T, typename Op>
void inclusive_scan_hillis_steele(const T* input, T* output, size_t size, Op op, int nThreads)
{
    // Duplicate input to avoid in-place scan
    std::vector<T> inputCopy(size); 
    std::copy(input, input + size, inputCopy.begin()); // should be faster then loop

    scan_hillis_steele_core(inputCopy.data(), output, size, op, nThreads);
}

// Exclusive Hillis–Steele
template<typename T, typename Op>
void exclusive_scan_hillis_steele(const T* input, T* output, size_t size, Op op, int nThreads)
{
    // Right-shifted input: [id, x0, x1, ..., x_{n-2}]
    std::vector<T> input_rshifted(size);
    input_rshifted[0] = op.identity();
    for(size_t i=1; i<size; ++i) input_rshifted[i] = input[i-1];

    scan_hillis_steele_core(input_rshifted.data(), output, size, op, nThreads);

}

}

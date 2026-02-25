#pragma once
#include <cassert>
#include <iostream>
#include <vector>
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx/utils/cuda_debug.h"
#include "mx/utils/next_pow2.h"
#include "mx/utils/device_utils.cuh"
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/algorithms.h"
#include "mx_scan/detail/device/blelloch_monoblock.cuh"
#include "mx_scan/detail/device/blelloch_largearray.cuh"
#include "mx_scan/detail/device/hillis_steele_monoblock.cuh"

namespace mx::detail {

int log_blocksize(size_t n, int blocksize, std::vector<size_t>& array_size_at_level){
    int nlevels{1};
    size_t arraysize{n};
    array_size_at_level.clear(); // avoids garbage in case of reuse
    array_size_at_level.push_back(n); // array_size_at_level[0] is `n`
    while(arraysize > blocksize) {
        arraysize = (arraysize + blocksize - 1) / blocksize;
        array_size_at_level.push_back(arraysize);
        ++nlevels; 
    }
    return nlevels;
}

template<ScanType scan_type, ScanAlgorithm scan_algo, typename T, typename Op>
void scan_cuda_device(const T* d_in, T* d_out, size_t n, const Op& op, const CUDA& policy)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(policy.stream);

    if (n <= 1024){
        // Monoblock kernels
        if constexpr (scan_algo == ScanAlgorithm::Hillis_Steele) {
            int logical_n = static_cast<int>(n);
            int n_pow2 = ::mx::next_pow2(logical_n);    
            size_t shmemBytes = 2 * n_pow2 * sizeof(T);
            hillis_steele_on_device_monoblock<scan_type, T, Op>
                    <<<1, n_pow2, shmemBytes, stream>>>(
                        d_in, d_out, logical_n, n_pow2, op);
            ::mx::cuda_debug::post_launch(policy, stream);
            return;
        } else {
            int logical_n = static_cast<int>(n);
            int tree_n_pow2 = ::mx::next_pow2(logical_n);   
            size_t shmemBytes = tree_n_pow2 * sizeof(T);
            blelloch_on_device_monoblock<scan_type, T, Op>
                    <<<1, tree_n_pow2, shmemBytes, stream>>>(
                        d_in, d_out, logical_n, tree_n_pow2, op); 
            ::mx::cuda_debug::post_launch(policy, stream);
            return;           
        }
    }
    // FALLBACK: Blelloch multiblock for large arrays & `n` non-power-of-2

    // no matter the user's chosen blocksize, rescale it to be pow of 2 and <= 1024
    int blocksize = std::min(::mx::next_pow2((int)policy.block), 1024);
    assert((blocksize & (blocksize - 1)) == 0);

    // Pre-compute the number of levels for the recursive scans
    std::vector<size_t> array_size_at_level;
    int nLevels = log_blocksize(n, blocksize, array_size_at_level);
    
    // For degubbing
    assert( array_size_at_level.size() == nLevels );
    
    // Need (at least) as many threads as size `n` i.e. Nblocks = ceil(n/threads_per_block)
    int nBlocks0 = (array_size_at_level[0] + blocksize - 1) / blocksize;

    // copy everything in *dout as scan is in-place
    ::mx::device::copy_array<<<nBlocks0, blocksize, 0, stream>>>(d_in, d_out, n);
    ::mx::cuda_debug::post_launch(policy, stream);

    // // right after fill_array sync, BEFORE the first_pass loop
    // ::mx::detail::dbg_print<<<1,1,0,stream>>>(d_out, n, /*level=*/-1, /*k=*/(int)n);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // Array of pointers to device memory containing blocksums_arrays at each level 
    std::vector<T*> blocksums_array_at_level(nLevels);
    blocksums_array_at_level[0] = d_out;

    // Allocate memory for the blocksums_arrays for each level
    for(int level = 1; level<nLevels; ++level)
    {
        blocksums_array_at_level[level] = nullptr;
        CUDA_CHECK( cudaMalloc(&(blocksums_array_at_level[level]),
                                            array_size_at_level[level] * sizeof(T)) 
        );

        int nb = (array_size_at_level[level] + blocksize - 1) / blocksize;
        ::mx::device::fill_array<<<nb, blocksize, 0, stream>>>(
                blocksums_array_at_level[level], 
                op.identity(), 
                array_size_at_level[level]);
        
        ::mx::cuda_debug::post_launch(policy, stream);
    }
    // // ********
    // // Debugging: Print the pointer attributes for the level 1 blocksums_array (should be device memory) ***************************
    
    // cudaPointerAttributes attr{};
    // CUDA_CHECK(cudaPointerGetAttributes(&attr, blocksums_array_at_level[1]));
    // std::cout << "level1 ptr = " << (void*)blocksums_array_at_level[1]
    // << " type = " << (int)attr.type << "\n";   // expect cudaMemoryTypeDevice
    
    // cudaPointerAttributes attr_out{};
    // CUDA_CHECK(cudaPointerGetAttributes(&attr_out, d_out));
    // printf("d_out type = %d, device = %d\n", (int)attr_out.type, attr_out.device);
    // std::cout << "d_out ptr = " << (void*)d_out << "\n";
    // // ********
    
    //  ***** First pass *****
    //  Recursive Multi-level scan until reaching monoblock launch   
    for(int level = 0; level<(nLevels-1); ++level)
    {   
        // Update the number of blocks to cover array size at next level
        int nBlocks_next_level = (array_size_at_level[level] + blocksize - 1) / blocksize;
        printf("Level %i: Launching kernel with %i blocks of size %i to scan array of size %lu\n", 
            level, nBlocks_next_level, blocksize, array_size_at_level[level]);
        
        // renew allocaion of shared mem at each level
        size_t shmemBytes = blocksize * sizeof(T); 

        if (level == 0) {
            // level 0: scan the actual data with requested scan_type
            ::mx::detail::blelloch_multiblock_first_pass<scan_type, T, Op>
                <<<nBlocks_next_level, blocksize, shmemBytes, stream>>>(
                    blocksums_array_at_level[level],
                    blocksums_array_at_level[level+1],
                    array_size_at_level[level],
                    op);
        } else {
            // levels >= 1: always inclusive scan of block sums (offsets)
            ::mx::detail::blelloch_multiblock_first_pass<ScanType::Inclusive, T, Op>
                <<<nBlocks_next_level, blocksize, shmemBytes, stream>>>(
                    blocksums_array_at_level[level],
                    blocksums_array_at_level[level+1],
                    array_size_at_level[level],
                    op);
        }
        
        ::mx::cuda_debug::post_launch(policy, stream);
        
        if(policy.debug_print){
            ::mx::detail::dbg_print<<<1,1,0,stream>>>(blocksums_array_at_level[level+1],
                                    array_size_at_level[level+1],
                                    level+1,
                                    16); 
                                    // (int)array_size_at_level[level+1]); // ok for small arrays
            
            ::mx::cuda_debug::post_launch(policy, stream);
        }
    }

    // ***** Intermediate pass: Inclusive scan of the bottom level blocksums_array *****

    // `blocksums_bottom_level` is buffer used as input 
    // for the scanning of the bottom level which result
    // will be strored in `array_size_at_level[nLevels-1]`
    T* blocksums_bottom_level = nullptr;
    CUDA_CHECK( cudaMalloc(&(blocksums_bottom_level),
                                        array_size_at_level[nLevels-1] * sizeof(T)) 
    );

    int n_bottom = array_size_at_level[nLevels-1];
    int tree_bottom = std::min(::mx::next_pow2(n_bottom), 1024);
    assert( (tree_bottom <= 1024) && 
                    (tree_bottom & (tree_bottom - 1)) == 0 );

    ::mx::device::copy_array<<<1, tree_bottom, 0, stream>>>(
        blocksums_array_at_level[nLevels-1], blocksums_bottom_level, n_bottom);

    ::mx::cuda_debug::post_launch(policy, stream);

    // // Could use `blocksize` for launch but (can be) wasteful
    // blelloch_on_device_monoblock<ScanType::Inclusive, T, Op>
    //         <<<1, blocksize, blocksize * sizeof(T), stream>>>(
    //             blocksums_bottom_level, 
    //             blocksums_array_at_level[nLevels-1], 
    //             n_bottom, blocksize, op);            
    
    blelloch_on_device_monoblock<ScanType::Inclusive, T, Op>
            <<<1, tree_bottom, tree_bottom * sizeof(T), stream>>>(
                blocksums_bottom_level, 
                blocksums_array_at_level[nLevels-1], 
                n_bottom, tree_bottom, op);

    ::mx::cuda_debug::post_launch(policy, stream);

    // ***** Second pass: Unwind back *****
    for(int level= nLevels-1; level>0; --level)
    {
        int nBlocks = (array_size_at_level[level - 1] + blocksize - 1) / blocksize;
        // No need for shared memory here as it is basically read/write
        // Also no need for scan_type as the logic is taken care ok in first pass
        blelloch_multiblock_second_pass<T, Op>
                <<<nBlocks, blocksize, 0, stream>>>(
                    blocksums_array_at_level[level], 
                    blocksums_array_at_level[level-1],
                    array_size_at_level[level-1], op);

        ::mx::cuda_debug::post_launch(policy, stream);
    }

    // Clear the lower-level pointers i.e. for level in [1, ... , nLevels-1]
    // Note: Level 0 which tracks back to `d_out` is managed by the caller fct mx::scan_cuda()
    for(int level = 1; level<nLevels; ++level) {
        CUDA_CHECK( cudaFree(blocksums_array_at_level[level]) );
    }
    CUDA_CHECK( cudaFree(blocksums_bottom_level) );
}

} // namespace mx::detail

#pragma once
#include <cassert>
#include <iostream>
#include <vector>
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
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
    array_size_at_level.push_back(n); // array_size_at_level[0] is `n`
    while(arraysize > blocksize) {
        arraysize = (arraysize + blocksize - 1) / blocksize;
        array_size_at_level.push_back(arraysize);
        ++nlevels; 
    }
    return nlevels;
}

template<ScanType scan_type, ScanAlgorithm scan_algo, typename T, typename Op>
void scan_cuda_device(const T* d_in, T* d_out, size_t n, Op op, const CUDA& policy)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(policy.stream);

    if (n <= 1024){ // TODO: Power-of-2 check for `n` as well !!
        // Monoblock kernels
        if constexpr (scan_algo == ScanAlgorithm::Hillis_Steele) {    
            int block = static_cast<int>(n);   
            size_t shmemBytes = 2 * n * sizeof(T);
            hillis_steele_on_device_monoblock<scan_type, T, Op>
                    <<<1, block, shmemBytes, stream>>>(d_in, d_out, static_cast<int>(n), op);
        } else {
            int block = static_cast<int>(n);   
            size_t shmemBytes = n * sizeof(T);
            blelloch_on_device_monoblock<scan_type, T, Op>
                    <<<1, block, shmemBytes, stream>>>(d_in, d_out, static_cast<int>(n), op);            
        }
    } else {
        // Blelloch multiblock for large arrays & `n` non-power-of-2
        int blocksize = policy.block;

        // Pre-compute the number of levels for the recursive scans
        std::vector<size_t> array_size_at_level;
        int nLevels = log_blocksize(n, blocksize, array_size_at_level);
        
        // For degubbing
        assert( array_size_at_level.size() == nLevels );
        
        // Need (at least) as many threads as size `n` i.e. Nblocks = ceil(n/threads_per_block)
        int nBlocks0 = (array_size_at_level[0] + blocksize - 1) / blocksize;

        // copy everything in *dout as scan is in-place
        ::mx::device::copy_array<<<nBlocks0, blocksize, 0, stream>>>(d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

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

            ::mx::device::fill_array<<< (array_size_at_level[level] + blocksize - 1) / blocksize, blocksize, 0, stream>>>(
                blocksums_array_at_level[level], op.identity(), array_size_at_level[level] 
            );
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        // ********
        // Debugging: Print the pointer attributes for the level 1 blocksums_array (should be device memory) ***************************
        
        cudaPointerAttributes attr{};
        CUDA_CHECK(cudaPointerGetAttributes(&attr, blocksums_array_at_level[1]));
        std::cout << "level1 ptr = " << (void*)blocksums_array_at_level[1]
        << " type = " << (int)attr.type << "\n";   // expect cudaMemoryTypeDevice
        
        cudaPointerAttributes attr_out{};
        CUDA_CHECK(cudaPointerGetAttributes(&attr_out, d_out));
        printf("d_out type = %d, device = %d\n", (int)attr_out.type, attr_out.device);
        std::cout << "d_out ptr = " << (void*)d_out << "\n";
        // ********
        
        // First pass: Recursive Multi-level scan until reaching monoblock launch where {array size of block sums <= number_threads_per_block}   
        for(int level = 0; level<(nLevels-1); ++level)
        {   
            // Update the number of blocks to cover array size at next level
            int nBlocks_next_level = (array_size_at_level[level] + blocksize - 1) / blocksize;
            printf("Level %i: Launching kernel with %i blocks of size %i to scan array of size %lu\n", level, nBlocks_next_level, blocksize, array_size_at_level[level]);
            // renew allocaion of shared mem at each level
            size_t shmemBytes = blocksize * sizeof(T); 
            
            ::mx::detail::blelloch_multiblock_first_pass<scan_type, T, Op><<<nBlocks_next_level, blocksize, shmemBytes, stream>>>(
                        blocksums_array_at_level[level], blocksums_array_at_level[level+1], array_size_at_level[level], op
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));

            ::mx::detail::dbg_print<<<1,1,0,stream>>>(blocksums_array_at_level[level+1],
                                    array_size_at_level[level+1],
                                    level+1, 
                                    (int)array_size_at_level[level+1]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));

        }

        // Intermediate pass: Inclusive scan of the bottom level blocksums_array
        T* blocksums_bottom_level = nullptr;
        CUDA_CHECK( cudaMalloc(&(blocksums_bottom_level),
                                            array_size_at_level[nLevels-1] * sizeof(T)) 
        );

        ::mx::device::copy_array<<<1, blocksize, 0, stream>>>(blocksums_array_at_level[nLevels-1], blocksums_bottom_level, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        blelloch_on_device_monoblock<ScanType::Inclusive, T, Op>
                <<<1, blocksize, blocksize * sizeof(T), stream>>>(blocksums_bottom_level, blocksums_array_at_level[nLevels-1], blocksize, op);            

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Second pass: Unwind back
        for(int level= nLevels-1; level>0; --level)
        {
            int nBlocks = (array_size_at_level[level - 1] + blocksize - 1) / blocksize;
            // No need for shared memory here as it is basically read/write
            // Also no need for scan_type as the logic is taken care ok in first pass
            blelloch_multiblock_second_pass<T, Op>
                    <<<nBlocks, blocksize, 0, stream>>>(blocksums_array_at_level[level], 
                                                    blocksums_array_at_level[level-1],
                                                    array_size_at_level[level-1], op);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));

        }

        // Clear the lower-level pointers i.e. for level in [1, ... , nLevels-1]
        // Note: Level 0 which tracks back to `d_out` is managed by the caller fct mx::scan_cuda()
        for(int level = 1; level<nLevels; ++level) {
            CUDA_CHECK( cudaFree(blocksums_array_at_level[level]) );
        }
        CUDA_CHECK( cudaFree(blocksums_bottom_level) );

    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace mx::detail

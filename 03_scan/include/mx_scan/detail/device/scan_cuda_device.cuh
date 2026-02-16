#pragma once
#include <cassert>
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/algorithms.h"
#include "mx_scan/detail/device/blelloch.cuh"
#include "mx_scan/detail/device/hillis_steele.cuh"

namespace mx::detail {

template<ScanType scan_type, ScanAlgorithm scan_algo, typename T, typename Op>
void scan_cuda_device(const T* d_in, T* d_out, size_t n, Op op, const CUDA& policy)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(policy.stream);

    // For now: monoblock only
    if constexpr (scan_algo == ScanAlgorithm::Hillis_Steele) {
        // Guard: Hillis monoblock requires n <= max threads per block
        // can also fetch device properties and check against that; for now just hardcoded 1024 
        if (n > 1024) {
            // More pragmatic -> fall back to Blelloch later; for now throw/assert
            assert(false && "Hillis-Steele monoblock supports n <= 1024");
            // return;
        }

        int block = static_cast<int>(n);
        dim3 grid(1,1,1);

        size_t shmemBytes = 2 * n * sizeof(T);

        hillis_steele_on_device_monoblock<scan_type, T, Op><<<grid, block, shmemBytes, stream>>>(d_in, d_out, n, op);

    }
    else {
        // Blelloch monoblock/multiblock later
        // For now just return
        return;
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace mx::detail

#pragma once
#include <vector>

#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/algorithms.h"
#include "mx_scan/detail/hillis_steele_core.h"
#include "mx_scan/detail/blelloch_core.h"

namespace mx::detail {

template<ScanType scan_type, ScanAlgorithm scan_algo = ScanAlgorithm::Blelloch,  typename T, typename Op>
void scan_parallel(const T* input, T* output, size_t size, Op op, int nThreads)
{
    if constexpr (scan_type == ScanType::Exclusive)
    {
        if constexpr (scan_algo == ScanAlgorithm::Hillis_Steele) exclusive_scan_hillis_steele(input, output, size, op, nThreads);
        else exclusive_scan_blelloch(input, output, size, op, nThreads);
    } 
    else if constexpr (scan_type == ScanType::Inclusive)
    {   
        if constexpr (scan_algo == ScanAlgorithm::Hillis_Steele) inclusive_scan_hillis_steele(input, output, size, op, nThreads);
        else inclusive_scan_blelloch(input, output, size, op, nThreads);
    } 
    else 
    {
        static_assert(scan_type != scan_type, "Unknown ScanType variant");
    }
}

} // namespace mx::detail
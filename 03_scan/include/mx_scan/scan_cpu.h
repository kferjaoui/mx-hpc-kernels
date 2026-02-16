#pragma once
#include "mx_scan/detail/host/scan_serial.h"
#include "mx_scan/detail/host/scan_parallel.h"

namespace mx {

template<ScanType scan_type, detail::ScanAlgorithm scan_algo = detail::ScanAlgorithm::Blelloch, typename T, typename Op>
void scan_cpu(const T* input, T* output, size_t size, Op op, int nThreads = 1){
    if (size == 0) return;

    if (nThreads <= 1) 
    {
        if (nThreads <= 0) nThreads = 1;
        detail::scan_serial<scan_type>(input, output, size, op);
    }
    else 
    {
        detail::scan_parallel<scan_type, scan_algo>(input, output, size, op, nThreads);
    }
}

} // namespace mx
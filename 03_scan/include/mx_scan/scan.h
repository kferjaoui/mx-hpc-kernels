#pragma once
#include <type_traits>
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_scan/scan_cpu.h"
#include "mx_scan/scan_cuda.h"

namespace mx{

template <class> 
inline constexpr bool always_false_v = false;

// Main reduce function
template<ScanType scan_type, detail::ScanAlgorithm scan_algo = detail::ScanAlgorithm::Blelloch, typename T, class Op, class Policy>
void scan(const T* input, T* output, size_t size, Op op, Policy policy){

    if constexpr (std::is_same_v<Policy, CPU>)
    {
        scan_cpu<scan_type, scan_algo>(input, output, size, op, policy.threads);
    }
    else if constexpr (std::is_same_v<Policy, CUDA>)
    {
        scan_cuda<scan_type, scan_algo>(input, output, size, op, policy);
    }
    else
    {
        static_assert(always_false_v<Policy>, "mx::scan: unsupported Policy type");
    }
}

// Default overload: CPU policy
template <typename T, class Op>
void scan(const T* input, size_t size, Op op){
    CPU serial_policy{};
    scan(input, size, op, serial_policy); // Calls the main scan function ''scan<T, Op, Policy>''
}

} // namespace mx
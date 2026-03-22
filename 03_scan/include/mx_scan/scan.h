#pragma once
#include <type_traits>
#include "mx/utils/meta.h"
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_scan/scan_cpu.h"
#include "mx_scan/scan_cuda.h"

namespace mx{

// Main scan function
template<ScanType scan_type, 
        detail::ScanAlgorithm scan_algo = detail::ScanAlgorithm::Blelloch, 
        typename T, 
        ::mx::BinaryOp<T> Op = Sum<T>, 
        class Policy = CPU<>>
void scan(const T* input, T* output, size_t size, Op op, Policy policy = Policy{}){

    if constexpr (is_cpu_policy_v<Policy>)
    {
        scan_cpu<scan_type, scan_algo>(input, output, size, op, policy.threads);
    }
    else if constexpr (is_cuda_policy_v<Policy>)
    {
        scan_cuda<scan_type, scan_algo>(input, output, size, op, policy);
    }
    else
    {
        static_assert(always_false_v<Policy>, "mx::scan: unsupported Policy type");
    }
}

} // namespace mx
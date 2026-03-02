#pragma once
#include "mx/utils/policy.h"
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"
#include "mx_scan/detail/algorithms.h"

namespace mx {

template<ScanType scan_type, detail::ScanAlgorithm scan_algo = detail::ScanAlgorithm::Blelloch, typename T, typename Op>
void scan_cuda(const T* input, T* output, size_t size, const Op& op, const CUDA& cuda_policy);

} // namespace mx
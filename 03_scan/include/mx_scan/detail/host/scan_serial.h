#pragma once
#include <cstddef>
#include "mx/utils/operations.h"
#include "mx_scan/scan_types.h"

namespace mx::detail {

template<ScanType scan_type, typename T, typename Op>
void scan_serial(const T* input, T* output, size_t size, const Op& op)
{
    if constexpr (scan_type == ScanType::Exclusive)
    {
        output[0] = op.identity();
        for(size_t idx=1; idx<size; ++idx)
        {
            output[idx] = op(output[idx-1], input[idx-1]); 
        }
    } 
    else if constexpr (scan_type == ScanType::Inclusive)
    {
        output[0] = input[0];
        for(size_t idx=1; idx<size; ++idx)
        {
            output[idx] = op(output[idx-1], input[idx]); 
        }
    } 
    else 
    {
        static_assert(scan_type != scan_type, "Unknown ScanType variant");
    }
}

} // namespace mx::detail
#pragma once
#include <cstdint>

namespace mx{
    
struct CPU {
    int threads = 0;
};

struct CUDA {
    std::uint32_t block = 256;
    std::uint32_t grid_x = 2048;
    std::uint32_t grid_y = 1;
    std::uint32_t grid_z = 1;
    std::uintptr_t stream = 0; // 0 == default stream
};

}
#pragma once
#include <cuda_runtime.h>

namespace mx{

struct CPU {
    int threads = 0;
};

struct CUDA {
    const int block = 256;
    dim3 grid{4, 1, 1};
    cudaStream_t stream = 0;
};

}
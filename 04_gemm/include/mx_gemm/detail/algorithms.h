#pragma once

namespace mx::detail {

enum class GemmAlgorithm {
    Naive,
    Transposed,
    CacheBlocked,
    Microkernel,          // register blocking with micro-tiling
    VectorizedMicrokernel // simd vectorized microkernel
};

enum class CudaGemmAlgorithm {
    Naive,
    SharedMemTiling,
    RegisterTiling,
    RegisterTilingVectorized,
    WarpTiling,
    TensorCores
};

}
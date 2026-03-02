#pragma once
#include "mx/utils/meta.h"

#ifdef __CUDACC__
    // proper import of numeric_limits for CUDA device code
    #include <cuda/std/limits>
    namespace num = cuda::std;
#else
    // Define __host__ and __device__ for non-CUDA compilation
    #include <limits>
    namespace num = std;
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
#endif

namespace mx{

template <Addable T>
struct Sum{
    __host__ __device__ constexpr T operator()(T a, T b) const noexcept { return a + b; }
    __host__ __device__ static constexpr T identity() noexcept { return T{0}; }
};

template <Multipliable T>
struct Multiply{
    __host__ __device__ constexpr T operator()(T a, T b) const noexcept { return a * b; }
    __host__ __device__ static constexpr T identity() noexcept { return T{1}; }
};

template <Comparable T>
struct Max{
    __host__ __device__ constexpr T operator()(T a, T b) const noexcept { return (a > b) ? a : b; }
    __host__ __device__ static constexpr T identity() noexcept { return num::numeric_limits<T>::lowest(); }
};

template <Comparable T>
struct Min{
    __host__ __device__ constexpr T operator()(T a, T b) const noexcept { return (a < b) ? a : b; }
    __host__ __device__ static constexpr T identity() noexcept { return num::numeric_limits<T>::max(); }
};

}
#pragma once

// Define __host__ and __device__ for non-CUDA compilation
#ifndef __CUDACC__
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
#endif

namespace mx{

template <typename T>
struct Sum{
    __host__ __device__ T operator()(T a, T b) const { return a + b; }
    __host__ __device__ T identity() const { return T{0}; }
};

template <typename T>
struct Multiply{
    __host__ __device__ T operator()(T a, T b) const { return a * b; }
    __host__ __device__ T identity() const { return T{1}; }
};

template <typename T>
struct Max{
    __host__ __device__ T operator()(T a, T b) const { return (a > b) ? a : b; }
    __host__ __device__ T identity() const { return std::numeric_limits<T>::lowest(); }
};

template <typename T>
struct Min{
    __host__ __device__ T operator()(T a, T b) const { return (a < b) ? a : b; }
    __host__ __device__ T identity() const { return std::numeric_limits<T>::max(); }
};

}
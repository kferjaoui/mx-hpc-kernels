#pragma once
#include <cstddef>

template<typename T>
__global__ void dotproduct_singleblock_warp_downsweep(const T* __restrict__ x,
                                                      const T* __restrict__ y,
                                                      const size_t n, 
                                                      T* __restrict__ result);

template<typename T>
__global__ void dotproduct_multiblock_warp_downsweep(const T* __restrict__ x,
                                                     const T* __restrict__ y,
                                                     const size_t n,
                                                     T* __restrict__ result);

#pragma once
#include <cuda_runtime.h>

namespace mx {

// ***************
// Multiply atomic
// ***************
// Implementation of atomic multiplication for single/double floating-point types
// using atomicCAS (compare-and-swap) loop since CUDA does not provide

__device__ float atomicMul(float* address, float val) {
    // float & unsigned int are both 32 bits
    unsigned int* addr_as_ui = (unsigned int*)(address); 
    
    unsigned int old = atomicAdd(addr_as_ui, 0); // Atomic read
    unsigned int assumed;

    do {
        assumed = old;
        float old_f = __uint_as_float(assumed);
        float desired_f = old_f * val;
        unsigned int desired_ui = __float_as_uint(desired_f);
        old = atomicCAS(addr_as_ui, assumed, desired_ui);
    } while (old != assumed);

    // Returns value before the atomic multiplication 
    return __uint_as_float(old);
}

__device__ double atomicMul(double* address, double val) {
    // double & unsigned long long are both 64 bits
    unsigned long long* addr_as_ull = (unsigned long long*)(address); 
    
    unsigned long long old = atomicAdd(addr_as_ull, 0);
    unsigned long long assumed;

    do { 
        assumed = old;
        double old_f = __longlong_as_double(assumed);
        double desired_f = old_f * val;
        unsigned long long desired_ui = __double_as_longlong(desired_f);
        old = atomicCAS(addr_as_ull, assumed, desired_ui);
    } while (old != assumed);

    return __longlong_as_double(old);
}

// ***************
// Min/Max atomics
// ***************
// Implementation of atomic Min/Max for single/double floating-point types
// using atomicCAS (compare-and-swap) loop since CUDA does not provide

__device__ float atomicMax_fp(float* address, float val) {
    unsigned int* addr_as_ui = (unsigned int*)(address); 
    unsigned int old = atomicAdd(addr_as_ui, 0);
    unsigned int assumed;

    do {
        assumed = old;
        float old_f = __uint_as_float(assumed);
        float desired_f = fmaxf(old_f, val); // IEEE 754 compliant i.e. deals with NaNs
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(desired_f));
    } while (old != assumed);

    return __uint_as_float(old);
}

__device__ double atomicMax_fp(double* address, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)(address); 
    unsigned long long old = atomicAdd(addr_as_ull, 0);
    unsigned long long assumed;

    do { 
        assumed = old;
        double old_f = __longlong_as_double(assumed);
        double desired_f = fmaxf(old_f, val); // IEEE 754 compliant i.e. deals with NaNs
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(desired_f));
    } while (old != assumed);

    return __longlong_as_double(old);
}


__device__ float atomicMin_fp(float* address, float val) {
    unsigned int* addr_as_ui = (unsigned int*)(address); 
    unsigned int old = atomicAdd(addr_as_ui, 0);
    unsigned int assumed;

    do {
        assumed = old;
        float old_f = __uint_as_float(assumed);
        float desired_f = fminf(old_f, val); // IEEE 754 compliant i.e. deals with NaNs
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(desired_f));
    } while (old != assumed);

    return __uint_as_float(old);
}

__device__ double atomicMin_fp(double* address, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)(address); 
    unsigned long long old = atomicAdd(addr_as_ull, 0);
    unsigned long long assumed;

    do { 
        assumed = old;
        double old_f = __longlong_as_double(assumed);
        double desired_f = fminf(old_f, val); // IEEE 754 compliant i.e. deals with NaNs
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(desired_f));
    } while (old != assumed);

    return __longlong_as_double(old);
}

} // namespace mx
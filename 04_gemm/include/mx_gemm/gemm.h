#pragma once
#include <type_traits>

#include "mx/utils/meta.h"
#include "mx/utils/policy.h"
#include "mx/utils/schedulers.h"
#include "mx/layout.h"
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx_gemm/detail/algorithms.h"
#include "mx_gemm/gemm_cpu.h"
// #include "mx_gemm/gemm_cuda.h"


namespace mx{

// Dense overload forwards to views
template<
    detail::GemmAlgorithm gemm_algo = detail::GemmAlgorithm::VectorizedMicrokernel, 
    typename T, 
    class Layout = RowMajor,
    class Policy = CPU<> >
void gemm(const T alpha, 
        const Dense<T, Layout>& A,
        const Dense<T, Layout>& B,
        const T beta, 
        Dense<T, Layout>& C, 
        Policy policy = Policy{})
{
    gemm<gemm_algo>(alpha, A.view(), B.view(), beta, C.view(), policy);
}

// CPU overload (Views)
template<
    detail::GemmAlgorithm gemm_algo = detail::GemmAlgorithm::VectorizedMicrokernel, 
    typename T, 
    class Layout = RowMajor,
    class Scheduler >
void gemm(const T alpha, 
        DenseView<const T, Layout> A, 
        DenseView<const T, Layout> B,
        const T beta, 
        DenseView<T, Layout> C, 
        CPU<Scheduler> policy)
{
        gemm_cpu<gemm_algo>(alpha, A, B, beta, C, policy.threads, policy.scheduler);
}

// CUDA overload (Views)
template<
    detail::CudaGemmAlgorithm gemm_algo = detail::CudaGemmAlgorithm::Naive, 
    typename T, 
    class Layout = RowMajor >
void gemm(const T alpha, 
        DenseView<const T, Layout> A, 
        DenseView<const T, Layout> B,
        const T beta, 
        DenseView<T, Layout> C, 
        CUDA policy)
{
        gemm_cuda<gemm_algo>(alpha, A, B, beta, C, policy);
}

// Default overload: CPU policy
template<
    detail::GemmAlgorithm gemm_algo = detail::GemmAlgorithm::VectorizedMicrokernel, 
    typename T, 
    class Layout = RowMajor >
void gemm(const T alpha, 
        DenseView<const T, Layout> A, 
        DenseView<const T, Layout> B,
        const T beta, 
        DenseView<T, Layout> C)
{
        gemm<gemm_algo>(alpha, A, B, beta, C, CPU<>{});
}

} // namespace mx

#pragma once
#include <type_traits>
#include "mx/utils/meta.h"
#include "mx/utils/schedulers.h"
#include "mx_gemm/detail/algorithms.h"
#include "mx_gemm/detail/host/gemm_naive.h"
#include "mx_gemm/detail/host/gemm_transposed.h"
#include "mx_gemm/detail/host/gemm_cache_blocked.h"
#include "mx_gemm/detail/host/gemm_microkernel.h"
#include "mx_gemm/detail/host/gemm_simd_vectorized.h"

namespace mx {

template<detail::GemmAlgorithm gemm_algo = detail::GemmAlgorithm::VectorizedMicrokernel, 
        typename T,  
        class Layout = RowMajor,
        class Scheduler = BlockScheduler>
void gemm_cpu(const T alpha, 
            DenseView<const T, Layout> A, 
            DenseView<const T, Layout> B,
            const T beta, DenseView<T, Layout> C, 
            int nThreads = 1,
            Scheduler sched = Scheduler{})
{
    if constexpr (gemm_algo == detail::GemmAlgorithm::Naive) {
        gemm_naive(alpha, A, B, beta, C, nThreads, sched);
    } else if constexpr (gemm_algo == detail::GemmAlgorithm::Transposed) {
        gemm_transposed(alpha, A, B, beta, C, nThreads, sched);
    } else if constexpr (gemm_algo == detail::GemmAlgorithm::CacheBlocked) {
        gemm_cache_blocked(alpha, A, B, beta, C, nThreads, sched);
    } else if constexpr (gemm_algo == detail::GemmAlgorithm::Microkernel) {
        gemm_microkernel(alpha, A, B, beta, C, nThreads);
    } else if constexpr (gemm_algo == detail::GemmAlgorithm::VectorizedMicrokernel) {
        gemm_vectorized(alpha, A, B, beta, C, nThreads, sched);
    } else {
        static_assert(always_false_value_v<gemm_algo>, "Unsupported GemmAlgorithm");
    }
}

} // namespace mx
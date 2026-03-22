#pragma once
#include <stdexcept>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/layout.h"
#include "mx/types.h"
#include "mx/utils/parallel.h"
#include "mx/utils/schedulers.h"

namespace mx{
    
template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_naive(const T alpha,
                const Dense<T, Layout>& A,
                const Dense<T, Layout>& B,
                const T beta, Dense<T, Layout>& C,
                index_t nthreads = 1,
                Scheduler sched = Scheduler{})
{
    gemm_naive(alpha, A.view(), B.view(), beta, C.view(), nthreads, sched);
}

template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_naive(const T alpha,
                DenseView<const T, Layout> A,
                DenseView<const T, Layout> B,
                const T beta, DenseView<T, Layout> C,
                index_t nthreads = 1, 
                Scheduler  sched = Scheduler{})
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    if (K != B.rows() || N != C.rows() || M != C.cols()) {
        throw std::runtime_error("Mismatch in matrix sizes; cannot perform multiplication.");
    }

    // Clamp thread count
    if (nthreads < 1) {
        nthreads = 1;
    }

    // Never use more threads than rows
    if (nthreads > N) {
        nthreads = (N > 0) ? N : 1;
    }

    // Matrix multiplication lambda with row-wise partitioning of C
    auto mm = [&](index_t tid){
        sched(tid, nthreads, N, 
            [&](index_t i) {
                for(index_t j=0; j<M; j++){
                    T sum{};
                    for(index_t k=0; k<K; k++){
                        sum += A(i,k)*B(k,j); 
                    }
                    C(i,j) = alpha * sum + beta * C(i,j);
                }
            }
        );
    };

    if (nthreads == 1) { // avoid spawning threads for sequential execution
        mm(0);
    } else {
        launch_threads(mm, nthreads);
    }
}

} // namespace mx
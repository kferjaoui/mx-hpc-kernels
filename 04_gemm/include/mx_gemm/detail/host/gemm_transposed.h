#pragma once
#include <stdexcept>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/layout.h"
#include "mx/transpose.h"
#include "mx/scale_matrix.h"
#include "mx/utils/parallel.h"
#include "mx/utils/schedulers.h"

namespace mx{
    
template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_transposed(const T alpha,
                    const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    const T beta, Dense<T, Layout>& C,
                    index_t nthreads = 1,
                    Scheduler sched = Scheduler{})
{
    gemm_transposed(alpha, A.view(), B.view(), beta, C.view(), nthreads, sched);
}


template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_transposed(const T alpha, 
                    DenseView<const T, Layout> A,
                    DenseView<const T, Layout> B,
                    const T beta, DenseView<T, Layout> C,
                    index_t nthreads = 1,
                    Scheduler sched = Scheduler{})
{
    
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();
    
    if (K != B.rows() || N != C.rows() || M != C.cols()) {
        throw std::runtime_error("Mismatch in matrix sizes; cannot perform multiplication.");
    }
        
    // Scale C + Quick out for alpha=0
    scale_matrix(beta, C);
    if (alpha == T{0}) return;

    constexpr index_t TILE_L1 = 32; // decent tile size for my 32 KB L1 cache (for 8B double precision) in order to avoid spills
    
    using AView = DenseView<const T, Layout>;

    const index_t work_item = AView::is_row_major ? N : M ; 
    if (nthreads < 1) nthreads = 1;
    if (nthreads > work_item) nthreads = (work_item > 0 ? work_item : 1);

    if constexpr (AView::is_row_major) {

        // Materialize the B_transposed
        Dense<T, RowMajor> BT(M,K);
        transpose_matrix_tiled(B, BT.view(), TILE_L1);

        if (nthreads < 1) {
            nthreads = 1;
        }

        if (nthreads > N) {
            nthreads = (N > 0) ? N : 1;
        }
        
        auto mm = [&](index_t tid){
            // RowMajor: First loop on rows `i` offers better locality for C
            sched(tid, nthreads, N, 
                [&](index_t i) {
                    for(index_t j=0; j<M; ++j){
                        T sum{};
                        for(index_t k=0; k<K; ++k){
                            sum += A(i,k) * BT(j,k);
                        }
                        C(i,j) += alpha * sum;
                    }
                }
            );
        };

        if (nthreads == 1) mm(0);
        else launch_threads(mm, nthreads);

    } else{

        // Materialize the A_transposed
        Dense<T, ColMajor> AT(K, N);
        transpose_matrix_tiled(A, AT.view(), TILE_L1);

        auto mm = [&](index_t tid){
            // ColMajor: First loop on columns `j` offers better locality for C
            sched(tid, nthreads, M, 
                [&](index_t j) {
                    for(index_t i=0; i<N; ++i){
                        T sum{};
                        for(index_t k=0; k<K; ++k){
                            sum += AT(k,i) * B(k,j);
                        }
                        C(i,j) += alpha * sum;
                    }
                }
            );
        };

        if (nthreads == 1) mm(0);
        else launch_threads(mm, nthreads);

    }
}

} // namespace mx
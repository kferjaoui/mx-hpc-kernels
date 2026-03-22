#pragma once
#include<stdexcept>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/transpose.h"
#include "mx/scale_matrix.h"
#include "mx/utils/parallel.h"
#include "mx/utils/schedulers.h"

namespace mx{

template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_cache_blocked(const T alpha,
                        const Dense<T, Layout>& A,
                        const Dense<T, Layout>& B,
                        const T beta, Dense<T, Layout>& C,
                        index_t nthreads = 1,
                        Scheduler sched = Scheduler{})
{
    gemm_cache_blocked(alpha, A.view(), B.view(), beta, C.view(), nthreads, sched);
}


template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_cache_blocked(const T alpha, 
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
    if (N == 0 || M == 0 || K == 0) return;

    scale_matrix(beta, C);
    if (alpha == T{0}) return;

    // Cache blocking parameters for one-level tiling.
    // Choose nc/kc/mc so A-tile + transposed tile + C-tile remain comfortably L2-resident.
    // Values below were tuned for a ~1 MiB L2 cache.
    const index_t nc = 256; // rows of A/C per tile
    const index_t kc = 128; // shared K-depth per tile
    const index_t mc = 160; // cols of B/C per tile

    const index_t Nb = (N + nc - 1) / nc; // number of row tiles of C

    if (nthreads < 1) nthreads = 1;
    if (nthreads > Nb) nthreads = (Nb > 0 ? Nb : 1);

    using AView = DenseView<const T, Layout>;

    if constexpr (AView::is_row_major) {// Row-major: transpose B so k is contiguous in both operands

        Dense<T, Layout> BT(M, K);
        transpose_matrix_tiled(B, BT.view()/* , TILE_L1=32*/);

        auto mm_cache_blocked = [&](index_t tid){
            // Parallelize over row tiles of C.
            // Outer tile order: I-K-J. Inner order: i-j-k.
            sched(tid, nthreads, Nb,
                [&](index_t Ni){
                    // Shape of loop inside the scheduler: for(index_t Ni = C_row_start; Ni<C_row_end; Ni++)  
                    const index_t i0 = Ni*nc; // start row index of C_tile
                    const index_t i_end = std::min<index_t>(N, i0 + nc);
        
                    for (index_t k0 = 0; k0 < K; k0 += kc) {
                        const index_t k_end = std::min<index_t>(K, k0 + kc);
        
                        for (index_t j0 = 0; j0 < M; j0 += mc) {
                            const index_t j_end = std::min<index_t>(M, j0 + mc);
        
                            // Tile-tile multiply {i-j-k}
                            // A(i,k) and BT(j,k) are contiguous in k.
                            // Sweeping j gives contiguous writes in row-major C.
                            for (index_t i = i0; i < i_end; ++i) {
                                for (index_t j = j0; j < j_end; ++j) {
                                    T sum = T{0};
        
                                    // Partial tile C_K(I,J) += A(I,K) * B(K,J) at fixed depth K of size {k0:k_end}
                                    // A(i,k) and BT(j,k) both contiguous in k for RowMajor
                                    for (index_t k = k0; k < k_end; ++k) {
                                        sum += A(i,k) * BT(j,k);
                                    }
                                    C(i,j) += alpha * sum;
                                }
                            }
                        }
                    }
                }
            );
        };

        if (nthreads == 1) mm_cache_blocked(0);
        else launch_threads(mm_cache_blocked, nthreads);

    } else { // Col-major: transpose A so k is contiguous in both operands

        Dense<T, Layout> AT(K, N);
        transpose_matrix_tiled(A, AT.view()/* , TILE_L1=32*/);

        auto mm_cache_blocked = [&](index_t tid){
            // Parallelize over row tiles of C.
            // Outer tile order: I-K-J. Inner order: j-i-k.
            sched(tid, nthreads, Nb,
                [&](index_t Ni){
                    const index_t i0 = Ni*nc;
                    const index_t i_end = std::min<index_t>(N, i0 + nc);

                    for (index_t k0 = 0; k0 < K; k0 += kc) {
                        const index_t k_end = std::min<index_t>(K, k0 + kc);

                        for (index_t j0 = 0; j0 < M; j0 += mc) {
                            const index_t j_end = std::min<index_t>(M, j0 + mc);

                            // Tile-tile multiply {j-i-k}
                            // AT(k,i) and B(k,j) are contiguous in k.
                            // Sweeping i gives contiguous writes in col-major C.
                            for (index_t j = j0; j < j_end; ++j) {
                                    for (index_t i = i0; i < i_end; ++i) {
                                    T sum = T{0};

                                    for (index_t k = k0; k < k_end; ++k) {
                                        sum += AT(k,i) * B(k,j);
                                    }
                                    C(i,j) += alpha * sum;
                                }
                            }
                        }
                    }
                }
            );
        };
        
        if (nthreads == 1) mm_cache_blocked(0);
        else launch_threads(mm_cache_blocked, nthreads);
    }

}


} // namespace mx
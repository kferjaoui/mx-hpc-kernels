#pragma once
#include<stdexcept>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/transpose.h"
#include "mx/scale_matrix.h"
#include "mx/utils/parallel.h"
#include "mx/utils/schedulers.h"

namespace mx{

namespace detail {

// TODO: Make this microkernel more "microkernel-like" by passing pointers
// already offset to the current A/B/C micro-panels and using only local
// indices inside the hot loop. That would remove global index arithmetic
// from the kernel and give the compiler a simpler inner loop to optimize.
template<typename T, index_t NR, index_t MR, bool RowMajorC = true>
void microkernel_outer_product(const index_t k0_tile, const index_t k_end_tile,
                               const index_t i0_tile, const index_t i_micro_end,
                               const index_t j0_tile, const index_t j_micro_end,
                               const index_t nr_actual, const index_t mr_actual,
                               const T alpha,
                               const T* __restrict__ A_panel,  const index_t stride_a,
                               const T* __restrict__ BT_panel, const index_t stride_bT, // notation here is for RowMajor
                               T* __restrict__ C_panel,        const index_t stride_c)
{
    // Register block for micro-tile of C (nr x mr) + Initialize to zero
    T sum[NR * MR]{};

    // Outer-product accumulation: k outermost, broadcast a_val across mr columns
    // Layout-agnostic: both RowMajor(A) and ColMajor(AT) store k as stride-1
    for (index_t k = k0_tile; k < k_end_tile; ++k) {
        T b_reg[MR];

        // Note: the BT fetch across j is strided at fixed k, but each access lands
        // in a BT row that is contiguous in k. Advancing k then reuses those rows as
        // sequential streams, so transposing B can still be beneficial despite the
        // strided MR-wide load.
        for(index_t jl = 0; jl < mr_actual; ++jl){
            b_reg[jl] = BT_panel[k + stride_bT * (j0_tile + jl)];
        }

        for (index_t il = 0; il < nr_actual; ++il) { // `l` for local micro-tile indices
            const T a_val = A_panel[k + stride_a * (i0_tile + il)]; // load once, reuse mr times
            for (index_t jl = 0; jl < mr_actual; ++jl) {
                sum[jl + MR * il] += a_val * b_reg[jl];
            }
        }
    }

    // Write-back: loop order matches C's memory layout for contiguous stores
    if constexpr (RowMajorC) {
        // RowMajor C: j is stride-1, so j-inner gives contiguous writes
        for (index_t ig = i0_tile; ig < i_micro_end; ++ig) { // `g` for global tile indices
            for (index_t jg = j0_tile; jg < j_micro_end; ++jg) {
                // C(ig_micro,jg_micro) += alpha * sum[(jg_micro-j0_tile) + mr * (ig_micro-i)];
                C_panel[jg + stride_c * ig] += alpha * sum[(jg - j0_tile) + MR * (ig - i0_tile)];
            }
        }
    } else {
        // ColMajor C: i is stride-1, so i-inner gives contiguous writes
        for (index_t jg = j0_tile; jg < j_micro_end; ++jg) {
            for (index_t ig = i0_tile; ig < i_micro_end; ++ig) {
                C_panel[ig + stride_c * jg] += alpha * sum[(jg - j0_tile) + MR * (ig - i0_tile)];
            }
        }
    }
}

} // namespace detail

template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_microkernel(const T alpha, 
                    const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    const T beta, Dense<T, Layout>& C,
                    index_t nthreads = 1,
                    Scheduler sched = Scheduler{})
{
    gemm_microkernel(alpha, A.view(), B.view(), beta, C.view(), nthreads, sched);
}


template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_microkernel(const T alpha, 
                      DenseView<const T, Layout> A,
                      DenseView<const T, Layout> B,
                      const T beta,
                      DenseView<T,       Layout> C,
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

    // Tiling parameters
    const index_t nc = 256; // N tile
    const index_t kc = 128; // K tile
    const index_t mc = 160; // M tile

    // Number of tiles along each dimension (for parallelization)
    const index_t Nb = (N + nc - 1) / nc;

    if (nthreads < 1) nthreads = 1;
    if (nthreads > Nb) nthreads = (Nb > 0 ? Nb : 1);

    // Microkernel parameters (should be tuned to fit in registers)
    constexpr index_t nr = 6; // Number of rows in A micro-panel
    constexpr index_t mr = 8; // Number of cols in B micro-panel

    using AView = DenseView<const T, Layout>;

    if constexpr (AView::is_row_major) {

        Dense<T, Layout> BT(M, K);
        transpose_matrix_tiled(B, BT.view()/* , TILE_L1=32*/);

        auto mm_microkernel = [&](index_t tid){
            // Parallelize over row tiles of C.
            // Outer tile order: I-K-J. Inner order: i-j-k.
            sched(tid, nthreads, Nb,
                [&](index_t Ni){
                    const index_t i0 = Ni*nc;
                    const index_t i_end = std::min<index_t>(N, i0 + nc);

                    for (index_t k0 = 0; k0 < K; k0 += kc) {
                        const index_t k_end = std::min<index_t>(K, k0 + kc);

                        for (index_t j0 = 0; j0 < M; j0 += mc) {
                            const index_t j_end = std::min<index_t>(M, j0 + mc);

                            // Tile-tile multiply
                            // Divide each tile:
                            // - Tile A into micro-panels of size [nr;kc] 
                            // - Tile B into micro-panels of size [kc;mr] 
                            
                            for (index_t i = i0; i < i_end; i+=nr) { // Rows of A tile
                                index_t i_micro_end = std::min(i+nr, i_end);
                                
                                for (index_t j = j0; j < j_end; j+=mr) { // Cols of B tile
                                    index_t j_micro_end = std::min(j+mr, j_end);
                                    
                                    index_t nr_actual = std::min(nr, i_micro_end - i); // number of valid rows in the A micro-panel
                                    index_t mr_actual = std::min(mr, j_micro_end - j); // number of valid cols in the B micro-panel
                                    
                                    detail::microkernel_outer_product<T, nr, mr, true>(k0, k_end, i, i_micro_end, j, j_micro_end, nr_actual, mr_actual,
                                                                            alpha, A.data(),  /*stride_a =*/ A.row_stride(),
                                                                                    BT.data(), /*stride_bT =*/ BT.row_stride(),
                                                                                    C.data(),  /*stride_c =*/ C.row_stride());
                                
                                }
                            }
                        }
                    }
                }
            );
        };

        if (nthreads == 1) mm_microkernel(0);
        else launch_threads(mm_microkernel, nthreads);

    } else {// Col-major

        Dense<T, Layout> AT(K, N);
        transpose_matrix_tiled(A, AT.view()/* , TILE_L1=32*/);

        auto mm_microkernel = [&](index_t tid){
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

                            // Tile-tile multiply                    
                            for (index_t j = j0; j < j_end; j+=mr) { // Cols of B tile
                                index_t j_micro_end = std::min(j+mr, j_end);

                                for (index_t i = i0; i < i_end; i+=nr) { // Rows of A tile
                                    index_t i_micro_end = std::min(i+nr, i_end);
                                    
                                    index_t nr_actual = std::min(nr, i_micro_end - i);
                                    index_t mr_actual = std::min(mr, j_micro_end - j);

                                    detail::microkernel_outer_product<T, nr, mr, false>(k0, k_end, i, i_micro_end, j, j_micro_end, nr_actual, mr_actual,
                                                                                alpha, AT.data(), /*stride_a =*/ AT.col_stride(),
                                                                                    B.data(),  /*stride_b =*/ B.col_stride(),
                                                                                    C.data(),  /*stride_c =*/ C.col_stride());
                                
                                }
                            }
                        }
                    }
                }
            );
        };

        if (nthreads == 1) mm_microkernel(0);
        else launch_threads(mm_microkernel, nthreads);
    }
}


} // namespace mx
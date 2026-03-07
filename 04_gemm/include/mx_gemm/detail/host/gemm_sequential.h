#pragma once
#include<cassert>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"
#include"mx/layout.h"
#include"mx/transpose.h"
#include"mx/scale_matrix.h"

namespace mx{
    
template<typename T, class Layout = RowMajor>
void gemm_naive(const T alpha,
                    const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    const T beta,
                    Dense<T, Layout>& C)
{
    gemm_naive(alpha, A.view(), B.view(), beta, C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_transposed (const T alpha,
                    const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    const T beta,
                    Dense<T, Layout>& C)
{
    gemm_transposed (alpha, A.view(), B.view(), beta, C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_cache_blocked(const T alpha,
                        const Dense<T, Layout>& A,
                        const Dense<T, Layout>& B,
                        const T beta,
                        Dense<T, Layout>& C)
{
    gemm_cache_blocked(alpha, A.view(), B.view(), beta, C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_microkernel(const T alpha, 
                      const Dense<T, Layout>& A,
                      const Dense<T, Layout>& B,
                      const T beta,
                      Dense<T, Layout>& C)
{
    gemm_microkernel(alpha, A.view(), B.view(), beta, C.view());
}


template<typename T, class Layout = RowMajor>
void gemm_naive(const T alpha,
                    DenseView<const T, Layout> A,
                    DenseView<const T, Layout> B,
                    const T beta,
                    DenseView<T,       Layout> C)
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());

    for(index_t i=0; i<N; i++){
        for(index_t j=0; j<M; j++){
            T sum{};
            for(index_t k=0; k<K; k++){
                sum += A(i,k)*B(k,j); 
            }
            C(i,j) = alpha * sum + beta * C(i,j);
        }
    }
}


template<typename T, class Layout = RowMajor>
void gemm_transposed (const T alpha, 
                    DenseView<const T, Layout> A,
                    DenseView<const T, Layout> B,
                    const T beta,
                    DenseView<T,       Layout> C)
{
    
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();
    
    assert(K == B.rows() && N == C.rows() && M == C.cols());
    
    constexpr index_t TILE_L1 = 32; // tile size for L1 cache (assuming 32KB L1 cache and 8B double precision)
    
    using AView = DenseView<const T, Layout>;

    if constexpr (AView::is_row_major) {
        // Scale C
        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }
        scale_matrix(beta, C);

        // Materialize the B_transposed for better locality for RowMajor operations
        Dense<T, RowMajor> BT(M,K);
        transpose_matrix_tiled(B, BT.view(), TILE_L1);
        
        for(index_t i=0; i<N; i++){
            for(index_t j=0; j<M; j++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k);
                }
                C(i,j) += alpha * sum;
            }
        }
    } else{

        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }
        scale_matrix(beta, C);

        // Materialize the A_transposed for better locality for ColMajor operations
        Dense<T, ColMajor> AT(K, N);
        transpose_matrix_tiled(A, AT.view(), TILE_L1);

        // First loop on columns offers better locality for I/O on C matrix in with ColMajor layout
        for(index_t j=0; j<M; j++){
            for(index_t i=0; i<N; i++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += AT(k,i)*B(k,j);
                }
                C(i,j) += alpha * sum;
            }
        } 
    }
}

template<typename T, class Layout = RowMajor>
void gemm_cache_blocked(const T alpha, 
                        DenseView<const T, Layout> A,
                        DenseView<const T, Layout> B,
                        const T beta,
                        DenseView<T,       Layout> C)
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    using AView = DenseView<const T, Layout>;

    // *** Tiling parameters ***
    // -> Highly dependent on the cache sizes of the target architecture, 
    // especially L1 and L2 cache for a 1-layer tiling 
    // Idea: Keep A-tile, BT-tile and C-tile reasonably L2-friendly
    // i.e. sizeof(A_tile) + sizeof(BT_tile) + sizeof(C_tile) < L2_cache_size (~ 60% - 70% of total is comfortable)
    // Computation for the 1MB L2 cache for my i9-7940X CPU (Skylake-X):
    const index_t nc = 256; // N tile; rows of A/C per tile 
    const index_t kc = 128; // K tile; depth of A/B per tile
    const index_t mc = 160; // M tile; columns of B/C per tile

    if constexpr (AView::is_row_major) {// Row-major: transpose B to get contiguous access in k

        // Quick outs
        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }

        // Scale C once (avoid doing it inside the K-loop)
        scale_matrix(beta, C);

        // Transpose B into BT for better locality
        Dense<T, Layout> BT(M, K);
        transpose_matrix_tiled(B, BT.view()/* , TILE_L1=32*/);

        // IKJ tile ordering, then {i,j,k} multiply order within tiles
        for (index_t i0 = 0; i0 < N; i0 += nc) {
            const index_t i_end = std::min<index_t>(N, i0 + nc);

            for (index_t k0 = 0; k0 < K; k0 += kc) {
                const index_t k_end = std::min<index_t>(K, k0 + kc);

                for (index_t j0 = 0; j0 < M; j0 += mc) {
                    const index_t j_end = std::min<index_t>(M, j0 + mc);

                    // Tile-tile multiply with inner order i, j, k
                    // The access pattern is optimized for RowMajor reads/writes on C 
                    // and pretty good for L2-resident tiles of A
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

    } else {// Col-major: transpose A instead, keep B, walk k down columns

        // Quick outs
        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }

        scale_matrix(beta, C);

        // Transpose A into AT so that the k-dimension is contiguous in ColMajor
        Dense<T, Layout> AT(K, N);
        transpose_matrix_tiled(A, AT.view()/* , TILE_L1=32*/);

        // IKJ tile ordering, then {j,i,k} order for innermost loops
        for (index_t i0 = 0; i0 < N; i0 += nc) {
            const index_t i_end = std::min<index_t>(N, i0 + nc);

            for (index_t k0 = 0; k0 < K; k0 += kc) {
                const index_t k_end = std::min<index_t>(K, k0 + kc);

                for (index_t j0 = 0; j0 < M; j0 += mc) {
                    const index_t j_end = std::min<index_t>(M, j0 + mc);

                    // Tile-tile multiply with inner order j, i, k
                    // The access pattern is optimized for ColMajor reads/writes on C
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
    }
}

namespace detail {

template<typename T, index_t NR, index_t MR, bool RowMajorC = true>
void microkernel_outer_product(const index_t k0_tile, const index_t k_end_tile,
                               const index_t i0_tile, const index_t i_micro_end,
                               const index_t j0_tile, const index_t j_micro_end,
                               const index_t nr_actual, const index_t mr_actual,
                               const T alpha,
                               const T* __restrict__ A_panel,  const index_t stride_a,
                               const T* __restrict__ BT_panel, const index_t stride_b,
                               T* __restrict__ C_panel,        const index_t stride_c)
{
    // Register block for micro-tile of C (nr x mr) + Initialize to zero
    T sum[NR * MR]{};

    // Outer-product accumulation: k outermost, broadcast a_val across mr columns
    // Layout-agnostic: both RowMajor(A) and ColMajor(AT) store k as stride-1
    for (index_t k = k0_tile; k < k_end_tile; ++k) {
        for (index_t il = 0; il < nr_actual; ++il) { // `l` for local micro-tile indices
            // const T a_val = A(i0_tile + il, k);  // load once, reuse mr times
            const T a_val = A_panel[k + stride_a * (i0_tile + il)];
            for (index_t jl = 0; jl < mr_actual; ++jl) {
                // sum[jl_micro + mr * il_micro] += a_val * B(k, j0_tile + jl); // BT for RowMajor is useful for temporal locality
                sum[jl + MR * il] += a_val * BT_panel[k + stride_b * (j0_tile + jl)];
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

}

template<typename T, class Layout = RowMajor>
void gemm_microkernel(const T alpha, 
                      DenseView<const T, Layout> A,
                      DenseView<const T, Layout> B,
                      const T beta,
                      DenseView<T,       Layout> C)
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    using AView = DenseView<const T, Layout>;

    // *** Tiling parameters ***
    const index_t nc = 256; // N tile
    const index_t kc = 128; // K tile
    const index_t mc = 160; // M tile

    constexpr index_t nr = 6; // Number of rows in A micro-panel
    constexpr index_t mr = 8; // Number of cols in B micro-panel

    if constexpr (AView::is_row_major) {

        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }

        scale_matrix(beta, C);

        // NOTE: Transpose B into BT for better TEMPORAL locality in the microkernel for RowMajor 
        // -> B is actually better in spatial locality here but temporal wins makes the bigger perf gains !!
        Dense<T, Layout> BT(M, K);
        transpose_matrix_tiled(B, BT.view()/* , TILE_L1=32*/);

        // IKJ tile ordering
        for (index_t i0 = 0; i0 < N; i0 += nc) {
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
                                                                              BT.data(), /*stride_b =*/ BT.row_stride(),
                                                                              C.data(),  /*stride_c =*/ C.row_stride());
                        
                        }
                    }
                }
            }
        }

    } else {// Col-major

        if (alpha == T{0}) {
            scale_matrix(beta, C);
            return;
        }

        scale_matrix(beta, C);

        Dense<T, Layout> AT(K, N);
        transpose_matrix_tiled(A, AT.view()/* , TILE_L1=32*/);

        // IKJ tile ordering, then {j,i,k} order for innermost loops
        for (index_t i0 = 0; i0 < N; i0 += nc) {
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
    }
}


}
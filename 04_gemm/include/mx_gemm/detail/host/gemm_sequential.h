#pragma once
#include<cassert>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"
#include"mx/layout.h"
#include"mx/transpose.h"

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

        // Materialize the B_transposed for better locality for RowMajor operations
        Dense<T, RowMajor> BT(M,K);
        transpose_matrix_tiled(B, BT.view(), TILE_L1);
        
        for(index_t i=0; i<N; i++){
            for(index_t j=0; j<M; j++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k);
                }
                C(i,j) = alpha * sum + beta * C(i,j);
            }
        }
    } else{

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
                C(i,j) = alpha * sum + beta * C(i,j);
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
            if (beta == T{0}) {
                for (index_t i = 0; i < N; ++i)
                    for (index_t j = 0; j < M; ++j)
                        C(i,j) = T{0};
            } else if (beta != T{1}) {
                for (index_t i = 0; i < N; ++i)
                    for (index_t j = 0; j < M; ++j)
                        C(i,j) *= beta;
            }
            return;
        }

        // Scale C once (avoid doing it inside the K-loop)
        if (beta == T{0}) {
            for (index_t i = 0; i < N; ++i)
                for (index_t j = 0; j < M; ++j)
                    C(i,j) = T{0};
        } else if (beta != T{1}) {
            for (index_t i = 0; i < N; ++i)
                for (index_t j = 0; j < M; ++j)
                    C(i,j) *= beta;
        }

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

        // // Alternative: JKI tile ordering, then {i,j,k} multiply order within tiles
        // for (index_t j0 = 0; j0 < M; j0 += mc) {
        //     const index_t j_end = std::min<index_t>(M, j0 + mc);
            
        //     for (index_t k0 = 0; k0 < K; k0 += kc) {
        //         const index_t k_end = std::min<index_t>(K, k0 + kc);
                
        //         for (index_t i0 = 0; i0 < N; i0 += nc) {
        //             const index_t i_end = std::min<index_t>(N, i0 + nc);

        //             // Keep {i,j,k} order for tile-tile multiply (C-friedly) 
        //             for (index_t i = i0; i < i_end; ++i) {
        //                     for (index_t j = j0; j < j_end; ++j) {
        //                     T sum = T{0};

        //                     for (index_t k = k0; k < k_end; ++k) {
        //                         sum += A(i,k) * BT(j,k);
        //                     }
        //                     C(i,j) += alpha * sum;
        //                 }
        //             }
        //         }
        //     }
        // }

    } else {// Col-major: transpose A instead, keep B, walk k down columns

        // Quick outs
        if (alpha == T{0}) {
            if (beta == T{0}) {
                for (index_t j = 0; j < M; ++j)
                    for (index_t i = 0; i < N; ++i)
                        C(i,j) = T{0};
            } else if (beta != T{1}) {
                for (index_t j = 0; j < M; ++j)
                    for (index_t i = 0; i < N; ++i)
                        C(i,j) *= beta;
            }
            return;
        }

        // Scale C once (avoid doing it inside the K-loop)
        if (beta == T{0}) {
            for (index_t j = 0; j < M; ++j)
                for (index_t i = 0; i < N; ++i)
                    C(i,j) = T{0};
        } else if (beta != T{1}) {
            for (index_t j = 0; j < M; ++j)
                for (index_t i = 0; i < N; ++i)
                    C(i,j) *= beta;
        }

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

}
#pragma once
#include<cassert>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"
#include"mx/layout.h"

namespace mx{

template<typename T, class Layout = RowMajor>
void gemm_cache_blocked(const Dense<T, Layout>& A,
                        const Dense<T, Layout>& B,
                        Dense<T, Layout>& C)
{
    gemm_cache_blocked(A.view(), B.view(), C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_optimized(const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    Dense<T,       Layout>& C)
{
    gemm_optimized(A.view(), B.view(), C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_reference(const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    Dense<T, Layout>& C)
{
    gemm_reference(A.view(), B.view(), C.view());
}



template<typename T, class Layout = RowMajor>
void gemm_reference(DenseView<const T, Layout> A,
                    DenseView<const T, Layout> B,
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
            C(i,j) = sum;
        }
    }
}



template<typename T, class Layout = RowMajor>
void gemm_optimized(DenseView<const T, Layout> A,
                    DenseView<const T, Layout> B,
                    DenseView<T,       Layout> C)
{
    constexpr bool row_major = std::is_same_v<Layout, RowMajor>;

    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());

    if constexpr (row_major){
        // Materilize the B_transposed for better locality for RowMajor operations
        Dense<T,Layout> BT(M, K);
        for(index_t c=0; c<M; c++){ // Col loop first for sequential writes to BT
            for(index_t r=0; r<K; r++){
                BT(c,r) = B(r,c); 
            }
        }
        
        for(index_t i=0; i<N; i++){
            for(index_t j=0; j<M; j++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += A(i,k)*BT(j,k);
                }
                C(i,j) = sum;
            }
        }
    } else{
        // Materilize the A_transposed for better locality for ColMajor operations
        Dense<T,Layout> AT(K, N);
        for(index_t r=0; r<N; r++){ // Row loop first for sequential writes to AT
            for(index_t c=0; c<K; c++){
                AT(c,r) = A(r,c); 
            }
        }
        // First loop on columns offers better locality for I/O on C matrix in with ColMajor layout
        for(index_t j=0; j<M; j++){
            for(index_t i=0; i<N; i++){
                T sum{};
                for(index_t k=0; k<K; k++){
                    sum += AT(k,i)*B(k,j);
                }
                C(i,j) = sum;
            }
        } 
    }
}


template<typename T, class Layout = RowMajor>
void gemm_cache_blocked(DenseView<const T, Layout> A,
                        DenseView<const T, Layout> B,
                        DenseView<T,       Layout> C)
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    using AView = DenseView<const T, Layout>;
    constexpr bool row_major = AView::is_row_major;

    // Tiling parameters (same for both layouts)
    const index_t nc = 256; // rows of A/C per block
    const index_t kc = 256; // depth of A/B per block
    const index_t mc = 96; // columns of B/C per block

    const index_t Nb = (N + nc - 1) / nc; // number of row blocks 
    const index_t Kb = (K + kc - 1) / kc; // number of depth blocks
    const index_t Mb = (M + mc - 1) / mc; // number of column blocks

    if constexpr (row_major) {// Row-major: transpose B to get contiguous access in k
        Dense<T, Layout> BT(M, K);
        for (index_t r = 0; r < K; ++r) {
            for (index_t c = 0; c < M; ++c) {
                BT(c, r) = B(r, c); // BT(j,k) = B(k,j)
            }
        }

        for (index_t Mi = 0; Mi < Mb; ++Mi) {
            const index_t jc   = Mi * mc;
            const index_t jend = std::min(jc + mc, M);

            for (index_t Ki = 0; Ki < Kb; ++Ki) {
                const index_t pc   = Ki * kc;
                const index_t pend = std::min(pc + kc, K);

                for (index_t Ni = 0; Ni < Nb; ++Ni) {
                    const index_t ic   = Ni * nc;
                    const index_t iend = std::min(ic + nc, N);

                    // C_block(Ni,Mi) += A_block(Ni,Ki) * B_block(Ki,Mi) 
                    for (index_t i = ic; i < iend; ++i) {
                        for (index_t j = jc; j < jend; ++j) {
                            T sum{}; // register accumulator
                            for (index_t p = pc; p < pend; ++p) {
                                // A(i,p) and BT(j,p) both contiguous in p for RowMajor
                                sum += A(i,p) * BT(j,p);
                            }
                            C(i,j) += sum;
                        }
                    }
                }
            }
        }

    } else {// Col-major: transpose A instead, keep B, walk k down columns
        Dense<T, Layout> AT(K, N);
        for (index_t r = 0; r < N; ++r) {
            for (index_t c = 0; c < K; ++c) {
                AT(c, r) = A(r, c); // AT(p,i) = A(i,p)
            }
        }

        for (index_t Mi = 0; Mi < Mb; ++Mi) {
            const index_t jc   = Mi * mc;
            const index_t jend = std::min(jc + mc, M);

            for (index_t Ki = 0; Ki < Kb; ++Ki) {
                const index_t pc   = Ki * kc;
                const index_t pend = std::min(pc + kc, K);

                for (index_t Ni = 0; Ni < Nb; ++Ni) {
                    const index_t ic   = Ni * nc;
                    const index_t iend = std::min(ic + nc, N);

                    for (index_t i = ic; i < iend; ++i) {
                        for (index_t j = jc; j < jend; ++j) {
                            T sum{};
                            for (index_t p = pc; p < pend; ++p) {
                                // For ColMajor, both AT(p,i) and B(p,j) are contiguous in p
                                sum += AT(p, i) * B(p, j);
                            }
                            C(i,j) += sum;
                        }
                    }
                }
            }
        }
    }
}

}
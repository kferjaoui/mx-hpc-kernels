#pragma once
#include<cassert>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"
#include"mx/layout.h"

namespace mx{

template<typename T, class Layout = RowMajor>
void gemm(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C){
    gemm(A.view(), B.view(), C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_naive(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C){
    gemm_naive(A.view(), B.view(), C.view());
}

template<typename T, class Layout = RowMajor>
void gemm_naive(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C) {
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
void gemm(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C) {

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

}
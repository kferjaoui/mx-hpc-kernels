#pragma once
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/types.h"
#include "pivot.h"

namespace mx{

// Forward solve: L is unit-lower, stored in strict lower part of LU
// Find Y for L * Y = B (Y stored in-place in B)
template<typename T, class Layout>
inline void forward_substitution_unit_lower(const DenseView<T, Layout> LU, const std::vector<index_t>& piv, DenseView<T, Layout> B){
    const index_t N = LU.rows(); // In general, L is of size N x N; and Y is of size (N,1)

    const index_t Mrhs = B.cols(); // if Mrhs=1: solving 1 system | if Mrhs>1: solving multiple linear systems at once

    for(index_t i=0; i<N; i++){
        for(index_t j=0; j<i; j++){
            const T l_ij =  LU(i,j);
            if (l_ij == T(0)) continue; // no need to actually compute
            for(index_t col=0; col<Mrhs; col++){
                B(i,col) = B(i, col) - l_ij *B(j,col);
            }
        }       
    }

}

// Backward solve: U is upper triangular, stored in upper part of LU (inclusing diagonal)
// // Find X for U * X = B (X stored in-place in B) 
template<typename T, class Layout>
inline LUStatus backward_substitution_upper(const DenseView<T, Layout> LU, DenseView<T, Layout> B){
    const index_t N = LU.rows(); 
    const index_t M = LU.cols(); // In general, U is of size N x M; and X is of size (N,1)
    const index_t K = std::min(N,M);

    const index_t Mrhs = B.cols(); 
    if(N != B.rows()) return LUStatus::BAD_ARG; // incompatible shapes

    for(index_t i=K-1; i>=0; i--){
        const T u_ii =  LU(i,i);
        if (u_ii == T(0)) return LUStatus::SINGULAR; // no unique solution
        
        for(index_t col=0; col<Mrhs; col++){ //one column at a time

            T b_i = B(i, col);
            for(index_t j=i+1; j<K; j++){
                const T u_ij =  LU(i,j);
                if (u_ij == T(0)) continue;
                b_i = b_i - u_ij * B(j,col);
            }
            B(i,col) = b_i/u_ii;
        }       
    }

    return LUStatus::SUCCESS;
  
}

template<typename T, class Layout>
[[nodiscard]] LUInfo lu_solve(const DenseView<T, Layout> LU, const std::vector<index_t>& piv, DenseView<T, Layout> B){

    const index_t N = LU.rows();
    const index_t M = LU.cols(); 
    const index_t K = std::min(N, M);
    
    const index_t Nrhs = B.rows();
    
    LUInfo info_solve;
    
    if (N != Nrhs) {
        info_solve.status = LUStatus::BAD_ARG;
        return info_solve;
    }

    // 1. apply in-place permutation on the rhs i.e. B = P * B (P compounding the swaps used during the factorization)
    apply_pivots_to_rhs(B, piv, K);

    // 2. Solving L * Y = B' where B' = P * B
    forward_substitution_unit_lower(LU, piv, B);
    
    // 3. Solving U * X = Y
    info_solve.status = backward_substitution_upper(LU, B);
    if (info_solve.status == LUStatus::SINGULAR) {
        info_solve.first_zero_pivot = 0; // TODO: find first zero pivot during backward substitution
    }

    return info_solve;

}


}  
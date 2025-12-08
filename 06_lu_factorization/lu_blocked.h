#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "mx/types.h"
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "pivot.h"
#include "factors.h"
#include "lu_unblocked.h"
#include "gemm.h"


namespace mx{

template<typename T, class Layout>
void invert_lower_triangular(const Dense<T, Layout> L,
                             Dense<T, Layout>& L_inv)
{
    invert_lower_triangular(L.view(), L_inv);
}

template<typename T, class Layout>
void invert_lower_triangular(DenseView<const T, Layout> L,
                             Dense<T, Layout>& L_inv)
{
    assert(L.rows() == L.cols() && "Matrix must be square");
    assert(L_inv.rows() == L.rows() && L_inv.cols() == L.cols() && "Output matrix must have the same shape as input");
    
    const index_t N = L.rows();
    
    for (index_t j = 0; j < N; j++) {
        // Diagonal elements are by definition the inverse of the original diagonal elements
        if (std::abs(L(j,j)) < 1e-14) throw std::runtime_error("Matrix is singular");
        L_inv(j,j) = 1.0 / L(j,j);
        
        for (index_t i = j+1; i < N; i++) {
            T sum{0};
            for (index_t k = j; k < i; k++) {
                sum += L(i,k) * L_inv(k,j);
            }
            L_inv(i,j) = -sum / L(i,i);
        }
    }
}

template<typename T, class Layout>
[[nodiscard]] 
LUInfo lu_blocked(Dense<T, Layout>& LU,
                  std::vector<index_t>& piv)
{
    return lu_blocked(LU.view(), piv);
}

template<typename T, class Layout>
[[nodiscard]] 
LUInfo lu_blocked(DenseView<T, Layout> LU,
                  std::vector<index_t>& piv)
{
    const index_t N = LU.rows();
    const index_t M = LU.cols(); 
    const index_t K = std::min(N, M);

    if (piv.size() < K){
        std::cout << "Warning: pivot vector resized from " << piv.size() << " to " << K << "\n";
        piv.resize(K);
    }
    return lu_blocked_impl(LU, std::span<index_t>(piv.data(), K));
}

template<typename T, class Layout>
[[nodiscard]] constexpr 
LUInfo lu_blocked_impl(DenseView<T, Layout> LU,
                       std::span<index_t> piv)
{
    const index_t N = LU.rows();
    const index_t M = LU.cols(); 
    const index_t K = std::min(N, M);

    assert(piv.size() >= K && "pivot vector too small");

    LUInfo info;

    // Block size (tunable parameter)
    const index_t blockSize = 128; // this should be tuned for the target architecture
    
    for(index_t s=0; s<K; s+=blockSize){
        // printf("Processing block #%d \n", int(s/blockSize));

        const index_t current_block_size = std::min(blockSize, K - s);
        
        DenseView<T, Layout> LU_panel_k = LU.subview(s, s, N - s, current_block_size);
        
        // 1. Factorize the current panel i.e. compute L00, U00 and L10
        std::span<index_t> piv_k(&piv[s], current_block_size);
        LUInfo info_k = lu_unblocked_impl(LU_panel_k, piv_k);

        // convert local pivots to global indices
        for (index_t p = 0; p < current_block_size; ++p) {
            piv[s + p] += s;               // local -> global
        }

        // Propagate info
        if (!info_k.ok() && info.ok()) {
            info = info_k; // propagate first encountered singularity
            info.first_zero_pivot += s; // adjust index relative to full matrix
        }

        // 2. Extract L00 (top-left block of panel)
        DenseView<T, Layout> LU00 = LU_panel_k.subview(0,0, 
                                                       current_block_size, current_block_size); //use local indices

        std::vector<T> buf_L00(current_block_size*current_block_size, T{0}); 
        DenseView<T, Layout> L00(buf_L00.data(), current_block_size, current_block_size);
        extract_unit_lower(as_const(LU00), L00);

        // 3. Apply the pivots to the rest of the matrix i.e. to the laft AND right of the panel
        for (index_t k = 0; k < current_block_size; ++k) {
            index_t r0 = s + k;          // global
            index_t r1 = piv[s + k];     // global
            if (r0 == r1) continue;

            // Apply to LEFT part [0 .. s)
            if (s > 0) swap_row_prefix(LU, r0, r1, s);

            // Skip the panel [s .. s+ib): already swapped inside the panel subview

            // Apply to RIGHT part [s+ib .. M)
            const index_t start_col = s + current_block_size;
            if (start_col < M) swap_row_tails(LU, r0, r1, start_col);
        }

        // 4. Inverse L00
        Dense<T, Layout> L00_inv(L00.rows(), L00.cols(), T{0}); // initialize to zero
        invert_lower_triangular(as_const(L00), L00_inv);

        // // Debug: check L00 * L00_inv = I
        // Dense<T, Layout> Identity_block(current_block_size, current_block_size, T{0});
        // gemm_cache_blocked(as_const(L00), as_const(L00_inv.view()), Identity_block.view());
        // std::cout << "L00 * L00_inv (should be identity):\n" << Identity_block << "\n";

        // 5. Update U01 (= LU01) = L00^{-1} * A01
        const index_t n_cols_trailing = M - (s + current_block_size);

        if (n_cols_trailing>0) {
            DenseView<T, Layout> LU01 = LU.subview(s, s + current_block_size,
                                                current_block_size, n_cols_trailing); // view into original matrix
            
            Dense<T, Layout> A01(LU01); // copy to a seperate buffer before overwriting
            gemm_cpu_threads_vectorized(T{1},
                                        as_const(L00_inv.view()), 
                                        as_const(A01.view()), 
                                        T{0},
                                        LU01,
                                        std::thread::hardware_concurrency());
        }

        // 6. Update trailing matrix i.e. A11 = A11 - L10 * U01
        index_t n_rows_trailing = N - (s + current_block_size);
        if (n_cols_trailing >0 && n_rows_trailing>0) {
             // View into original matrix (global indices)
            DenseView<T, Layout> L10 = LU.subview(s + current_block_size, s,
                                                n_rows_trailing, current_block_size);
            DenseView<T, Layout> U01 = LU.subview(s, s + current_block_size,
                                                current_block_size, n_cols_trailing);
            DenseView<T, Layout> A11 = LU.subview(s + current_block_size, s + current_block_size,
                                                n_rows_trailing, n_cols_trailing);
            // Perform A11 -= L10 * U01
            gemm_cpu_threads_vectorized(-T{1}, 
                                        as_const(L10), 
                                        as_const(U01),
                                        T{1}, 
                                        A11,
                                        std::thread::hardware_concurrency());
        }

    }
    return info;
}


}
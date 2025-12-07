#pragma once
#include <cstdint>
#include "mx/dense_view.h"

namespace mx{

template<typename T, class Layout>
void swap_full_rows(DenseView<T, Layout> A,
                    index_t row1,
                    index_t row2) noexcept 
{
    if (row1 == row2) return;
    const index_t cols = A.cols();
    for (index_t c = 0; c < cols; ++c) {
        std::swap(A(row1, c), A(row2, c));
    }
}

template<typename T, class Layout>
void swap_row_tails(DenseView<T, Layout> A, 
                    index_t row1, 
                    index_t row2, 
                    index_t start_col) noexcept 
{
    if (row1 == row2) return;
    const index_t cols = A.cols();
    for (index_t c = start_col; c < cols; ++c) {
        std::swap(A(row1, c), A(row2, c));
    }
}

template<typename T, class Layout>
void swap_row_prefix(DenseView<T, Layout> A,
                     index_t row1,
                     index_t row2,
                     index_t end_col) noexcept
{
    if (row1 == row2) return;
    for (index_t c = 0; c < end_col; ++c) {
        std::swap(A(row1, c), A(row2, c));
    }
}


// Permute the Right Hand Side (RHS): B = P * B using first Kp pivots (0-based)
template<typename T, class Layout>
void apply_pivots_to_rhs(DenseView<T, Layout> B,
                         const std::vector<index_t>& piv, 
                         index_t Kp)
{
    for(index_t k=0; k<Kp; k++) swap_full_rows(B, k, piv[k]);
}

}
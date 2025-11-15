#pragma once
#include <cstdint>
#include "mx/dense_view.h"

namespace mx{

template<typename T, class Layout>
void swap_full_row(DenseView<T, Layout> A, index_t i, index_t j) noexcept {
    if (i == j) return;
    const index_t cols = A.cols();
    for (index_t c = 0; c < cols; ++c) {
        std::swap(A(i, c), A(j, c));
    }
}

// Permute the Right Hand Side (RHS): B = P * B using first Kp pivots (0-based)
// TODO: Beware for benching against LAPACK which is 1-based
template<typename T, class Layout>
void apply_pivots_to_rhs(DenseView<T, Layout> B, const std::vector<index_t>& piv, index_t Kp){
    for(index_t k=0; k<Kp; k++) swap_full_row(B, k, piv[k]);
}


}
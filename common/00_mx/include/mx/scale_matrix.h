#pragma once
#include "dense.h"
#include "dense_view.h"
#include "layout.h"

namespace mx{

template<typename T, class Layout = RowMajor>
void scale_matrix(const T beta, Dense<T,Layout> C){
    scale_matrix(beta, C.view());

}
template<typename T, class Layout = RowMajor>
void scale_matrix(const T beta, DenseView<T,Layout> C){
    const index_t N = C.rows();
    const index_t M = C.rows();
    if constexpr (C.is_row_major){
        if (beta == T{0}) {
            for (index_t i = 0; i < N; ++i)
                for (index_t j = 0; j < M; ++j)
                    C(i,j) = T{0};
        } else if (beta != T{1}) {
            for (index_t i = 0; i < N; ++i)
                for (index_t j = 0; j < M; ++j)
                    C(i,j) *= beta;
        }
    } else {
        if (beta == T{0}) {
            for (index_t j = 0; j < M; ++j)
                for (index_t i = 0; i < N; ++i)
                    C(i,j) = T{0};
        } else if (beta != T{1}) {
            for (index_t j = 0; j < M; ++j)
                for (index_t i = 0; i < N; ++i)
                    C(i,j) *= beta;
        }
    }
}

}
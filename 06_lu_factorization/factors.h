#include <iostream>
#include <cassert>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"

namespace mx{

template<typename T, class Layout>
void extract_unit_lower(const Dense<T, Layout>& LU,
                        Dense<T,       Layout>& L)
{
    extract_unit_lower(LU.view(), L.view());
}

template<typename T, class Layout>
void extract_unit_lower(DenseView<const T, Layout> LU,
                        DenseView<T,       Layout> L)
{
    const index_t n = LU.rows();
    assert(L.rows() == n && L.cols() == n); // L is an invertible, square matrix
    for(index_t i=0; i<n; i++){
        for(index_t j=0; j<n; j++){
            if(i==j){
                L(i,j) = T(1);
                continue;
            }
            if (i>j) L(i,j) = LU(i,j);
            else L(i,j) = T(0);        
        }
    }
}


template<typename T, class Layout>
void extract_upper(const Dense<T, Layout>& LU,
                   Dense<T,       Layout>& U)
{
    extract_upper(LU.view(), U.view());
}

template<typename T, class Layout>
void extract_upper(DenseView<const T, Layout> LU,
                   DenseView<T,       Layout> U)
{
    const index_t n = LU.rows();
    const index_t m = LU.cols();
    assert(U.rows() == n && U.cols() == m); // U has the same shape as the original matrix
    for(index_t i=0; i<n; i++){
        for(index_t j=0; j<m; j++){
            if(i<=j) U(i,j) = LU(i,j);
            else U(i,j) = T(0);        
        }
    }
}

}
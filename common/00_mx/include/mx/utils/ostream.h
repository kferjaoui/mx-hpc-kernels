#pragma once
#include<ostream>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T, class Layout>
std::ostream& operator<<(std::ostream& os, const Dense<T, Layout>& Matrix){
    const index_t r = Matrix.rows(), c = Matrix.cols();
    os << "[\n";
    for(index_t i = 0; i < r; ++i){   
        os << " [";
        for(index_t j = 0; j < c; ++j){
            if (j) os << ", ";
            os << Matrix(i,j);
        }
        os << "]";
        if (i+1 < r) os << ",";
        os << "\n";
    }
    os << "]";
    return os;
}

template<typename T, class Layout>
std::ostream& operator<<(std::ostream& os, const DenseView<T, Layout>& MatrixView){
    const index_t r = MatrixView.rows(), c = MatrixView.cols();
    os << "[\n";
    for(index_t i = 0; i < r; ++i){   
        os << " [";
        for(index_t j = 0; j < c; ++j){
            if (j) os << ", ";
            os << MatrixView(i,j);
        }
        os << "]";
        if (i+1 < r) os << ",";
        os << "\n";
    }
    os << "]";
    return os;
}

}
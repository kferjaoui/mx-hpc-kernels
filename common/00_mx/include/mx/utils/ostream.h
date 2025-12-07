#pragma once
#include<ostream>
#include<vector>
#include<sstream>
#include<iomanip>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx {

namespace detail {

// Generic printer for matrices
template<typename MatrixLike>
std::ostream& print_matrix(std::ostream& os, const MatrixLike& A)
{
    const index_t rows = A.rows();
    const index_t cols = A.cols();

    // First pass: compute max width per column
    std::vector<std::size_t> col_widths(cols, 0);

    for (index_t j = 0; j < cols; ++j) {
        std::size_t w = 0;
        for (index_t i = 0; i < rows; ++i) {
            std::ostringstream oss;
            oss << A(i, j);
            w = std::max(w, oss.str().size());
        }
        col_widths[j] = w;
    }

    // Optional: can tweak formatting here;
    // auto old_flags = os.flags();
    // auto old_prec  = os.precision();
    // os << std::fixed << std::setprecision(4);

    os << "[\n";
    for (index_t i = 0; i < rows; ++i) {
        os << "  [";
        for (index_t j = 0; j < cols; ++j) {
            if (j) os << ", ";

            std::ostringstream oss;
            oss << A(i, j);
            const std::string s = oss.str();

            os << std::setw(static_cast<int>(col_widths[j])) << s;
        }
        os << "]";
        if (i + 1 < rows) os << ",";
        os << "\n";
    }
    os << "]";

    // os.flags(old_flags);
    // os.precision(old_prec);

    return os;
}

} // namespace detail


// Dense overload
template<typename T, class Layout>
std::ostream& operator<<(std::ostream& os, const Dense<T, Layout>& M)
{
    return detail::print_matrix(os, M);
}

// DenseView overload
template<typename T, class Layout>
std::ostream& operator<<(std::ostream& os, const DenseView<T, Layout>& V)
{
    return detail::print_matrix(os, V);
}

}
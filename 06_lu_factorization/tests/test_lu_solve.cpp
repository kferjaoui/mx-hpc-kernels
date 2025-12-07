#include <iostream>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "lu_unblocked.h"
#include "lu_solve.h"

int main(){
    // M is a rectangular matrix (i.e. more rows than columns)
    mx::Dense<double> M(6, 4, {1,2,3,4, 2,5,6,7, 3,1,4,1, 0,2,2,2, 5,0,1,3, 1,1,0,0});
    std::cout << "Original matrix M:\n" << M << std::endl;

    std::vector<mx::index_t> piv_M(3);
    mx::LUInfo info = mx::lu_unblocked(M.view(), piv_M);

    std::cout << "LU factorization info: " << info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv_M) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "The in-place LU after factorization:\n" << M << std::endl;

    // =============================================================
    // Solve Cx = b
    mx::Dense<double> C(3,3, {2,1,1, 4,-6,0, -2,7,2});
    std::cout << "Original matrix C:\n" << C << std::endl;
    mx::Dense<double> b(3,1, {5,-2,9});                 // single RHS
    std::vector<mx::index_t> piv_C(3);

    mx::LUInfo lu_info = mx::lu_unblocked(C.view(), piv_C);
    // Expect info.ok(), piv == [1,1,2]

    std::cout << "LU factorization info: " << lu_info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv_C) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "The in-place LU after factorization:\n" << C << std::endl;

    // Solve
    mx::LUInfo s_info = mx::lu_solve(C.view(), piv_C, b.view());
    // Expect b now equals [1,1,2]^T

    std::cout << "LU Solve info (single-RHS): " << s_info << std::endl;
    std::cout << "The solution vector x is:\n" << b << std::endl;

    // Multi-RHS variant
    mx::Dense<double> b2(3,2, {5,3, -2,2, 9,1});
    mx::LUInfo s_info2 = mx::lu_solve(C.view(), piv_C, b2.view());
    // Expect b2 columns to be [1,1,2]^T and [1.25, 0.5, 0]^T

    std::cout << "LU Solve info (multi-RHS): " << s_info2 << std::endl;
    std::cout << "The solution matrix X is:\n" << b2 << std::endl;

    return 0;
}
#include <iostream>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "lu_unblocked.h"
#include "lu_solve.h"
#include "factors.h"

#include <Eigen/Dense>

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXi = Eigen::VectorXi;

int main(){
    // A is a full-rank matrix
    // =================== MX ====================
    mx::Dense<double> A(4, 4, {5, 0, 1, 3, 2, 5, 4, 5, 3, 1, 3, 2, 1, 2, 3, 4});
    mx::Dense<double> LU_mx = A; // This will be used for in-place modifications -> keeps a copy of the original in A

    std::cout << "Original matrix A:\n" << LU_mx << std::endl;
    std::vector<mx::index_t> pivA_mx(2);
    mx::LUInfo info = mx::lu_unblocked(LU_mx, pivA_mx);

    std::cout << "LU factorization info: " << info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : pivA_mx) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "Combined LU:\n" << LU_mx << std::endl;

    const mx::index_t n = A.rows();
    const mx::index_t m = A.cols();
    mx::Dense<double> L_mx(n, n);
    mx::Dense<double> U_mx(n, m);

    extract_unit_lower(LU_mx, L_mx);
    extract_upper(LU_mx, U_mx);

    std::cout << "Extracted unit-lower L: " << L_mx << std::endl;
    std::cout << "Extracted upper U: " << U_mx << std::endl;

    // Convert to Eigen for comparison
    auto L_mx_eigen = L_mx.to_eigen();
    auto U_mx_eigen = U_mx.to_eigen();

    // =================== EIGEN ====================
    Matrix A_eigen = A.to_eigen();

    // LU partial pivoting with Eigen (row-major)
    Eigen::PartialPivLU<Matrix> lu_partial(A_eigen);

    // Get permutation as a 0-based pivot vector
    VectorXi piv_eigen = lu_partial.permutationP().indices(); // Eigen returns a PermutationMatrix object
    std::cout << "Eigen::Pivot vector (0-based):\n" << piv_eigen.transpose() << "\n";

    // Extract L and U
    Matrix LU_eigen = lu_partial.matrixLU(); // Combined L and U

    std::cout << "Eigen::Combined LU:\n" << LU_eigen << "\n";
    
    Matrix L_eigen = Matrix::Identity(A_eigen.rows(), A_eigen.cols());
    L_eigen.triangularView<Eigen::StrictlyLower>() = LU_eigen.triangularView<Eigen::StrictlyLower>();

    Matrix U_eigen = LU_eigen.triangularView<Eigen::Upper>();

    double L_diff = (L_mx_eigen - L_eigen).norm();
    double U_diff = (U_mx_eigen - U_eigen).norm();

    std::cout << "‖L_MX - L_Eigen‖_F = " << L_diff << "\n";
    std::cout << "‖U_MX - U_Eigen‖_F = " << U_diff << "\n";

    // Reconstruct PA and compare to LU
    Matrix PA_eigen = lu_partial.permutationP() * A_eigen;
    Matrix LUtogether = L_eigen * U_eigen;

    std::cout << "Eigen::Permuted matrix PA:\n" << PA_eigen << "\n";
    std::cout << "Eigen::Reconstructed L*U:\n" << LUtogether << "\n";

    // Norm of the mismatch
    double error_norm = (PA_eigen - LUtogether).norm();
    std::cout << "Eigen::‖PA - LU‖_F = " << error_norm << "\n";
}
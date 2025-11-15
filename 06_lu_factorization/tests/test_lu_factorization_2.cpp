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
    // B is a singular matrix i.e. rank deficient
    // N.B: 
    // In this example, I have tested with non-owning views (DenseView),
    // instead of buffer owning (Dense) types for validaiton purposes.
    // =================== MX ====================
    mx::Dense<double> B(3, 3, {1, 2, 3, 2, 4, 6, 1, 0, 1});
    mx::DenseView<double> LU_mx = B.view();

    std::cout << "Original matrix B:\n" << LU_mx << std::endl;
    std::vector<mx::index_t> pivB_mx(2);
    mx::LUInfo info = mx::lu_factor_unblocked(LU_mx, pivB_mx);

    std::cout << "mx::LU factorization info: " << info << std::endl;

    std::cout << "mx::Pivot vector (0-based): [ ";
    for(const auto& p : pivB_mx) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "mx::Combined LU:\n" << LU_mx << std::endl;

    const mx::index_t n = B.rows();
    const mx::index_t m = B.cols();
    mx::Dense<double> L_mx(n, n);
    mx::Dense<double> U_mx(n, m);

    extract_unit_lower(mx::DenseView<const double>(LU_mx), L_mx.view());
    extract_upper(mx::DenseView<const double>(LU_mx), U_mx.view());

    std::cout << "mx::Extracted unit-lower L: " << L_mx << std::endl;
    std::cout << "mx::Extracted upper U: " << U_mx << std::endl;

    // Convert to Eigen for comparison
    auto L_mx_eigen = L_mx.to_eigen();
    auto U_mx_eigen = U_mx.to_eigen();

    // =================== EIGEN ====================
    const auto B_view = B.view();
    Matrix B_eigen = B_view.to_eigen();
    std::cout << "Eigen::Original matrix B_eigen:\n" << B_eigen << std::endl;

    // LU partial pivoting with Eigen (row-major)
    Eigen::PartialPivLU<Matrix> lu_partial(B_eigen);

    // Get permutation as a 0-based pivot vector
    VectorXi piv_eigen = lu_partial.permutationP().indices(); // Eigen returns a PermutationMatrix object
    std::cout << "Eigen::Pivot vector (0-based):\n" << piv_eigen.transpose() << "\n";

    // Extract L and U
    Matrix LU_eigen = lu_partial.matrixLU(); // Combined L and U

    std::cout << "Eigen::Combined LU:\n" << LU_eigen << "\n";
    
    Matrix L_eigen = Matrix::Identity(B_eigen.rows(), B_eigen.cols());
    L_eigen.triangularView<Eigen::StrictlyLower>() = LU_eigen.triangularView<Eigen::StrictlyLower>();

    Matrix U_eigen = LU_eigen.triangularView<Eigen::Upper>();

    std::cout << "Eigen::Extracted unit-lower L:\n" << L_eigen << std::endl;
    std::cout << "Eigen::Extracted upper U:\n" << U_eigen << std::endl;

    double L_diff = (L_mx_eigen - L_eigen).norm();
    double U_diff = (U_mx_eigen - U_eigen).norm();

    std::cout << "‖L_MX - L_Eigen‖_F = " << L_diff << "\n";
    std::cout << "‖U_MX - U_Eigen‖_F = " << U_diff << "\n";

    // Reconstruct PB and compare to LU
    Matrix PB_eigen = lu_partial.permutationP() * B_eigen;
    Matrix LUtogether = L_eigen * U_eigen;

    std::cout << "Eigen::Permuted matrix PB:\n" << PB_eigen << "\n";
    std::cout << "Eigen::Reconstructed L*U:\n" << LUtogether << "\n";

    // Norm of the mismatch
    double error_norm = (PB_eigen - LUtogether).norm();
    std::cout << "Eigen::‖PB - LU‖_F = " << error_norm << "\n";
}
#include <iostream>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "lu_unblocked.h"
#include "lu_blocked.h"
#include "lu_solve.h"
#include "factors.h"
#include "mx_test/test_matrices.h"

#include "CycleTimer.h"

#include <Eigen/Dense>

using Matrix  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorI = Eigen::Matrix<mx::index_t, Eigen::Dynamic, 1>;

int main() {
    // ===================== Parameters =====================
    constexpr mx::index_t N = 2000; // matrix dimension
    constexpr mx::index_t K = 4;  // Gram matrix "rank"
    constexpr int Nattempts = 5;   // timing repetitions (take best)

    std::cout << "Testing LU factorization on SPD Gram matrix "
              << "N = " << N << ", K = " << K << "\n";

    // ===================== Generate test matrix =====================
    mx::Dense<double> G = mx::make_spd_gram_matrix<double>(
        N, K,
        /*alpha=*/1e-1,
        mx::GramPattern::TrigSmooth
    );

    // ===================== Eigen reference LU =====================
    Matrix G_eigen = G.to_eigen();

    double best_eigen_ms = std::numeric_limits<double>::max();

    // We'll reuse this object so allocations don't dominate the timing
    Eigen::PartialPivLU<Matrix> lu_ref;

    for (int attempt = 0; attempt < Nattempts; ++attempt) {
        Matrix G_eigen_local = G_eigen; // Eigen also overwrites its input

        double t0 = CycleTimer::currentSeconds();
        lu_ref.compute(G_eigen_local);
        double t1 = CycleTimer::currentSeconds();

        double elapsed_ms = (t1 - t0) * 1000.0;
        best_eigen_ms = std::min(best_eigen_ms, elapsed_ms);
    }

    std::cout << "[Eigen]   best of " << Nattempts
            << " runs: " << best_eigen_ms << " ms\n";

    // ===== Use last factorization in lu_ref for correctness checks =====
    Matrix LU_eigen = lu_ref.matrixLU();

    Matrix L_eigen = Matrix::Identity(N, N);
    L_eigen.triangularView<Eigen::StrictlyLower>() = LU_eigen.triangularView<Eigen::StrictlyLower>();

    Matrix U_eigen = LU_eigen.triangularView<Eigen::Upper>();

    Eigen::VectorXi piv_eigen = lu_ref.permutationP().indices();

    Matrix PG_eigen = lu_ref.permutationP() * G_eigen;
    Matrix LU_eigen_reconstructed = L_eigen * U_eigen;
    double eigen_residual = (PG_eigen - LU_eigen_reconstructed).norm();

    std::cout << "[Eigen] ‖P·G - L·U‖_F = " << eigen_residual << "\n";


    // ===================== MX: Unblocked LU =====================
    mx::Dense<double> LU_unblocked;  // will be re-used for timing
    std::vector<mx::index_t> piv_unblocked(N);

    double best_unblocked_ms = std::numeric_limits<double>::max();

    for (int attempt = 0; attempt < Nattempts; ++attempt) {
        LU_unblocked = G; // copy

        double t0 = CycleTimer::currentSeconds();
        mx::LUInfo info_unblocked = mx::lu_unblocked(LU_unblocked, piv_unblocked);
        double t1 = CycleTimer::currentSeconds();
        double elapsed_ms = (t1 - t0) * 1000.0;
        best_unblocked_ms = std::min(best_unblocked_ms, elapsed_ms);

        if (info_unblocked.status != mx::LUStatus::SUCCESS) {
            std::cerr << "[Unblocked] LU failed with status: " << info_unblocked << "\n";
            return 1;
        }
    }

    std::cout << "[LU Unblocked] best of " << Nattempts
              << " runs: " << best_unblocked_ms << " ms\n";

    // Extract L,U for unblocked
    mx::Dense<double> L_unblocked(N, N);
    mx::Dense<double> U_unblocked(N, N);
    extract_unit_lower(LU_unblocked, L_unblocked);
    extract_upper(LU_unblocked, U_unblocked);

    Matrix L_unblocked_e = L_unblocked.to_eigen();
    Matrix U_unblocked_e = U_unblocked.to_eigen();
    Matrix LU_unblocked_reconstructed = L_unblocked_e * U_unblocked_e;

    double res_unblocked = (PG_eigen - LU_unblocked_reconstructed).norm();
    double L_diff_unblocked = (L_unblocked_e - L_eigen).norm();
    double U_diff_unblocked = (U_unblocked_e - U_eigen).norm();

    std::cout << "[Unblocked] ‖P·G - L·U‖_F = " << res_unblocked << "\n";
    std::cout << "[Unblocked] ‖L_mx - L_eigen‖_F = " << L_diff_unblocked << "\n";
    std::cout << "[Unblocked] ‖U_mx - U_eigen‖_F = " << U_diff_unblocked << "\n";

    // // Pivot comparison (Eigen pivots are int; mx pivots are mx::index_t)
    // VectorI piv_unblocked_eiglike(N);
    // for (mx::index_t i = 0; i < N; ++i) {
    //     piv_unblocked_eiglike(i) = static_cast<mx::index_t>(piv_eigen(i));
    // }

    // double pivot_diff_unblocked =
    //     (piv_unblocked_eiglike - Eigen::Map<VectorI>(piv_unblocked.data(), N)).cast<double>().lpNorm<Eigen::Infinity>();

    // std::cout << "[Unblocked] max |pivot_mx - pivot_eigen| = "
    //           << pivot_diff_unblocked << "\n";

    // ===================== MX: Blocked LU =====================
    mx::Dense<double> LU_blocked;
    std::vector<mx::index_t> piv_blocked(N);

    double best_blocked_ms = std::numeric_limits<double>::max();

    for (int attempt = 0; attempt < Nattempts; ++attempt) {
        LU_blocked = G; // copy

        double t0 = CycleTimer::currentSeconds();
        mx::LUInfo info_blocked = mx::lu_blocked(LU_blocked, piv_blocked);
        double t1 = CycleTimer::currentSeconds();
        double elapsed_ms = (t1 - t0) * 1000.0;
        best_blocked_ms = std::min(best_blocked_ms, elapsed_ms);

        if (info_blocked.status != mx::LUStatus::SUCCESS) {
            std::cerr << "[Blocked] LU failed with status: " << info_blocked << "\n";
            return 1;
        }
    }

    std::cout << "[LU Blocked]   best of " << Nattempts
              << " runs: " << best_blocked_ms << " ms\n";

    // Extract L,U for blocked
    mx::Dense<double> L_blocked(N, N);
    mx::Dense<double> U_blocked(N, N);
    extract_unit_lower(LU_blocked, L_blocked);
    extract_upper(LU_blocked, U_blocked);

    Matrix L_blocked_e = L_blocked.to_eigen();
    Matrix U_blocked_e = U_blocked.to_eigen();
    Matrix LU_blocked_reconstructed = L_blocked_e * U_blocked_e;

    double res_blocked = (PG_eigen - LU_blocked_reconstructed).norm();
    double L_diff_blocked = (L_blocked_e - L_eigen).norm();
    double U_diff_blocked = (U_blocked_e - U_eigen).norm();

    std::cout << "[Blocked]   ‖P·G - L·U‖_F = " << res_blocked << "\n";
    std::cout << "[Blocked]   ‖L_mx - L_eigen‖_F = " << L_diff_blocked << "\n";
    std::cout << "[Blocked]   ‖U_mx - U_eigen‖_F = " << U_diff_blocked << "\n";

    // // Pivot comparison blocked vs Eigen
    // VectorI piv_blocked_eiglike(N);
    // for (mx::index_t i = 0; i < N; ++i) {
    //     piv_blocked_eiglike(i) = static_cast<mx::index_t>(piv_eigen(i));
    // }

    // double pivot_diff_blocked =
    //     (piv_blocked_eiglike - Eigen::Map<VectorI>(piv_blocked.data(), N)).cast<double>().lpNorm<Eigen::Infinity>();

    // std::cout << "[Blocked]   max |pivot_mx - pivot_eigen| = "
    //           << pivot_diff_blocked << "\n";

    // ===================== Unblocked vs Blocked (internal consistency) =====================
    Matrix LU_unblocked_e = LU_unblocked.to_eigen();
    Matrix LU_blocked_e   = LU_blocked.to_eigen();

    double LU_diff_unb_vs_blk = (LU_unblocked_e - LU_blocked_e).norm();

    std::cout << "[Unb vs Blk] ‖LU_unblocked - LU_blocked‖_F = "
              << LU_diff_unb_vs_blk << "\n";

    bool ok =
        (res_unblocked < 1e-10 * PG_eigen.norm()) &&
        (res_blocked   < 1e-10 * PG_eigen.norm()); // &&
        // (pivot_diff_unblocked == 0.0) &&
        // (pivot_diff_blocked   == 0.0);

    std::cout << "\nOverall status: " << (ok ? "OK ✅" : "FAIL ❌") << "\n";

    return ok ? 0 : 1;
}
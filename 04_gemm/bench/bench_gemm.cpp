#include <iostream>
#include <cstdio>
#include <functional>
#include <thread>

#include "mx_gemm/gemm.h"
#include "mx/utils/ostream.h"
#include "mx/dense_view.h"
#include <Eigen/Dense>
#include "CycleTimer.h"

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int main() {

    // ---- Configuration ----

    constexpr size_t   N_ATTEMPTS = 3;
    constexpr size_t   DIM        = 2000;
    constexpr double   ALPHA      = 1.0;
    constexpr double   BETA       = 2.0;

    using Layout = mx::RowMajor;
    using Mat    = mx::Dense<double, Layout>;

    // ---- Setup ----

    Mat A(DIM, DIM, 1.0);
    Mat B(DIM, DIM, 2.0);
    Mat C(DIM, DIM, 0.0);

    const Mat Zeros(DIM, DIM, 0.0);

    // ---- Timing & Validation Helper ----
    // Runs `fn` N_ATTEMPTS times (taking the min), resets C before each run,
    // validates against `expected`, and prints the result.

    auto benchmark = [&](const char* label, const Mat& expected, double baseline,
                         size_t attempts, std::function<void()> fn) -> double
    {
        double min_time = baseline;

        for (size_t i = 0; i < attempts; i++) {
            C = Zeros;
            double t0 = CycleTimer::currentSeconds();
            fn();
            double t1 = CycleTimer::currentSeconds();
            min_time = std::min(min_time, t1 - t0);
        }

        if (C == expected) {
            printf("[%-35s]: %8.3f ms  ( x %5.2f )\n", label, min_time * 1000, baseline / min_time);
        } else {
            printf("[%-35s]: MISMATCH\n", label);
        }

        return min_time;
    };

    // ---- Reference: Naive GEMM (single run) ----

    Mat Z(DIM, DIM, 0.0);
    double t0 = CycleTimer::currentSeconds();
    mx::gemm_naive(ALPHA, A, B, BETA, Z);
    double t1 = CycleTimer::currentSeconds();
    double naive_time = t1 - t0;
    printf("[%-35s]: %8.3f ms\n", "Naive GEMM (baseline)", naive_time * 1000);

    // ---- Reference: Eigen ----

    #ifdef EIGEN_USE_MKL_ALL
        std::cout << "Eigen was compiled with Intel MKL support enabled\n";
    #elif defined(EIGEN_USE_BLAS)
        std::cout << "Eigen was compiled with BLAS support enabled\n";
    #else
        std::cout << "Eigen is using its internal kernels\n";
    #endif

    Matrix A_eigen = A.to_eigen();
    Matrix B_eigen = B.to_eigen();
    Matrix C_eigen;

    double eigen_min = naive_time;
    for (size_t i = 0; i < N_ATTEMPTS; i++) {
        C_eigen = Matrix::Zero(DIM, DIM);
        t0 = CycleTimer::currentSeconds();
        C_eigen.noalias() = ALPHA * (A_eigen * B_eigen) + BETA * C_eigen;
        t1 = CycleTimer::currentSeconds();
        eigen_min = std::min(eigen_min, t1 - t0);
    }

    Mat Z_eigen(C_eigen.rows(), C_eigen.cols(), C_eigen.data());
    if (Z_eigen == Z) {
        printf("[%-35s]: %8.3f ms  ( x %5.2f )\n", "Eigen", eigen_min * 1000, naive_time / eigen_min);
    } else {
        printf("[%-35s]: MISMATCH\n", "Eigen");
    }

    // ---- Sequential GEMMs ----

    printf("\n ==== Sequential GEMMs ====\n\n");

    benchmark("GEMM Transposed", Z, naive_time, N_ATTEMPTS, [&]() {
        mx::gemm_transposed(ALPHA, A, B, BETA, C);
    });

    benchmark("GEMM Cache Blocked", Z, naive_time, N_ATTEMPTS, [&]() {
        mx::gemm_cache_blocked(ALPHA, A, B, BETA, C);
    });

    benchmark("GEMM Microkernel", Z, naive_time, N_ATTEMPTS, [&]() {
        mx::gemm_microkernel(ALPHA, A, B, BETA, C);
    });

    // ---- Parallel GEMMs (RowMajor only) ----

    if constexpr (std::is_same_v<Layout, mx::RowMajor>) {

        printf("\n ==== Parallel GEMMs ====\n\n");

        size_t Nthreads = std::thread::hardware_concurrency();
        printf("Hardware concurrency: %zu threads\n\n", Nthreads);

        benchmark("GEMM // Cache Blocked (Modular)", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_cache_blocked_experimental(ALPHA, A, B, BETA, C, Nthreads);
        });

        benchmark("GEMM // Cache Blocked (Monolithic)", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_cache_blocked(ALPHA, A, B, BETA, C, Nthreads);
        });

        benchmark("GEMM // Cyclic", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_cyclic(ALPHA, A, B, BETA, C, 21);
        });

        benchmark("GEMM // Partitions", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_block(ALPHA, A, B, BETA, C, 21);
        });

        benchmark("GEMM // Microtiles", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_microtiles(ALPHA, A, B, BETA, C, Nthreads);
        });

        benchmark("GEMM // Vectorized", Z, naive_time, N_ATTEMPTS, [&]() {
            mx::gemm_cpu_threads_vectorized(ALPHA, A, B, BETA, C, Nthreads);
        });

    } else {
        printf("\n/!\\ Parallel GEMMs are only supported for RowMajor layout.\n");
    }

    return 0;
}
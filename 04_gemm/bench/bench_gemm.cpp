// #include <iostream>
// #include <cstdio>
// #include <functional>
// #include <thread>

// #include "mx_gemm/gemm.h"
// #include "mx/utils/ostream.h"
// #include "mx/dense_view.h"
// #include <Eigen/Dense>
// #include "CycleTimer.h"

// #include "mx_gemm/detail/host/legacy/gemm_vectorized.h"

// using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// int main() {

//     // ---- Configuration ----

//     constexpr size_t   N_ATTEMPTS = 3;
//     constexpr size_t   DIM        = 2000;
//     constexpr double   ALPHA      = 1.0;
//     constexpr double   BETA       = 2.0;

//     using Layout = mx::RowMajor;
//     using Mat    = mx::Dense<double, Layout>;

//     // ---- Setup ----

//     Mat A(DIM, DIM, 1.0);
//     Mat B(DIM, DIM, 2.0);
//     Mat C(DIM, DIM, 0.0);

//     const Mat Zeros(DIM, DIM, 0.0);

//     // ---- Timing & Validation Helper ----
//     // Runs `fn` N_ATTEMPTS times (taking the min), resets C before each run,
//     // validates against `expected`, and prints the result.

//     auto benchmark = [&](const char* label, const Mat& expected, double baseline,
//                          size_t attempts, std::function<void()> fn) -> double
//     {
//         double min_time = baseline;

//         for (size_t i = 0; i < attempts; i++) {
//             C = Zeros;
//             double t0 = CycleTimer::currentSeconds();
//             fn();
//             double t1 = CycleTimer::currentSeconds();
//             min_time = std::min(min_time, t1 - t0);
//         }

//         if (C == expected) {
//             printf("[%-35s]: %8.3f ms  ( x %5.2f )\n", label, min_time * 1000, baseline / min_time);
//         } else {
//             printf("[%-35s]: MISMATCH\n", label);
//         }

//         return min_time;
//     };

//     // ---- Reference: Naive GEMM (single run) ----

//     Mat Z(DIM, DIM, 0.0);
//     double t0 = CycleTimer::currentSeconds();
//     mx::gemm<mx::detail::GemmAlgorithm::Naive>(ALPHA, A, B, BETA, Z);
//     double t1 = CycleTimer::currentSeconds();
//     double naive_time = t1 - t0;
//     printf("[%-35s]: %8.3f ms\n", "Naive GEMM (baseline)", naive_time * 1000);

//     // ---- Reference: Eigen ----

//     #ifdef EIGEN_USE_MKL_ALL
//         std::cout << "Eigen was compiled with Intel MKL support enabled\n";
//     #elif defined(EIGEN_USE_BLAS)
//         std::cout << "Eigen was compiled with BLAS support enabled\n";
//     #else
//         std::cout << "Eigen is using its internal kernels\n";
//     #endif

//     Matrix A_eigen = A.to_eigen();
//     Matrix B_eigen = B.to_eigen();
//     Matrix C_eigen;

//     double eigen_min = naive_time;
//     for (size_t i = 0; i < N_ATTEMPTS; i++) {
//         C_eigen = Matrix::Zero(DIM, DIM);
//         t0 = CycleTimer::currentSeconds();
//         C_eigen.noalias() = ALPHA * (A_eigen * B_eigen) + BETA * C_eigen;
//         t1 = CycleTimer::currentSeconds();
//         eigen_min = std::min(eigen_min, t1 - t0);
//     }

//     Mat Z_eigen(C_eigen.rows(), C_eigen.cols(), C_eigen.data());
//     if (Z_eigen == Z) {
//         printf("[%-35s]: %8.3f ms  ( x %5.2f )\n", "Eigen", eigen_min * 1000, naive_time / eigen_min);
//     } else {
//         printf("[%-35s]: MISMATCH\n", "Eigen");
//     }

//     // ---- Sequential GEMMs ----

//     printf("\n ==== Sequential GEMMs ====\n\n");

//     benchmark("GEMM Transposed", Z, naive_time, N_ATTEMPTS, [&]() {
//          mx::gemm<mx::detail::GemmAlgorithm::Transposed>(ALPHA, A, B, BETA, C);
//     });

//     benchmark("GEMM Cache Blocked", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::CacheBlocked>(ALPHA, A, B, BETA, C);
//     });

//     benchmark("GEMM Microkernel", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::Microkernel>(ALPHA, A, B, BETA, C);
//     });
    
//     benchmark("GEMM SIMD Vectorized", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::VectorizedMicrokernel>(ALPHA, A, B, BETA, C);
//     });

//     // ---- Parallel GEMMs (RowMajor only) ----

//     printf("\n ==== Parallel GEMMs ====\n\n");

//     mx::index_t Nthreads = std::thread::hardware_concurrency();
//     printf("Hardware concurrency: %d threads\n\n", Nthreads);

//     benchmark("GEMM // Cyclic", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::Transposed>(ALPHA, A, B, BETA, C, mx::CPU<mx::CyclicScheduler>{Nthreads});
//     });

//     benchmark("GEMM // Partitions", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::Transposed>(ALPHA, A, B, BETA, C, mx::CPU<mx::BlockScheduler>{Nthreads});
//     });

//     benchmark("GEMM // Cache Blocked", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::CacheBlocked>(ALPHA, A, B, BETA, C, mx::CPU<mx::BlockScheduler>{Nthreads});
//     });

//     benchmark("GEMM // Microkernel", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::Microkernel>(ALPHA, A, B, BETA, C, mx::CPU<mx::BlockScheduler>{Nthreads});
//     });

//     benchmark("GEMM // Vectorized", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm<mx::detail::GemmAlgorithm::VectorizedMicrokernel>(ALPHA, A, B, BETA, C, mx::CPU<mx::BlockScheduler>{Nthreads});
//     });

//     benchmark("GEMM // Vectorized (old)", Z, naive_time, N_ATTEMPTS, [&]() {
//         mx::gemm_cpu_threads_vectorized_no_unroll(ALPHA, A, B, BETA, C, Nthreads);
//     });

//     return 0;
// }


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "CycleTimer.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "mx_gemm/gemm.h"

using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Layout      = mx::RowMajor;
using Mat         = mx::Dense<double, Layout>;

struct BenchStats {
    double min_s    = 0.0;
    double median_s = 0.0;
};

template <class MatT>
void fill_random(MatT& X, double low, double high, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(low, high);

    for (mx::index_t i = 0; i < X.rows(); ++i) {
        for (mx::index_t j = 0; j < X.cols(); ++j) {
            X(i, j) = dist(gen);
        }
    }
}

template <class MatT>
bool allclose(const MatT& A, const MatT& B, double rtol = 1e-10, double atol = 1e-12) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;

    for (mx::index_t i = 0; i < A.rows(); ++i) {
        for (mx::index_t j = 0; j < A.cols(); ++j) {
            const double a = A(i, j);
            const double b = B(i, j);
            const double diff = std::abs(a - b);
            const double tol  = atol + rtol * std::abs(b);
            if (diff > tol) return false;
        }
    }
    return true;
}

double gflops(std::size_t M, std::size_t K, std::size_t N, double seconds) {
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    return flops / seconds / 1.0e9;
}

template <class Fn>
BenchStats run_benchmark(std::size_t warmup, std::size_t attempts, Fn&& fn) {
    for (std::size_t i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> times;
    times.reserve(attempts);

    for (std::size_t i = 0; i < attempts; ++i) {
        const double t0 = CycleTimer::currentSeconds();
        fn();
        const double t1 = CycleTimer::currentSeconds();
        times.push_back(t1 - t0);
    }

    std::sort(times.begin(), times.end());

    BenchStats out;
    out.min_s = times.front();

    if (times.size() % 2 == 1) {
        out.median_s = times[times.size() / 2];
    } else {
        const std::size_t i = times.size() / 2;
        out.median_s = 0.5 * (times[i - 1] + times[i]);
    }

    return out;
}

void print_result(const char* label,
                  const BenchStats& stats,
                  std::size_t M, std::size_t K, std::size_t N,
                  double baseline_median_s,
                  bool ok)
{
    if (!ok) {
        std::printf("[%-30s] : MISMATCH\n", label);
        return;
    }

    const double median_ms = stats.median_s * 1000.0;
    const double min_ms    = stats.min_s * 1000.0;
    const double perf      = gflops(M, K, N, stats.median_s);
    const double speedup   = baseline_median_s / stats.median_s;

    std::printf("[%-30s] : median %9.3f ms | min %9.3f ms | %8.2f GF/s | x %6.2f\n",
                label, median_ms, min_ms, perf, speedup);
}

template <class Fn>
BenchStats benchmark_kernel(const char* label,
                            Mat& C,
                            const Mat& C0,
                            const Mat& expected,
                            std::size_t warmup,
                            std::size_t attempts,
                            std::size_t M, std::size_t K, std::size_t N,
                            double baseline_median_s,
                            Fn&& fn)
{
    auto stats = run_benchmark(warmup, attempts, [&]() {
        C = C0;
        fn();
    });

    const bool ok = allclose(C, expected);
    print_result(label, stats, M, K, N, baseline_median_s, ok);
    return stats;
}

void print_case_header(std::size_t M, std::size_t K, std::size_t N, mx::index_t nthreads) {
    std::printf("\n==============================================================\n");
    std::printf("Case: A(%zu x %zu), B(%zu x %zu), C(%zu x %zu) | threads = %d\n",
                M, K, K, N, M, N, static_cast<int>(nthreads));
    std::printf("==============================================================\n");
}

int main() {
    constexpr std::size_t WARMUP     = 1;
    constexpr std::size_t N_ATTEMPTS = 5;
    constexpr double ALPHA           = 1.0;
    constexpr double BETA            = 2.0;

    const mx::index_t hw_threads = static_cast<mx::index_t>(std::thread::hardware_concurrency());
    const mx::index_t nthreads   = hw_threads > 0 ? hw_threads : 1;

#ifdef EIGEN_USE_MKL_ALL
    std::cout << "Eigen was compiled with Intel MKL support enabled\n";
#elif defined(EIGEN_USE_BLAS)
    std::cout << "Eigen was compiled with BLAS support enabled\n";
#else
    std::cout << "Eigen is using its internal kernels\n";
#endif

    std::cout << "Warmup runs  : " << WARMUP << "\n";
    std::cout << "Timed runs   : " << N_ATTEMPTS << "\n";
    std::cout << "Note: set your BLAS thread count externally to match " << nthreads
              << " threads for a fair comparison.\n";
    std::cout << "Examples:\n";
    std::cout << "  OpenBLAS: export OPENBLAS_NUM_THREADS=" << nthreads << "\n";
    std::cout << "  MKL     : export MKL_NUM_THREADS=" << nthreads << "\n";
    std::cout << "  OMP     : export OMP_NUM_THREADS=" << nthreads << "\n";

    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> cases = {
        {257, 131,  79},
        {1024, 768, 1536},
        {2000, 2000, 2000}
    };

    for (std::size_t case_id = 0; case_id < cases.size(); ++case_id) {
        const auto [M, K, N] = cases[case_id];
        print_case_header(M, K, N, nthreads);

        Mat A(M, K);
        Mat B(K, N);
        Mat C0(M, N);
        Mat C(M, N);

        fill_random(A,  -1.0, 1.0, 1000u + 10u * static_cast<unsigned>(case_id) + 1u);
        fill_random(B,  -1.0, 1.0, 1000u + 10u * static_cast<unsigned>(case_id) + 2u);
        fill_random(C0, -1.0, 1.0, 1000u + 10u * static_cast<unsigned>(case_id) + 3u);

        // ---- Reference: Naive ----
        Mat Z = C0;
        const BenchStats naive_stats = run_benchmark(WARMUP, N_ATTEMPTS, [&]() {
            Z = C0;
            mx::gemm<mx::detail::GemmAlgorithm::Naive>(ALPHA, A, B, BETA, Z);
        });

        std::printf("\nReference\n\n");
        std::printf("[%-30s] : median %9.3f ms | min %9.3f ms | %8.2f GF/s\n",
                    "Naive GEMM",
                    naive_stats.median_s * 1000.0,
                    naive_stats.min_s * 1000.0,
                    gflops(M, K, N, naive_stats.median_s));

        // ---- Reference: Eigen / BLAS ----
        EigenMatrix A_eigen  = A.to_eigen();
        EigenMatrix B_eigen  = B.to_eigen();
        EigenMatrix C0_eigen = C0.to_eigen();
        EigenMatrix C_eigen(M, N);

        const BenchStats eigen_stats = run_benchmark(WARMUP, N_ATTEMPTS, [&]() {
            C_eigen = C0_eigen;
            C_eigen.noalias() = ALPHA * (A_eigen * B_eigen) + BETA * C_eigen;
        });

        Mat Z_eigen(C_eigen.rows(), C_eigen.cols(), C_eigen.data());
        print_result("Eigen", eigen_stats, M, K, N, naive_stats.median_s, allclose(Z_eigen, Z));

        // ---- Sequential ----
        std::printf("\nSequential kernels\n\n");

        benchmark_kernel("GEMM Transposed", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::Transposed>(ALPHA, A, B, BETA, C);
                         });

        benchmark_kernel("GEMM Cache Blocked", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::CacheBlocked>(ALPHA, A, B, BETA, C);
                         });

        benchmark_kernel("GEMM Microkernel", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::Microkernel>(ALPHA, A, B, BETA, C);
                         });

        benchmark_kernel("GEMM SIMD Vectorized", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::VectorizedMicrokernel>(ALPHA, A, B, BETA, C);
                         });

        // ---- Parallel ----
        std::printf("\nParallel kernels\n\n");

        benchmark_kernel("GEMM // Cyclic", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::Transposed>(
                                 ALPHA, A, B, BETA, C,
                                 mx::CPU<mx::CyclicScheduler>{nthreads}
                             );
                         });

        benchmark_kernel("GEMM // Partitions", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::Transposed>(
                                 ALPHA, A, B, BETA, C,
                                 mx::CPU<mx::BlockScheduler>{nthreads}
                             );
                         });

        benchmark_kernel("GEMM // Cache Blocked", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::CacheBlocked>(
                                 ALPHA, A, B, BETA, C,
                                 mx::CPU<mx::BlockScheduler>{nthreads}
                             );
                         });

        benchmark_kernel("GEMM // Microkernel", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::Microkernel>(
                                 ALPHA, A, B, BETA, C,
                                 mx::CPU<mx::BlockScheduler>{nthreads}
                             );
                         });

        benchmark_kernel("GEMM // Vectorized", C, C0, Z,
                         WARMUP, N_ATTEMPTS, M, K, N, naive_stats.median_s,
                         [&]() {
                             mx::gemm<mx::detail::GemmAlgorithm::VectorizedMicrokernel>(
                                 ALPHA, A, B, BETA, C,
                                 mx::CPU<mx::BlockScheduler>{nthreads}
                             );
                         });
    }

    return 0;
}
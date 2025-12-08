#include<iostream>
#include<cstdio>
#include"gemm.h"
#include"mx/utils/ostream.h"
#include"mx/dense_view.h"

#include <Eigen/Dense>

#include"CycleTimer.h"

using Matrix  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int main(){
    const size_t Nattempts = 10;

    using Layout = mx::RowMajor; // mx::RowMajor (or mx::ColMajor but only for sequential GEMMs for now)

    mx::Dense<double, Layout> A(2000, 2000, 1.0);
    mx::Dense<double, Layout> B(2000, 2000, 2.0);
    mx::Dense<double, Layout> Z(2000, 2000, 0.0);

    mx::Dense<double, Layout> C(2000, 2000, 0.0);
    mx::Dense<double, Layout> Zeros(2000, 2000, 0.0);

    double alpha = 1.0;
    double beta  = 0.0;

    // Naive GEMM 
    double startTime = CycleTimer::currentSeconds();
    mx::gemm_reference(alpha, A, B, beta, Z);
    double endTime = CycleTimer::currentSeconds();
    auto seq_time = endTime - startTime;
    printf("[Naive GEMM]: %.3f ms\n", (endTime - startTime) * 1000);

    // Eigen matrices for validation
    #ifdef EIGEN_USE_BLAS
        std::cout << "Eigen is using BLAS\n";
    #endif
    
    #ifdef EIGEN_USE_MKL_ALL
        std::cout << "Eigen is using Intel MKL\n";
    #endif

    Matrix A_eigen = A.to_eigen();
    Matrix B_eigen = B.to_eigen();
    Matrix C_eigen(2000,2000);

    auto min_time = seq_time;

    for(size_t i=0; i<Nattempts; i++){
        startTime = CycleTimer::currentSeconds();
        C_eigen.noalias() = alpha * (A_eigen * B_eigen) + beta * C_eigen;
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);  
    }

    C = mx::Dense<double, Layout>(C_eigen.rows(), C_eigen.cols(), C_eigen.data()); //construct mx::Dense from Eigen matrix

    if(C == Z){
        printf("[Eigen]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // ++++++++++++ SEQUENTIAL GEMMs +++++++++++++++

    printf("\n");
    printf(" ==== Sequential GEMMs ====");
    printf("\n");

    // Naive GEMM with better locality
    startTime = CycleTimer::currentSeconds();
    mx::gemm_optimized(alpha, A, B, beta, C);
    endTime = CycleTimer::currentSeconds();
    min_time = endTime - startTime;
    if(C == Z){
        printf("[GEMM Optimized Locality]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // C = Zeros;

    // GEMM with L1/L2/L3 cache blocking 
    min_time = seq_time;
    
    startTime = CycleTimer::currentSeconds();
    mx::gemm_cache_blocked(alpha, A, B, beta, C);
    endTime = CycleTimer::currentSeconds();
    min_time = std::min(min_time, endTime - startTime);

    if(C == Z){
        printf("[GEMM Cache Blocked]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // ++++++++++++ PARALLEL GEMMs +++++++++++++++
    if constexpr (std::is_same_v<Layout, mx::RowMajor>) {
        // Parallel GEMMs are only supported for RowMajor layout for now

        printf("\n");
        printf(" ==== Parallel GEMMs ====");
        printf("\n");
    
        size_t Nthreads = std::thread::hardware_concurrency();
        std::cout << Nthreads << " concurrent threads are supported.\n";
    
        // Threaded GEMM with cache blocking (modulat implementaiton)
        min_time = seq_time;
        
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_cache_blocked_experimental(alpha, A, B, beta, C, Nthreads);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);
        }
    
        if(C == Z){
            printf("[GEMM // Cache Blocked (Modular)]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
    
        // Threaded GEMM with cache blocking 
        min_time = seq_time;
        
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_cache_blocked(alpha, A, B, beta, C, Nthreads);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);
        }
    
        if(C == Z){
            printf("[GEMM // Cache Blocked (Monolithic)]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
    
        // Naive strided parallel GEMM over rows 
        min_time = seq_time;
        
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_cyclic(alpha, A, B, beta, C, 21);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);
        }
    
        if(C == Z){
            printf("[GEMM // Strides]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
    
        // Naive partionned parallel GEMM over rows
        min_time = seq_time;
        
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_block(alpha, A, B, beta, C, 21);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);  
        }
    
        if(C == Z){
            printf("[GEMM // Partitions]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
    
        // Cache Blocks + Microtiling of parallel GEMM
        min_time = seq_time;
        
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_microtiles(alpha, A, B, beta, C, Nthreads);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);  
        }
    
        if(C == Z){
            printf("[GEMM // Microtiles]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
    
        // Cache Blocks + Microtiling + SIMD vectorization of parallel GEMM
        min_time = seq_time;
        for(size_t i=0; i<Nattempts; i++){
            startTime = CycleTimer::currentSeconds();
            mx::gemm_cpu_threads_vectorized(alpha, A, B, beta, C, Nthreads);
            endTime = CycleTimer::currentSeconds();
            min_time = std::min(min_time, endTime - startTime);  
        }
    
        if(C == Z){
            printf("[GEMM // Vectorized]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
        } else std::cout << "Mismatch in values...\n";
        
    } else {
        std::cout << "/!\\ Parallel GEMMs are only supported for RowMajor layout for now.\n";
    }

    return 0;
}
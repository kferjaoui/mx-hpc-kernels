#include<iostream>
#include<cstdio>
#include"gemm.h"
#include"mx/utils/ostream.h"
#include"mx/dense_view.h"

#include"CycleTimer.h"

template<typename T>
mx::DenseView<const T> as_const(mx::DenseView<T> view) noexcept{
    return mx::DenseView<const T>(view.begin(), view.rows(), view.cols(), view.row_stride(), view.col_stride());
}

int main(){
    const size_t Nattempts = 10;

    using Layout = mx::RowMajor;
    // using Layout = mx::ColMajor;

    mx::Dense<double, Layout> A(2000, 2000, 1.0);
    mx::Dense<double, Layout> B(2000, 2000, 2.0);
    mx::Dense<double, Layout> Z(2000, 2000, 0.0);

    mx::Dense<double, Layout> C(2000, 2000, 0.0);
    mx::Dense<double, Layout> Zeros(2000, 2000, 0.0);

    // ++++++++++++ SEQUENTIAL GEMMs +++++++++++++++

    // Naive GEMM 
    double startTime = CycleTimer::currentSeconds();
    mx::gemm_reference(A, B, Z);
    double endTime = CycleTimer::currentSeconds();
    auto seq_time = endTime - startTime;
    printf("[Naive GEMM]: %.3f ms\n", (endTime - startTime) * 1000);

    // Naive GEMM with better locality
    startTime = CycleTimer::currentSeconds();
    mx::gemm_optimized(A, B, C);
    endTime = CycleTimer::currentSeconds();
    auto min_time = endTime - startTime;
    if(C == Z){
        printf("[GEMM Optimized Locality]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    C = Zeros;

    // GEMM with L1/L2/L3 cache blocking 
    min_time = seq_time;
    
    startTime = CycleTimer::currentSeconds();
    mx::gemm_cache_blocked(A, B, C);
    endTime = CycleTimer::currentSeconds();
    min_time = std::min(min_time, endTime - startTime);

    if(C == Z){
        printf("[GEMM Cache Blocked]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // ++++++++++++ PARALLEL GEMMs +++++++++++++++\

    printf("\n");
    printf(" ==== Parallel GEMMs ====");
    printf("\n");

    size_t Nthreads = std::thread::hardware_concurrency();
    std::cout << Nthreads << " concurrent threads are supported.\n";

    // Threaded GEMM with cache blocking (modulat implementaiton)
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_cache_blocked_experimental(A, B, C, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);
    }

    if(C == Z){
        printf("[GEMM // Cache Blocked (Modular)]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // Threaded GEMM with cache blocking 
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_cache_blocked(A, B, C, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);
    }

    if(C == Z){
        printf("[GEMM // Cache Blocked (Monolithic)]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // Naive strided parallel GEMM over rows 
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_cyclic(A, B, C, 21);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);
    }

    if(C == Z){
        printf("[GEMM // Strides]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // Naive partionned parallel GEMM over rows
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_block(A, B, C, 21);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);  
    }

    if(C == Z){
        printf("[GEMM // Partitions]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // Cache Blocks + Microtiling of parallel GEMM
    min_time = seq_time;
    
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_microtiles(A, B, C, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);  
    }

    if(C == Z){
        printf("[GEMM // Microtiles]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";

    // Cache Blocks + Microtiling + SIMD vectorization of parallel GEMM
    min_time = seq_time;
    for(size_t i=0; i<Nattempts; i++){
        C = Zeros;
        startTime = CycleTimer::currentSeconds();
        mx::gemm_cpu_threads_vectorized(A, B, C, Nthreads);
        endTime = CycleTimer::currentSeconds();
        min_time = std::min(min_time, endTime - startTime);  
    }

    if(C == Z){
        printf("[GEMM // Vectorized]: %.3f ms ( x %.2f speed-up)\n", (min_time) * 1000, seq_time/min_time);
    } else std::cout << "Mismatch in values...\n";
    
    return 0;
}
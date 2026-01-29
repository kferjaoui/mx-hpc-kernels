#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <numeric>
#include <execution>

#include "CycleTimer.h"
#include "mx_reduction/reduction.h"

int main() {
    const long long num_elements = 1ll << 30; // 2^30 elements
    std::vector<double> v1(num_elements);
    for(long long i=0; i<num_elements; i++) v1[i] = 1.0;

    // ########### SUM REDUCTION BENCHMARK ###########
    std::cout << "#### SUM REDUCTION ####" << std::endl;

    double expected_sum = static_cast<double>(num_elements);
    printf("Expected sum: %f\n", expected_sum);

    // CPU Reduction (std::reduce for baseline)
    std::cout << ">>>> STD library <<<<" << std::endl;

    volatile double sink = 0.0;

    double start = CycleTimer::currentSeconds();
    double computed_sum_cpu_1 = std::reduce(v1.begin(), v1.end(), 0.0);
    double end = CycleTimer::currentSeconds();
    sink = computed_sum_cpu_1; // prevents optimization
    printf("[Serial] Time: %f seconds, sum=%f\n", end-start, computed_sum_cpu_1);
    
    start = CycleTimer::currentSeconds();
    double computed_sum_cpu_2 = std::reduce(std::execution::par, v1.begin(), v1.end(), 0.0);
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, sum=%f\n", end - start, computed_sum_cpu_2);

    // CPU Reduction (mx::reduce with parallel policy)
    std::cout << ">>>> MX library <<<<" << std::endl;

    start = CycleTimer::currentSeconds();
    double computed_sum_cpu_serial_mx = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CPU{1});
    end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, sum=%f\n", end - start, computed_sum_cpu_serial_mx);

    start = CycleTimer::currentSeconds();
    double computed_sum_cpu_parallel_mx = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CPU{2});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, sum=%f\n", end - start, computed_sum_cpu_parallel_mx);

    // CUDA Reduction
    start = CycleTimer::currentSeconds();
    double computed_sum_cuda_mx = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CUDA{});
    end = CycleTimer::currentSeconds();
    printf("[CUDA] Time: %f seconds, sum=%f\n", end - start, computed_sum_cuda_mx);

    // #####################################################

    std::cout << std::endl;

    // ########### PRODUCT REDUCTION BENCHMARK ###########
    std::cout << "#### PRODUCT REDUCTION ####" << std::endl;
    
    double expected_prod = static_cast<double>(2.0);
    printf("Expected prod: %f\n", expected_prod);

    // CPU Reduction (std::reduce for baseline)
    std::cout << ">>>> STD library <<<<" << std::endl;

    // volatile double sink = 0.0;

    start = CycleTimer::currentSeconds();
    double computed_prod_cpu_1 = std::reduce(v1.begin(), v1.end(), 2.0, std::multiplies<double>{});
    end = CycleTimer::currentSeconds();
    sink = computed_prod_cpu_1; // prevents optimization
    printf("[Serial] Time: %f seconds, prod=%f\n", end-start, computed_prod_cpu_1);
    
    start = CycleTimer::currentSeconds();
    double computed_prod_cpu_2 = std::reduce(std::execution::par, v1.begin(), v1.end(), 2.0, std::multiplies<double>{});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, prod=%f\n", end - start, computed_prod_cpu_2);

    // CPU Reduction (mx::reduce with parallel policy)
    std::cout << ">>>> MX library <<<<" << std::endl;

    start = CycleTimer::currentSeconds();
    double computed_prod_cpu_serial_mx = mx::reduce(v1.data(), v1.size(), 2.0, mx::Multiply<double>{}, mx::CPU{1});
    end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, prod=%f\n", end - start, computed_prod_cpu_serial_mx);

    start = CycleTimer::currentSeconds();
    double computed_prod_cpu_parallel_mx = mx::reduce(v1.data(), v1.size(), 2.0, mx::Multiply<double>{}, mx::CPU{2});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, prod=%f\n", end - start, computed_prod_cpu_parallel_mx);

    // CUDA Reduction
    start = CycleTimer::currentSeconds();
    double computed_prod_cuda_mx = mx::reduce(v1.data(), v1.size(), 2.0, mx::Multiply<double>{}, mx::CUDA{});
    end = CycleTimer::currentSeconds();
    printf("[CUDA] Time: %f seconds, prod=%f\n", end - start, computed_prod_cuda_mx);

    // #####################################################

    return 0;
}
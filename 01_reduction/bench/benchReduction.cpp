#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <numeric>
#include <execution>

#include "CycleTimer.h"
#include "mx_reduction/profiling/reduction_profiled.h"

int main() {
    using dataType = double;

    // ########### SUM REDUCTION BENCHMARK ###########
    std::cout << "#### SUM REDUCTION ####" << std::endl;

    const long long num_elements = 1ll << 30; // 2^30 elements
    std::vector<dataType> vSum(num_elements);
    for(long long i=0; i<num_elements; i++) vSum[i] = static_cast<dataType>(1.0);

    dataType expected_sum = static_cast<dataType>(num_elements);
    printf("Expected sum: %f\n", static_cast<double>(expected_sum));

    // CPU Reduction (std::reduce)
    std::cout << ">>>> STD library <<<<" << std::endl;

    double start = CycleTimer::currentSeconds();
    dataType computed_sum_cpu_1 = std::reduce(vSum.begin(), vSum.end(), 0);
    double end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, sum=%f\n", end-start, static_cast<double>(computed_sum_cpu_1));
    
    start = CycleTimer::currentSeconds();
    dataType computed_sum_cpu_2 = std::reduce(std::execution::par, vSum.begin(), vSum.end(), static_cast<dataType>(0));
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, sum=%f\n", end - start, static_cast<double>(computed_sum_cpu_2));

    std::cout << std::endl;

    // CPU Reduction (mx::reduce with parallel policy)
    std::cout << ">>>> MX library <<<<" << std::endl;

    start = CycleTimer::currentSeconds();
    dataType computed_sum_cpu_serial_mx = mx::profile::reduce_profiled(vSum.data(), vSum.size(), static_cast<dataType>(0), mx::Sum<dataType>{}, mx::CPU<>{1});
    end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, sum=%f\n", end - start, static_cast<double>(computed_sum_cpu_serial_mx));

    std::cout << std::endl;

    start = CycleTimer::currentSeconds();
    dataType computed_sum_cpu_parallel_mx = mx::profile::reduce_profiled(vSum.data(), vSum.size(), static_cast<dataType>(0), mx::Sum<dataType>{}, mx::CPU<>{2});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, sum=%f\n", end - start, static_cast<double>(computed_sum_cpu_parallel_mx));

    std::cout << std::endl;

    // CUDA Reduction
    start = CycleTimer::currentSeconds();
    dataType computed_sum_cuda_mx = mx::profile::reduce_profiled(vSum.data(), vSum.size(), static_cast<dataType>(0), mx::Sum<dataType>{}, mx::CUDA{256, 2048});
    end = CycleTimer::currentSeconds();
    printf("[CUDA + PCIe] Time: %f seconds, sum=%f\n", end - start, static_cast<double>(computed_sum_cuda_mx));

    // #####################################################

    std::cout << std::endl;

    // ########### PRODUCT REDUCTION BENCHMARK ###########
    std::cout << "#### PRODUCT REDUCTION ####" << std::endl;

    const long long N2 = 1ll << 20; // 2^20 elements
    std::vector<dataType> vProd(N2);
    for(long long i=0; i<N2; i++) vProd[i] = static_cast<dataType>(1.0 + ((i & 1) ? 1e-6 : -1e-6)); // alternating values around 1 to avoid overflow
    
    dataType expected_prod = static_cast<dataType>(0.9999994757); // precomputed expected product
    printf("Expected prod: %f\n", static_cast<double>(expected_prod));

    std::cout << ">>>> STD library <<<<" << std::endl;
    
    // CPU Reduction (std::reduce)
    start = CycleTimer::currentSeconds();
    dataType computed_prod_cpu_1 = std::reduce(vProd.begin(), vProd.end(), static_cast<dataType>(1.0), std::multiplies<dataType>{});
    end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, prod=%f\n", end-start, static_cast<double>(computed_prod_cpu_1));
    
    start = CycleTimer::currentSeconds();
    dataType computed_prod_cpu_2 = std::reduce(std::execution::par, vProd.begin(), vProd.end(), static_cast<dataType>(1.0), std::multiplies<dataType>{});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, prod=%f\n", end - start, static_cast<double>(computed_prod_cpu_2));
    
    std::cout << std::endl;

    std::cout << ">>>> MX library <<<<" << std::endl;
    
    // CPU Reduction (mx::reduce with parallel policy)
    start = CycleTimer::currentSeconds();
    dataType computed_prod_cpu_serial_mx = mx::profile::reduce_profiled(vProd.data(), vProd.size(), static_cast<dataType>(1.0), mx::Multiply<dataType>{}, mx::CPU<>{1});
    end = CycleTimer::currentSeconds();
    printf("[Serial] Time: %f seconds, prod=%f\n", end - start, static_cast<double>(computed_prod_cpu_serial_mx));

    std::cout << std::endl;

    start = CycleTimer::currentSeconds();
    dataType computed_prod_cpu_parallel_mx = mx::profile::reduce_profiled(vProd.data(), vProd.size(), static_cast<dataType>(1.0), mx::Multiply<dataType>{}, mx::CPU<>{2});
    end = CycleTimer::currentSeconds();
    printf("[Parallel] Time: %f seconds, prod=%f\n", end - start, static_cast<double>(computed_prod_cpu_parallel_mx));

    std::cout << std::endl;

    // CUDA Reduction
    start = CycleTimer::currentSeconds();
    dataType computed_prod_cuda_mx = mx::profile::reduce_profiled(vProd.data(), vProd.size(), static_cast<dataType>(1.0), mx::Multiply<dataType>{}, mx::CUDA{256, 2048});
    end = CycleTimer::currentSeconds();
    printf("[CUDA + PCIe] Time: %f seconds, prod=%f\n", end - start, static_cast<double>(computed_prod_cuda_mx));

    // #####################################################

    return 0;
}
#include <cmath>
#include <limits>
#include <vector>

#include "CycleTimer.h"
#include "mx_reduction/reduction.h"

int main() {
    long long num_elements = 1ll << 30; // 2^30 elements
    std::vector<double> v1(num_elements);
    for(long long i=0; i<num_elements; i++) v1[i] = 1.0;

    double expected_sum = static_cast<double>(num_elements);

    // CPU Reduction
    double start = CycleTimer::currentSeconds();
    double computed_sum_cpu = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CPU{8});
    double end = CycleTimer::currentSeconds();
    printf("CPU Reduction Time: %f seconds\n", end - start);
    printf("CPU Reduction Result: %f (Expected: %f)\n", computed_sum_cpu, expected_sum);

    // CUDA Reduction
    start = CycleTimer::currentSeconds();
    double computed_sum_cuda = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CUDA{});
    end = CycleTimer::currentSeconds();
    printf("CUDA Reduction Time: %f seconds\n", end - start);
    printf("CUDA Reduction Result: %f (Expected: %f)\n", computed_sum_cuda, expected_sum);

}
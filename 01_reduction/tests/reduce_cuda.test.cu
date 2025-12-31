#include <cstdio>
#include <vector>
#include <cmath>

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduction.h"

int main(){
    std::vector<float> v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        
    float sum = mx::reduce(v1.data(), v1.size(), 0.0f, mx::Sum<float>{}, mx::CUDA{});
    float max = mx::reduce(v1.data(), v1.size(), std::numeric_limits<float>::lowest(), mx::Max<float>{}, mx::CUDA{});
    float min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<float>::max(), mx::Min<float>{}, mx::CUDA{});
    float product = mx::reduce(v1.data(), v1.size(), 1.0f, mx::Multiply<float>{}, mx::CUDA{});
    
    printf("Sum: %f\n", sum); // Expected output: Sum: 55.000000
    printf("Max: %f\n", max); // Expected output: Max: 10.000000
    printf("Min: %f\n", min); // Expected output: Min: 1.000000
    printf("Product: %f\n", product); // Expected output: Product: 3628800.000000

    return 0;
}
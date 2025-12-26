#include <cstdio>
#include <vector>
#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduction.h"
#include <cmath>

int main(){
    std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        
    // double sum = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CPU{4});
    // double max = mx::reduce(v1.data(), v1.size(), -std::numeric_limits<double>::infinity(), mx::Max<double>{}, mx::CPU{4});
    // double min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::infinity(), mx::Min<double>{}, mx::CPU{4});
    // double prod = mx::reduce(v1.data(), v1.size(), 1.0, mx::Multiply<double>{}, mx::CPU{4});
    
    double sum = mx::reduce(v1.data(), v1.size(), 0.0, std::plus<double>{}, mx::CPU{4});
    double max = mx::reduce(v1.data(), v1.size(), -std::numeric_limits<double>::infinity(), std::max<double>, mx::CPU{4});
    double min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::infinity(), std::min<double>, mx::CPU{4});
    double prod = mx::reduce(v1.data(), v1.size(), 1.0, std::multiplies<double>{}, mx::CPU{4});
    
    printf("Sum: %f\n", sum); // Expected output: Sum: 55.000000
    printf("Max: %f\n", max); // Expected output: Max: 10.000000
    printf("Min: %f\n", min); // Expected output: Min: 1.000000
    printf("Product: %f\n", prod); // Expected output: Product : 3628800.000000

    return 0;
}
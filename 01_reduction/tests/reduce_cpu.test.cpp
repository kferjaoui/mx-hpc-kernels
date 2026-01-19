#include <cstdio>
#include <vector>
#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduction.h"
#include <cmath>

int main(){
    int maxintfactorial = 170; // beyond this, double overflows i.e. 171! overflows to infinity
    std::vector<double> v1(maxintfactorial);
    for(int i=0; i<maxintfactorial; i++) v1[i] = (double)(i+1);
        
    double sum = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CPU{8});
    double max = mx::reduce(v1.data(), v1.size(), -std::numeric_limits<double>::infinity(), mx::Max<double>{}, mx::CPU{8});
    double min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::infinity(), mx::Min<double>{}, mx::CPU{8});
    double prod = mx::reduce(v1.data(), v1.size(), 1.0, mx::Multiply<double>{}, mx::CPU{8});
    
    // double sum = mx::reduce(v1.data(), v1.size(), 0.0, std::plus<double>{}, mx::CPU{4});
    // double max = mx::reduce(v1.data(), v1.size(), -std::numeric_limits<double>::infinity(), std::max<double>, mx::CPU{4});
    // double min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::infinity(), std::min<double>, mx::CPU{4});
    // double prod = mx::reduce(v1.data(), v1.size(), 1.0, std::multiplies<double>{}, mx::CPU{4});
    
    printf("Sum: %f\n", sum);
    printf("Max: %f\n", max);
    printf("Min: %f\n", min);
    printf("Product: %f\n", prod);

    return 0;
}
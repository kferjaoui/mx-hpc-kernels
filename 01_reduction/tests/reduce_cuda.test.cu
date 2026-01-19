#include <cstdio>
#include <vector>
#include <cmath>

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduction.h"

int main(){
    int maxintfactorial = 170; // beyond this, double overflows
    std::vector<double> v1(maxintfactorial);
    for(int i=0; i<maxintfactorial; i++) v1[i] = (double)(i+1);

    // printf("Max of double: %f \n", std::numeric_limits<double>::max());

    double sum = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CUDA{});
    double max = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::lowest(), mx::Max<double>{}, mx::CUDA{});
    double min = mx::reduce(v1.data(), v1.size(), std::numeric_limits<double>::max(), mx::Min<double>{}, mx::CUDA{});
    double product = mx::reduce(v1.data(), v1.size(), 1.0, mx::Multiply<double>{}, mx::CUDA{});
    
    printf("Sum: %f\n", sum);
    printf("Max: %f\n", max);
    printf("Min: %f\n", min);
    printf("Product: %f\n", product);

    return 0;
}
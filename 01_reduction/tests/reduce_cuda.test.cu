#include <cstdio>
#include <vector>
#include <cmath>

#include "mx_reduction/policy.h"
#include "mx_reduction/operations.h"
#include "mx_reduction/reduction.h"

int main(){
    std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        
    double sum = mx::reduce(v1.data(), v1.size(), 0.0, mx::Sum<double>{}, mx::CUDA{});
    
    printf("Sum: %f\n", sum); // Expected output: Sum: 55.000000

    return 0;
}
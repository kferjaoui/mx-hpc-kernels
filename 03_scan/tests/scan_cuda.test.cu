
#include <cstdio>

#include "mx/utils/operations.h"
#include "mx/utils/policy.h"
#include "mx_scan/scan.h"

int main() {
    size_t N = 32;
    int* inputArray = new int[N];
    int* outputSerial = new int[N];
    int* outputParallel_1 = new int[N];
    int* outputParallel_2 = new int[N];

    for (int i=0; i<N; i++) inputArray[i] = i;

    // Print input array
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%i", inputArray[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    // Serial 
    mx::scan<mx::ScanType::Inclusive>(inputArray, outputSerial, N, mx::Sum<int>{}, mx::CPU{});

    // Print result of inclusive scan
    printf("[Serial] Inclusive Scan: ");
    for (int i = 0; i < N; i++) {
        printf("%i", outputSerial[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    // Parallel 
    mx::scan<mx::ScanType::Inclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(inputArray, outputParallel_1, N, mx::Sum<int>{}, mx::CUDA{});

    printf("[CUDA] Hillis_Steele Inclusive Scan: \n");
    for (int i = 0; i < N; i++) {
        printf("%i", outputParallel_1[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    mx::scan<mx::ScanType::Exclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(inputArray, outputParallel_2, N, mx::Sum<int>{}, mx::CUDA{});

    printf("[CUDA] Hillis_Steele Exclusive Scan: \n");
    for (int i = 0; i < N; i++) {
        printf("%i", outputParallel_2[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    delete[] inputArray;
    delete[] outputSerial;
    delete[] outputParallel_1;
    delete[] outputParallel_2;

    return 0;
}
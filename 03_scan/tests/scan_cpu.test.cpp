
#include <cstdio>

#include "mx/utils/operations.h"
#include "mx/utils/policy.h"
#include "mx_scan/scan_cpu.h"
#include "mx_scan/scan.h"

int main() {
    size_t N = 32;
    int* inputArray = new int[N];
    int* outputSerial = new int[N];
    int* outputParallel = new int[N];

    for (int i=0; i<N; i++) inputArray[i] = i;

    // Print input array
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%i", inputArray[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    mx::scan<mx::ScanType::Exclusive>(inputArray, outputSerial, N, mx::Sum<int>{}, mx::CPU{});
    mx::scan<mx::ScanType::Exclusive>(inputArray, outputParallel, N, mx::Sum<int>{}, mx::CPU{3});

    // Print result of inclusive scan
    printf("[Serial] Inclusive Scan: ");
    for (int i = 0; i < N; i++) {
        printf("%i", outputSerial[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    printf("[Parallel] Inclusive Scan: ");
    for (int i = 0; i < N; i++) {
        printf("%i", outputParallel[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    delete[] inputArray;
    delete[] outputSerial;
    delete[] outputParallel;

    return 0;
}
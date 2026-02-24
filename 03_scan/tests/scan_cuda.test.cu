
#include <cstdio>

#include "mx/utils/operations.h"
#include "mx/utils/policy.h"
#include "mx_scan/scan.h"

int main() {
    size_t N = 1222;
    int* inputArray = new int[N];
    int* outputSerial = new int[N];
    int* outputParallel_hs1 = new int[N];
    int* outputParallel_hs2 = new int[N];
    int* outputParallel_bl1 = new int[N];
    int* outputParallel_bl2 = new int[N];


    for (int i=0; i<N; i++) inputArray[i] = 1;

    // // Print input array
    // printf("Input: ");
    // for (int i = 0; i < N; i++) {
    //     printf("%i", inputArray[i]);
    //     if (i < N - 1) printf(", ");
    // }
    // printf("\n");

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
    // 1. Inclusive

    printf(">>>>>>>>>>>>>>> Inclusive Scan <<<<<<<<<<<<<<<<\n");

    // mx::scan<mx::ScanType::Inclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(inputArray, outputParallel_hs1, N, mx::Sum<int>{}, mx::CUDA{});

    // printf("[CUDA] Hillis_Steele: \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%i", outputParallel_hs1[i]);
    //     if (i < N - 1) printf(", ");
    // }
    // printf("\n");

    mx::scan<mx::ScanType::Inclusive, mx::detail::ScanAlgorithm::Blelloch>(inputArray, outputParallel_bl1, N, mx::Sum<int>{}, mx::CUDA{});

    printf("[CUDA] Blelloch: \n");
    for (int i = 0; i < N; i++) {
        printf("%i", outputParallel_bl1[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");
    
    // 2. Exclusive
    printf(">>>>>>>>>>>>>>> Exclusive Scan <<<<<<<<<<<<<<<<\n");

    // mx::scan<mx::ScanType::Exclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(inputArray, outputParallel_hs2, N, mx::Sum<int>{}, mx::CUDA{});

    // printf("[CUDA] Hillis_Steele : \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%i", outputParallel_hs2[i]);
    //     if (i < N - 1) printf(", ");
    // }
    // printf("\n");
    
    mx::scan<mx::ScanType::Exclusive, mx::detail::ScanAlgorithm::Blelloch>(inputArray, outputParallel_bl2, N, mx::Sum<int>{}, mx::CUDA{});

    printf("[CUDA] Blelloch: \n");
    for (int i = 0; i < N; i++) {
        printf("%i", outputParallel_bl2[i]);
        if (i < N - 1) printf(", ");
    }
    printf("\n");

    delete[] inputArray;
    delete[] outputSerial;
    delete[] outputParallel_hs1;
    delete[] outputParallel_hs2;
    delete[] outputParallel_bl1;
    delete[] outputParallel_bl2;

    return 0;
}
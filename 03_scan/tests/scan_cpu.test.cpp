#include <cstdio>
#include <vector>
#include <thread>
#include <algorithm>

#include "mx/utils/operations.h"
#include "mx/utils/policy.h"
#include "mx_scan/scan.h"

static void print_preview(const char* label, const std::vector<int>& v, std::size_t k = 8) {
    std::printf("%s (n=%zu): ", label, v.size());
    std::size_t n = v.size();
    std::size_t kk = std::min(k, n);
    for (std::size_t i = 0; i < kk; ++i) std::printf("%d ", v[i]);
    if (n > kk) std::printf("... ");
    if (n > kk) {
        // show last few too
        std::size_t start = (n > kk) ? std::max(kk, n - kk) : 0;
        for (std::size_t i = start; i < n; ++i) std::printf("%d ", v[i]);
    }
    std::printf("\n");
}

static bool equal_vec(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <mx::ScanType ST, mx::detail::ScanAlgorithm Algo>
static void run_case_on_cpu(const char* name,
                          const std::vector<int>& input,
                          const std::vector<int>& ref,
                          mx::CPU pol)
{
    std::vector<int> out(input.size(), 0);

    mx::scan<ST, Algo>(input.data(), out.data(), input.size(), mx::Sum<int>{}, pol);

    print_preview(name, out);

    if (!equal_vec(out, ref)) {
        std::printf("\033[31m MISMATCH vs Sequential CPU reference\033[0m\n");
    } else {
        std::printf("\033[32m OK\033[0m\n");
    }
}

int main() {
    std::size_t N = (1<<20) + 259;

    std::vector<int> input(N, 1);
    std::vector<int> ref_inclusive(N, 0);
    std::vector<int> ref_exclusive(N, 0);

    // Sequential CPU references
    mx::scan<mx::ScanType::Inclusive>(input.data(), ref_inclusive.data(), N, mx::Sum<int>{});
    mx::scan<mx::ScanType::Exclusive>(input.data(), ref_exclusive.data(), N, mx::Sum<int>{});

    print_preview("[CPU] inclusive", ref_inclusive);
    print_preview("[CPU] exclusive", ref_exclusive);

    int nThreads{1};
    printf("Choose number of threads (max %i): \n", std::thread::hardware_concurrency());
    std::cin >> nThreads; 
    mx::CPU pol{nThreads};

    std::printf(">>>>>>>>>>>>>>> CPU Inclusive <<<<<<<<<<<<<<<<\n");
    run_case_on_cpu<mx::ScanType::Inclusive, mx::detail::ScanAlgorithm::Blelloch>(
        "[CPU] Blelloch inclusive", input, ref_inclusive, pol);

    // Hillis–Steele is enabled for N <= 1024
    if (N <= 1024){
        run_case_on_cpu<mx::ScanType::Inclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(
            "[CPU] Hillis-Steele inclusive", input, ref_inclusive, pol);
    }

    std::printf(">>>>>>>>>>>>>>> CPU Exclusive <<<<<<<<<<<<<<<<\n");
    run_case_on_cpu<mx::ScanType::Exclusive, mx::detail::ScanAlgorithm::Blelloch>(
        "[CPU] Blelloch exclusive", input, ref_exclusive, pol);

    if (N <= 1024) {
        run_case_on_cpu<mx::ScanType::Exclusive, mx::detail::ScanAlgorithm::Hillis_Steele>(
            "[CPU] Hillis-Steele exclusive", input, ref_exclusive, pol);
    }

    return 0;
}
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mx_scan/scan_cuda.h"
#include "mx_scan/detail/device/scan_cuda_device.cuh"

namespace mx {

template<ScanType scan_type, detail::ScanAlgorithm scan_algo, typename T, typename Op>
void scan_cuda(const T* input, T* output, size_t size, Op op, const CUDA& cuda_policy)
{
    if (size == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_policy.stream);

    const size_t nbytes = size * sizeof(T);

    T* d_in  = nullptr;
    T* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  nbytes));
    CUDA_CHECK(cudaMalloc(&d_out, nbytes));

    CUDA_CHECK(cudaMemcpyAsync(d_in, input, nbytes, cudaMemcpyHostToDevice, stream));

    mx::detail::scan_cuda_device<scan_type, scan_algo>(d_in, d_out, size, op, cuda_policy);

    CUDA_CHECK(cudaMemcpyAsync(output, d_out, nbytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// Explicit template instantiations for supported types and operations
#define INSTANTIATE_SCAN(T, OpName)                                                        \
    template void scan_cuda<ScanType::Inclusive,  detail::ScanAlgorithm::Blelloch>(         \
        const T*, T*, size_t, OpName<T>, const CUDA&);                                 \
    template void scan_cuda<ScanType::Inclusive,  detail::ScanAlgorithm::Hillis_Steele>(    \
        const T*, T*, size_t, OpName<T>, const CUDA&);                                 \
    template void scan_cuda<ScanType::Exclusive,  detail::ScanAlgorithm::Blelloch>(         \
        const T*, T*, size_t, OpName<T>, const CUDA&);                                 \
    template void scan_cuda<ScanType::Exclusive,  detail::ScanAlgorithm::Hillis_Steele>(    \
        const T*, T*, size_t, OpName<T>, const CUDA&);

#define INSTANTIATE_SCAN_ALL_OPS(T)  \
    INSTANTIATE_SCAN(T, Sum)         \
    INSTANTIATE_SCAN(T, Multiply)    \
    INSTANTIATE_SCAN(T, Max)         \
    INSTANTIATE_SCAN(T, Min)

INSTANTIATE_SCAN_ALL_OPS(double)
INSTANTIATE_SCAN_ALL_OPS(int)
// INSTANTIATE_SCAN_ALL_OPS(float)  // one line to add a new type

#undef INSTANTIATE_SCAN
#undef INSTANTIATE_SCAN_ALL_OPS

} // namespace mx
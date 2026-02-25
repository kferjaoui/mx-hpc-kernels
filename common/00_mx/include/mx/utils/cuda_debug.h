#pragma once
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mx/utils/policy.h"

namespace mx::cuda_debug {

// Check launch error (same as CUDA_CHECK(cudaGetLastError()))
inline void launch_check() {
    CUDA_CHECK(cudaGetLastError());
}

// Sync only if debug_sync is enabled in the policy
inline void sync_if_debug(const mx::CUDA& policy, cudaStream_t stream) {
    if (policy.debug_sync) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

// Convenience: do both
inline void post_launch(const mx::CUDA& policy, cudaStream_t stream) {
    launch_check();
    sync_if_debug(policy, stream);
}

} // namespace mx::cuda_debug
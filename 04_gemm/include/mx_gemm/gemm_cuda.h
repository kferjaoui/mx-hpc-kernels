#pragma once

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/layout.h"
#include "mx/types.h"
#include "mx/utils/policy.h"
#include "mx_gemm/detail/algorithms.h"
#include "mx_gemm/detail/device/gemm_naive.cuh"


namespace mx {


template<detail::CudaGemmAlgorithm gemm_algo, typename T, class Layout>
void gemm_cuda(const T alpha, 
            DenseView<const T, Layout> A, 
            DenseView<const T, Layout> B,
            const T beta, 
            DenseView<T, Layout> C,
            const CUDA& cuda_policy); 

}
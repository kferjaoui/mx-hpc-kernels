#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"gemm_parallel.h"
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

namespace mx_detail{
    
struct BlockParameters {
    const index_t nc = 256; // rows of A/C per block
    const index_t kc = 256; // depth of A/B per block
    const index_t mc = 96; // columns of B/C per block
};

template<class BlockKernel>
void parallel_blocked_impl(index_t N, index_t K, index_t M,
                           index_t numThreads,
                           const BlockParameters& params,
                           BlockKernel&& kernel)
{
    const index_t nc = params.nc;
    const index_t kc = params.kc;
    const index_t mc = params.mc;

    index_t Nb = (N + nc - 1) / nc; // number of row blocks 
    index_t Kb = (K + kc - 1) / kc; // number of depth blocks
    index_t Mb = (M + mc - 1) / mc; // number of column blocks

    numThreads = numThreads? std::min(numThreads, Mb) : 1;

    auto baseWork  = Mb / numThreads;
    auto remainder = Mb % numThreads;

    auto workFunction = [=, &kernel](index_t tid){
        auto workChunk = baseWork + (tid<remainder? 1:0); 
        auto C_col_start = tid * baseWork + std::min(tid,remainder);
        auto C_col_end = C_col_start + workChunk;
    
        for(index_t Mi = C_col_start; Mi<C_col_end; Mi++){       // loop over column blocks of C allocated to the running thread
            const index_t jc = Mi*mc;
            const index_t jend = std::min(Mi*mc + mc, M);

            for(index_t Ki = 0; Ki<Kb; Ki++){                    // loop over depth blocks of A/B   
                const index_t pc = Ki*kc;
                const index_t pend = std::min(Ki*kc + kc, K);
                
                for(index_t Ni = 0; Ni<Nb; Ni++){                // loop over row blocks of C
                    const index_t ic = Ni*nc;
                    const index_t iend = std::min(Ni*nc + nc, N);

                    kernel(ic, iend, jc, jend, pc, pend);

                }
            }
        }     
    };
    
    // std::cout << "[Blocked //] Spawning "<< numThreads << " concurrent threads...\n";

    launch_threads(numThreads, workFunction);

}

} // namespace mx_detail

// Public APIs:

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_cache_blocked_experimental(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C, index_t numThreads = 8){
    gemm_cpu_threads_cache_blocked_experimental(A.view(), B.view(), C.view(), numThreads);
}

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_cache_blocked_experimental(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C, index_t numThreads = 8){
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    Dense<T, Layout> BT(M, K);
    for(index_t r=0; r<K; r++){
        for(index_t c=0; c<M; c++){
            BT(c,r) = B(r,c); 
        }
    }

    mx_detail::BlockParameters block_params{};

    auto gemm_kernel = [&](index_t ic, index_t iend, index_t jc, index_t jend, index_t pc, index_t pend){
        for(index_t i=ic; i<iend; i++){
            for(index_t j=jc; j<jend; j++){
                T sum{}; // register accumulator
                for(index_t p=pc; p<pend; p++){
                    sum += A(i,p) * BT(j,p);
                }
                C(i,j) += sum; // accumulate the current K-block
            }
        }
    };

    mx_detail::parallel_blocked_impl(N,K, M, numThreads, block_params, gemm_kernel);

}


}
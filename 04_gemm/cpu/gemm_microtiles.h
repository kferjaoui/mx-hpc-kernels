#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"gemm_parallel.h"
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_microtiles(const T alpha,
                                 const Dense<T, Layout>& A, 
                                 const Dense<T, Layout>& B, 
                                 const T beta,
                                 Dense<T, Layout>& C, 
                                 index_t numThreads = 8)
{
    gemm_cpu_threads_microtiles(alpha, A.view(), B.view(), beta, C.view(), numThreads);
}

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_microtiles(const T alpha,
                                 DenseView<const T, Layout> A, 
                                 DenseView<const T, Layout> B, 
                                 const T beta, 
                                 DenseView<T, Layout> C, 
                                 index_t numThreads = 8)
{
    static_assert(DenseView<const T, Layout>::is_row_major,
                  "gemm_cpu_threads_microtiles() currently supports RowMajor only");

    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    // scale C by beta
    if (beta != T{1}) {
        for(index_t r=0; r<N; r++)
            for(index_t c=0; c<M; c++)
                C(r,c) *= beta;
    }

    const index_t nc = 256; // rows of A/C per block
    const index_t kc = 256; // depth of A/B per block
    const index_t mc = 96; // columns of B/C per block

    index_t Nb = (N + nc - 1) / nc; // number of row blocks 
    index_t Kb = (K + kc - 1) / kc; // number of depth blocks
    index_t Mb = (M + mc - 1) / mc; // number of column blocks

    numThreads = numThreads? std::min(numThreads, Mb) : 1;

    auto baseWork  = Mb / numThreads;
    auto remainder = Mb % numThreads;

    auto workFunction = [&, M, K, N, numThreads](index_t tid){
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
                    
                    constexpr index_t nr = 4; // micro-block rows
                    constexpr index_t mr = 8; // micro-block columns

                    index_t Nnr = (iend-ic + nr -1)/nr; // number of micro-tiles in a row block
                    index_t Nmr = (jend-jc + mr -1)/mr; // number of micro-tiles in a column block
                    
                    // Compute the micro-tile C_micro(N_micro, M_micro) = A_micro(N_micro, :) * B_micro(:, M_micro)
                    // where A_micro is (nr x K), B_micro is (K x mr) and C_micro is (nr x mr)
                    for(index_t N_micro=0; N_micro<Nnr; N_micro++){
                        const index_t i0_micro = ic + N_micro*nr;       // global starting row index of the micro-tile
                        const index_t i_valid = std::min(nr, iend - i0_micro); // number of valid rows in the micro-tile

                        for(index_t M_micro=0; M_micro<Nmr; M_micro++){
                            const index_t j0_micro = jc + M_micro*mr;   // global starting column index of the micro-tile
                            const index_t j_valid = std::min(mr, jend - j0_micro); // number of valid columns in the micro-tile
                            
                            // Register vector accumulator (per micro-tile)
                            T sum[nr*mr]; 
                            for(index_t idx=0; idx<nr*mr; idx++) sum[idx] = T{}; // set to zero basically

                            // Unroll the k-loop over the micro-tile
                            index_t p=pc;
                            const index_t pend4 = pc + ((pend - pc)/4)*4;
                            for(; p<pend4; p+=4){
                                const T* Bp0 = B.at(p,j0_micro);  //contiguous in memory across j
                                const T* Bp1 = B.at(p+1,j0_micro);
                                const T* Bp2 = B.at(p+2,j0_micro);
                                const T* Bp3 = B.at(p+3,j0_micro);

                                for(index_t i=0; i<i_valid; i++){
                                    // broadcast A(i+i0_micro,p)
                                    const T a0 = A(i0_micro + i,p); 
                                    const T a1 = A(i0_micro + i,p+1); 
                                    const T a2 = A(i0_micro + i,p+2); 
                                    const T a3 = A(i0_micro + i,p+3); 
                                    
                                    for(index_t j=0; j<j_valid; j++){
                                        T acc_ij{};
                                        acc_ij = std::fma(a0, Bp0[j], acc_ij);
                                        acc_ij = std::fma(a1, Bp1[j], acc_ij);
                                        acc_ij = std::fma(a2, Bp2[j], acc_ij);
                                        acc_ij = std::fma(a3, Bp3[j], acc_ij);

                                        sum[j+mr*i] +=  acc_ij;
                                    }
                                }
                            }

                            // loop over the remainder number of elements in K-block i.e. pend % 4
                            p = pend4;
                            for(; p<pend; p++){
                                const T* Bp = B.at(p,j0_micro);
                                for(index_t i=0; i<i_valid; i++){
                                    T a = A(i0_micro + i,p);
                                    for(index_t j=0; j<j_valid; j++){
                                        sum[j+mr*i] =  std::fma(a, Bp[j], sum[j+mr*i]);
                                    }
                                }
                            }

                            // Store the micro-tile back to C
                            for(index_t i=0; i<i_valid; i++){
                                for(index_t j=0; j<j_valid; j++){
                                    C(i0_micro + i, j0_micro + j) += alpha * sum[j+mr*i];
                                }
                            }

                        }
                    }
                }                    
           }     
        }   
    };

    mx_detail::launch_threads(numThreads, workFunction);

}

}
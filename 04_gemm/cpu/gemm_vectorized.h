#pragma once
#include<cassert>
#include<thread>
#include<cmath>
// #include"gemm_parallel.h"
#include"mx/dense.h"
#include"mx/dense_view.h"

#include <experimental/simd>

namespace stdx = std::experimental;

namespace mx{

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_vectorized(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C, index_t numThreads)
{
    gemm_cpu_threads_vectorized(A.view(), B.view(), C.view(), numThreads);
}


template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_vectorized(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C, index_t numThreads = 8)
{
    static_assert(DenseView<const T, Layout>::is_row_major,
                  "gemm_cpu_threads_vectorized() currently supports RowMajor only");

    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

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
                            
                            // SIMD vector of elements of type T with width W corresponding to the
                            // native vector register width for the current target architecture
                            using vT = stdx::native_simd<T>;
                            using mT = typename vT::mask_type;
                            mT mask(false); // default all lanes to false
                            const index_t W = vT::size();
                            
                            // vectorized loop over the 'j_valid/W' full J-blocks of size 'W' 
                            index_t j=0;
                            for(; j < j_valid; j+=W){
                                const index_t j0_simd = j + j0_micro;
                                vT acc[nr];
                                for(index_t i=0; i<nr; i++) acc[i] = vT{};
                                for(index_t jv=0; jv < std::min(W, j_valid-j); jv++) mask[jv] = true; // set the valid lanes in the mask

                                // Unroll the K-loop by 4 
                                index_t p=pc;
                                const index_t pend4 = pc + ((pend - pc)/4)*4;
                                for(; p<pend4; p+=4){
                                    vT vB0{}, vB1{}, vB2{}, vB3{};
                                    stdx::where(mask, vB0).copy_from(&B(p, j0_simd), stdx::element_aligned);
                                    stdx::where(mask, vB1).copy_from(&B(p+1, j0_simd), stdx::element_aligned);
                                    stdx::where(mask, vB2).copy_from(&B(p+2, j0_simd), stdx::element_aligned);
                                    stdx::where(mask, vB3).copy_from(&B(p+3, j0_simd), stdx::element_aligned);

                               
                                    for(index_t i=0; i<i_valid; i++){
                                        vT vA0(A(i0_micro + i,p)); // all elemnts in the simd vecor are equal to A(i0_micro + i,p) and broadcasted
                                        vT vA1(A(i0_micro + i,p+1));
                                        vT vA2(A(i0_micro + i,p+2));
                                        vT vA3(A(i0_micro + i,p+3));
                                    
                                        stdx::where(mask, acc[i]) = acc[i] + vA0 * vB0; //all invalid lanes remain unchanged
                                        stdx::where(mask, acc[i]) = acc[i] + vA1 * vB1;
                                        stdx::where(mask, acc[i]) = acc[i] + vA2 * vB2;
                                        stdx::where(mask, acc[i]) = acc[i] + vA3 * vB3;
                                    }
                                }

                                // K-tail (0to 3 leftover)
                                p = pend4;
                                for(; p<pend; p++){
                                    vT vB{};
                                    std:where(mask, vB).copy_from(&B(p, j0_simd), stdx::element_aligned);
                                    for(index_t i=0; i<i_valid; i++){
                                        vT vA(A(i0_micro + i,p));
                                         stdx::where(mask, acc[i]) = acc[i] + vA * vB;
                                    }
                                }

                                // Store back to C lane by lane
                                for(index_t i=0; i<i_valid; i++){
                                    // load C into a masked vector, add, then store back (also masked)
                                    vT vC{};
                                    stdx::where(mask, vC).copy_from(&C(i0_micro + i, j0_simd), stdx::element_aligned);
                                    vC += acc[i];
                                    stdx::where(mask, vC).copy_to(&C(i0_micro + i, j0_simd), stdx::element_aligned);
                                }
                            }

                        } //M_micro
                    }     //N_micro

                }//Ni                  
            }    //Ki
        }        //Mi
    };

    mx_detail::launch_threads(numThreads, workFunction);

}

}
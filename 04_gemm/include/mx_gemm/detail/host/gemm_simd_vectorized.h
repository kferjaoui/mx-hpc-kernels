#pragma once
#include<stdexcept>
#include <experimental/simd>

#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/transpose.h"
#include "mx/scale_matrix.h"
#include "mx/utils/parallel.h"
#include "mx/utils/schedulers.h"

namespace mx{

namespace stdx = std::experimental;

template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_vectorized(const T alpha, 
                    const Dense<T, Layout>& A,
                    const Dense<T, Layout>& B,
                    const T beta, Dense<T, Layout>& C,
                    index_t nthreads = 1,
                    Scheduler sched = Scheduler{})
{
    gemm_vectorized(alpha, A.view(), B.view(), beta, C.view(), nthreads, sched);
}


template<typename T, class Layout = RowMajor, class Scheduler = BlockScheduler>
void gemm_vectorized(const T alpha, 
                      DenseView<const T, Layout> A,
                      DenseView<const T, Layout> B,
                      const T beta, DenseView<T, Layout> C,
                      index_t nthreads = 1,
                      Scheduler sched = Scheduler{})
{
    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    if (K != B.rows() || N != C.rows() || M != C.cols()) {
        throw std::runtime_error("Mismatch in matrix sizes; cannot perform multiplication.");
    }
    if (N == 0 || M == 0 || K == 0) return;

    scale_matrix(beta, C);
    if (alpha == T{0}) return;

    // Tiling parameters
    const index_t nc = 72; // N tile
    const index_t kc = 256; // K tile
    const index_t mc = 256; // M tile

    // Number of tiles along each dimension (for parallelization)
    const index_t Nb = (N + nc - 1) / nc;

    if (nthreads < 1) nthreads = 1;
    if (nthreads > Nb) nthreads = (Nb > 0 ? Nb : 1);

    using AView = DenseView<const T, Layout>;

    if constexpr (AView::is_row_major) {

        constexpr index_t nr = 6; // Number of rows in A micro-panel
        constexpr index_t mr = 8; // Number of cols in B micro-panel

        auto mm_microkernel_simd = [&](index_t tid){
            // Parallelize over row tiles of C.
            // Outer tile order: I-K-J. Inner order: i-j-k.
            sched(tid, nthreads, Nb,
                [&](index_t Ni){
                    const index_t i0 = Ni*nc;
                    const index_t i_end = std::min<index_t>(N, i0 + nc);

                    for (index_t k0 = 0; k0 < K; k0 += kc) {
                        const index_t k_end = std::min<index_t>(K, k0 + kc);

                        for (index_t j0 = 0; j0 < M; j0 += mc) {
                            const index_t j_end = std::min<index_t>(M, j0 + mc);

                            // Tile-tile multiply with innermost microtile-microtile order k,i,j                  
                            for (index_t i = i0; i < i_end; i+=nr) { // Sweep i down rows of A tile; i values correspond to global indices of `micro-tile starting row`
                                index_t i_micro_end = std::min(i+nr, i_end);
                                
                                // *****************************************
                                // SIMD vector of elements of type T with width W corresponding to the
                                // native vector register width for the current target architecture
                                using vT = stdx::native_simd<T>;
                                using mT = typename vT::mask_type;

                                constexpr index_t W = vT::size(); // for AVX512, W=8 for double precision, W=16 for single precision

                                for (index_t j = j0; j < j_end; j+=mr) { // Cols of B tile; j values correspond to global indices of `micro-tile starting column`
                                    index_t j_micro_end = std::min(j+mr, j_end);
                                    
                                    index_t nr_actual = std::min(nr, i_micro_end - i); // number of valid rows in the A micro-panel
                                    index_t mr_actual = std::min(mr, j_micro_end - j); // number of valid cols in the B micro-panel
                                
                                    // number of SIMD vectors needed to cover mr columns
                                    constexpr index_t mr_simd = (mr + W - 1) / W; 
                                    vT acc[nr*mr_simd]{0}; 
                                    
                                    // default all lanes to false
                                    mT mask[mr_simd]{};  
                                    
                                    // set the valid lanes in the mask
                                    for (index_t idx_simd=0; idx_simd < mr_simd; ++idx_simd){
                                        
                                        const index_t offset = idx_simd * W;
                                        const index_t lanes_valid = (offset < mr_actual) ? std::min<index_t>(W, mr_actual - offset) : 0;

                                        for(index_t jv=0; jv < lanes_valid; ++jv){
                                            mask[idx_simd][jv] = true;
                                        }
                                    }

                                    for (index_t k = k0; k < k_end; ++k) {
                                        for(index_t j_simd = 0; j_simd < mr_simd; ++j_simd){
                                            // create a simd vector for the contiguous elements of B
                                            vT b_vec{}; // one simd load, reused for all rows `il` in the A micro-panel

                                            stdx::where(mask[j_simd], b_vec).copy_from(&B(k, j + j_simd*W), stdx::element_aligned);

                                            for (index_t il = 0; il < nr_actual; ++il) { // `l` for local micro-tile indices
                                                const T a_val = A(i + il, k);
                                                acc[il*mr_simd + j_simd] +=  vT(a_val) * b_vec;                                   
                                            }
                                        }
                                    }

                                    // Write-back: loop order matches C's memory layout for contiguous stores
                                    for (index_t ig = i; ig < i_micro_end; ++ig) { // `g` for global tile indices
                                        for(index_t j_simd = 0; j_simd < mr_simd; ++j_simd){
                                            vT c_vec{};
                                            stdx::where(mask[j_simd], c_vec).copy_from(&C(ig, j + j_simd*W), stdx::element_aligned);  // load
                                            c_vec += vT(alpha) * acc[(ig - i)*mr_simd + j_simd];                                          // update
                                            stdx::where(mask[j_simd], c_vec).copy_to(&C(ig, j + j_simd*W), stdx::element_aligned);    // store
                                        }
                                    }
                                    // *****************************************
                                }
                            }
                        }
                    }
                }
            );
        };

        if (nthreads == 1) mm_microkernel_simd(0);
        else launch_threads(mm_microkernel_simd, nthreads);

    } else {// Col-major

        constexpr index_t nr = 8; // Number of rows in A micro-panel
        constexpr index_t mr = 6; // Number of cols in B micro-panel
        
        auto mm_microkernel_simd = [&](index_t tid){
            // Parallelize over row tiles of C.
            // Outer tile order: I-K-J. Inner order: i-j-k.
            sched(tid, nthreads, Nb,
                [&](index_t Ni){
                    const index_t i0 = Ni*nc;
                    const index_t i_end = std::min<index_t>(N, i0 + nc);

                    for (index_t k0 = 0; k0 < K; k0 += kc) {
                        const index_t k_end = std::min<index_t>(K, k0 + kc);

                        for (index_t j0 = 0; j0 < M; j0 += mc) {
                            const index_t j_end = std::min<index_t>(M, j0 + mc);

                            // Tile-tile multiply with innermost microtile-microtile order k,i,j                  
                            for (index_t j = j0; j < j_end; j+=mr) { // Sweep i down rows of A tile; i values correspond to global indices of `micro-tile starting row`
                                index_t j_micro_end = std::min(j+mr, j_end);              
                        
                                // ***** SIMD ******************************
                                using vT = stdx::native_simd<T>;
                                using mT = typename vT::mask_type;

                                constexpr index_t W = vT::size(); // for AVX512, W=8 for double precision, W=16 for single precision

                                for (index_t i = i0; i < i_end; i+=nr) { // Sweep i down rows of A tile; i values correspond to global indices of `micro-tile starting row`
                                    index_t i_micro_end = std::min(i+nr, i_end);
                                    
                                    index_t mr_actual = std::min(mr, j_micro_end - j); // number of valid cols in the B micro-panel
                                    index_t nr_actual = std::min(nr, i_micro_end - i); // number of valid rows in the A micro-panel
                                
                                    // number of SIMD vectors needed to cover nr rows
                                    constexpr index_t nr_simd = (nr + W - 1) / W; 
                                    vT acc[nr_simd*mr]{0}; 
                                    
                                    // default all lanes to false
                                    mT mask[nr_simd]{};  
                                    
                                    // set the valid lanes in the mask
                                    for (index_t idx_simd=0; idx_simd < nr_simd; ++idx_simd){

                                        const index_t offset = idx_simd * W;
                                        const index_t lanes_valid = (offset < nr_actual) ? std::min<index_t>(W, nr_actual - offset) : 0;

                                        for(index_t iv=0; iv < lanes_valid; ++iv){
                                            mask[idx_simd][iv] = true;
                                        }
                                    }

                                    for (index_t k = k0; k < k_end; ++k) {
                                        for(index_t i_simd = 0; i_simd < nr_simd; ++i_simd){
                                            vT a_vec{};
                                            stdx::where(mask[i_simd], a_vec).copy_from(&A(i + i_simd*W, k), stdx::element_aligned);

                                            for (index_t jl = 0; jl < mr_actual; ++jl) {
                                                const T b_val = B(k, j + jl);
                                                acc[jl * nr_simd + i_simd] +=  vT(b_val) * a_vec; 
                                            }
                                        }
                                    }

                                    // Write-back: loop order matches C's memory layout for contiguous stores
                                    for (index_t jg = j; jg < j_micro_end; ++jg) {
                                        for(index_t i_simd = 0; i_simd < nr_simd; ++i_simd){
                                            vT c_vec{};
                                            stdx::where(mask[i_simd], c_vec).copy_from(&C(i + i_simd*W, jg), stdx::element_aligned);  // load
                                            c_vec += vT(alpha) * acc[(jg - j)*nr_simd + i_simd];                                      // update
                                            stdx::where(mask[i_simd], c_vec).copy_to(&C(i + i_simd*W, jg), stdx::element_aligned);    // store
                                        }
                                    }
                                    // *****************************************
                                }
                            }
                        }
                    }
                }
            );
        };

        if (nthreads == 1) mm_microkernel_simd(0);
        else launch_threads(mm_microkernel_simd, nthreads);
    }

}


} // namespace mx


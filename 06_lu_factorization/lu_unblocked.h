#pragma once
#include "mx/types.h"
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "pivot.h"
#include <algorithm>
#include <cmath>

namespace mx{

template<typename T, class Layout>
[[nodiscard]] constexpr 
LUInfo lu_factor_unblocked(DenseView<T, Layout> LU, std::vector<index_t>& piv){
    const index_t N = LU.rows();
    const index_t M = LU.cols(); 
    const index_t K = std::min(N, M);

    // assert(piv.size() >= K && "pivot vector too small");
    if (piv.size() < K){
        std::cout << "Warning: pivot vector resized from " << piv.size() << " to " << K << "\n";
        piv.resize(K);
    }

    LUInfo info;

    for(index_t k = 0; k<K; k++){

        // 1. find the pivot
        index_t i_pivot = k;
        auto pivot = LU(k,k);
        for(index_t ii=k+1; ii<N; ii++){
            if(std::abs(LU(ii,k)) > std::abs(pivot)){
                i_pivot = ii;
                pivot = LU(ii,k);
            }
        }
        piv[k] = i_pivot;

        // 2. swap full rows
        if (i_pivot != k) swap_full_row(LU, k, i_pivot);

        // 3. guard against zero; TODO: near-zero pivot
        const T akk = LU(k,k);
        if (akk == T(0)) {
            info.status = LUStatus::SINGULAR;  // Singular matrix at column k
            info.first_zero_pivot = k;
            continue;
        }

        // 4. save the multipliers
        for(index_t i=k+1; i<N; i++){
            LU(i,k) = LU(i,k) / akk;
        }

        // 5. update the trailing submatrix
        for(index_t i=k+1; i<N; i++){
            const T l_ik = LU(i,k);
            if (l_ik == T(0)) continue;
            for(index_t j = k+1; j<M; j++) LU(i,j) = LU(i,j) - l_ik * LU(k,j);
        }

    }

    return info;

}

}
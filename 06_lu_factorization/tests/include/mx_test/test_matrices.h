#pragma once

#include "mx/dense.h"
#include <cmath>

namespace mx {

enum class GramPattern {
    TrigSmooth,   // sin/cos pattern (deterministic, smooth)
    Poly,         // simple polynomial in (i,j)
    IndexBased    // cheap, integer-based pattern
};

template<typename T = double, class Layout = RowMajor>
Dense<T, Layout> make_spd_gram_matrix(index_t N,
                                      index_t K,
                                      T alpha = T(1e-1),
                                      GramPattern pattern = GramPattern::TrigSmooth)
{
    // Safety: clamp K to [1, N]
    if (K < 1) K = 1;
    if (K > N) K = N;

    // M is K x N
    Dense<T, Layout> M(K, N);

    for (index_t k = 0; k < K; ++k) {
        for (index_t j = 0; j < N; ++j) {
            T val{};
            switch (pattern) {
                case GramPattern::TrigSmooth: {
                    T x = T(0.01) * T(k + 1) * T(j + 3);
                    T y = T(0.02) * T(k + 2) * T(j + 1);
                    val = std::sin(x) + std::cos(y);
                    break;
                }
                case GramPattern::Poly: {
                    // Simple polynomial in k,j
                    T kk = T(k + 1);
                    T jj = T(j + 1);
                    val = kk + T(0.1) * jj + T(0.001) * kk * jj;
                    break;
                }
                case GramPattern::IndexBased: {
                    // Cheap integer-based pattern
                    // (wraps but deterministic)
                    int base = int((k + 1) * 131 + (j + 3) * 17);
                    val = T((base % 100) - 50) / T(10); // in [-5, 5)
                    break;
                }
            }
            M(k, j) = val;
        }
    }

    // A = M^T * M  (N x N)
    Dense<T, Layout> A(N, N, T(0));

    for (index_t i = 0; i < N; ++i) {
        for (index_t j = 0; j < N; ++j) {
            T acc = T(0);
            for (index_t k = 0; k < K; ++k) {
                acc += M(k, i) * M(k, j);
            }
            A(i, j) = acc;
        }
    }

    // Shift by alpha * I to ensure SPD
    for (index_t i = 0; i < N; ++i) {
        A(i, i) += alpha;
    }

    return A;
}

} // namespace mx

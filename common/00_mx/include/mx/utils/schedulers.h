#pragma once
#include "mx/types.h"

namespace mx {

struct CyclicScheduler {
    template<class F>
    void operator()(index_t tid, index_t numThreads, index_t N, F&& body) const {
        for (index_t i = tid; i < N; i += numThreads) {
            body(i);
        }
    }
};

struct BlockScheduler {
    template<class F>
    void operator()(index_t tid, index_t numThreads, index_t N, F&& body) const {
        const index_t baseWork  = N / numThreads;
        const index_t remainder = N % numThreads;

        const index_t workChunk = baseWork + (tid < remainder ? 1 : 0);
        const index_t start     = tid * baseWork + ( tid < remainder ? tid : remainder);
        const index_t end       = start + workChunk;

        for (index_t i = start; i < end; ++i) {
            body(i);
        }
    }
};

} // namespace mx
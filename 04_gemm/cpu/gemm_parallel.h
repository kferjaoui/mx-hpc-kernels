#pragma once
#include<cassert>
#include<thread>
#include<cmath>
#include"mx/dense.h"
#include"mx/dense_view.h"

namespace mx{

namespace mx_detail{

template <class Worker>
void launch_threads(index_t numThreads, Worker&& worker){
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for(index_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(worker, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }

}

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
        auto baseWork  = N / numThreads;
        auto remainder = N % numThreads;

        auto workChunk = baseWork + (tid < remainder ? 1 : 0);
        auto start     = tid * baseWork + std::min(tid, remainder);
        auto end       = start + workChunk;

        for (index_t i = start; i < end; ++i) {
            body(i);
        }
    }
};

template<typename T, class Scheduler, class Layout = RowMajor>
void gemm_cpu_threads_impl(DenseView<const T, Layout> A,
                           DenseView<const T, Layout> B,
                           DenseView<T, Layout>       C,
                           index_t            numThreads,
                           Scheduler          sched) 
{
    static_assert(DenseView<const T, Layout>::is_row_major,
                  "gemm_cpu_threads_cyclic() and gemm_cpu_threads_block() currently support RowMajor only");

    const index_t N = A.rows();
    const index_t K = A.cols();
    const index_t M = B.cols();

    assert(K == B.rows() && N == C.rows() && M == C.cols());
    if (N == 0 || M == 0 || K == 0) return;

    numThreads = numThreads ? std::min(numThreads, N) : 1;

    // Materialize the transposed of B for better locality
    Dense<T, Layout> BT(M, K);
    for (index_t r = 0; r < K; ++r) {
        for (index_t c = 0; c < M; ++c) {
            BT(c, r) = B(r, c);
        }
    }

    auto worker = [&, N, M, K, numThreads](index_t tid) {
        sched(tid, numThreads, N, [&](index_t i) {
            for (index_t j = 0; j < M; ++j) {
                T sum{};
                for (index_t k = 0; k < K; ++k) {
                    sum += A(i, k) * BT(j, k);
                }
                C(i, j) = sum;
            }
        });
    };

    launch_threads(numThreads, worker);
}

} // mx_detail

// Public APIs:

// Row-Strided Parallel pattern
template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_cyclic(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C, index_t numThreads = 8)
{
    gemm_cpu_threads_cyclic(A.view(), B.view(), C.view(), numThreads);
}

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_cyclic(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C, index_t numThreads = 8)
{
    gemm_cpu_threads_impl(A, B, C, numThreads, mx_detail::CyclicScheduler{});
}

// Row-Blocked parallel pattern
template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_block(const Dense<T, Layout>& A, const Dense<T, Layout>& B, Dense<T, Layout>& C, index_t numThreads = 8)
{
    gemm_cpu_threads_block(A.view(), B.view(), C.view(), numThreads);
}

template<typename T, class Layout = RowMajor>
void gemm_cpu_threads_block(DenseView<const T, Layout> A, DenseView<const T, Layout> B, DenseView<T, Layout> C, index_t numThreads = 8)
{
    gemm_cpu_threads_impl(A, B, C, numThreads, mx_detail::BlockScheduler{});
}

} // mx
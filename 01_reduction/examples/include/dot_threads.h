#include<thread>
#include<barrier>
#include<vector>
#include<cmath>

void dotThreadWorker(int threadIdx,
                    const double* x, 
                    const double* y, 
                    size_t numThreads, 
                    size_t n,
                    std::barrier<>& sync_point,
                    std::vector<double>& vResult){

    for(size_t idx=threadIdx; idx<n; idx+=numThreads){
        vResult[threadIdx] += x[idx]*y[idx];
    }

    sync_point.arrive_and_wait();

    for(size_t stride=numThreads>>1; stride>0; stride>>=1){
        if (threadIdx < stride) {
            vResult[threadIdx] += vResult[threadIdx+stride]; 
        }
        sync_point.arrive_and_wait();
    }
}

double dotThreads(const double* hx, const double* hy, size_t n){

    size_t numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback
    std::vector<std::thread> T;
    T.reserve(numThreads);
    std::barrier<> sync_point(numThreads);
    std::vector<double> vResult(numThreads, 0.0);

    for (size_t idx=0; idx<numThreads; idx++) T.emplace_back(dotThreadWorker, idx,
                                                                            hx, 
                                                                            hy, 
                                                                            numThreads, 
                                                                            n, 
                                                                            std::ref(sync_point), 
                                                                            std::ref(vResult));

    for(auto& t: T) t.join();

    return vResult[0];

}
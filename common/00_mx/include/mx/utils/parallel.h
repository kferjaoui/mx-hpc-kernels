#pragma once
#include <vector>
#include <thread>
#include "mx/types.h"

namespace mx{

template <class Worker>
void launch_threads(Worker&& worker, index_t numThreads){
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for(index_t tid=0; tid<numThreads; tid++){
        threads.emplace_back(worker, tid);
    }

    for(auto& t:threads){
        if(t.joinable()) t.join();
    }

}

} // mx
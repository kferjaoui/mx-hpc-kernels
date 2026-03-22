#pragma once
#include <cstdint>
#include <type_traits>
#include "mx/utils/schedulers.h"

namespace mx{

template <class Scheduler = BlockScheduler>
struct CPU {
    int threads = 1;
    Scheduler scheduler = Scheduler{};
};

struct CUDA {
    std::uint32_t block = 256;
    std::uint32_t grid_x = 2048;
    std::uint32_t grid_y = 1;
    std::uint32_t grid_z = 1;
    std::uintptr_t stream = 0; // 0 == default stream

    bool debug_sync  = false;  // sync after each launch
    bool debug_print  = false; // enable prints in debug mode
};


// policy traits

template<class Policy>
struct is_cpu_policy : std::false_type {};

template<class Scheduler>
struct is_cpu_policy<CPU<Scheduler>> : std::true_type {};

template<class Policy>
inline constexpr bool is_cpu_policy_v =
    is_cpu_policy<std::remove_cvref_t<Policy>>::value;

template<class Policy>
struct is_cuda_policy : std::false_type {};

template<>
struct is_cuda_policy<CUDA> : std::true_type {};

template<class Policy>
inline constexpr bool is_cuda_policy_v =
    is_cuda_policy<std::remove_cvref_t<Policy>>::value;

}
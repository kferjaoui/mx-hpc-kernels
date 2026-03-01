#pragma once

namespace mx::detail {

enum class GemmAlgorithm {
    Cyclic,
    Blocked,
    CacheBlocked,
    Microtiles,
    Vectorized
};

}
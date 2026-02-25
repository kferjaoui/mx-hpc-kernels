#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mx {

// next power of two >= x, for unsigned integer types
template <typename UInt>
constexpr UInt next_pow2(UInt x) {
    static_assert(std::is_unsigned_v<UInt>, "next_pow2 requires an unsigned integer type");

    if (x <= 1) return 1;

    // If x is already a power of two, return it
    if ((x & (x - 1)) == 0) return x;

    // Round up by filling lower bits then adding 1
    UInt v = x - 1;
    for (std::size_t shift = 1; shift < sizeof(UInt) * 8; shift <<= 1) {
        v |= (v >> shift);
    }

    // Overflow check: if v was all 1s, v+1 wraps to 0
    if (v == std::numeric_limits<UInt>::max()) {
        return 0; // signal overflow
    }
    return v + 1;
}

// convenience overload for signed ints (clamps negatives)
constexpr int next_pow2(int x) {
    if (x <= 1) return 1;
    auto u = static_cast<std::uint32_t>(x);
    auto r = next_pow2<std::uint32_t>(u);
    return (r == 0) ? 0 : static_cast<int>(r); // 0 means overflow
}

} // namespace mx
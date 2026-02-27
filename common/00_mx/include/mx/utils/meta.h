#pragma once
#include <concepts>

namespace mx {

template <class>
inline constexpr bool always_false_v = false;

template<typename Op, typename T>
concept BinaryOp = requires(Op op, T a, T b) {
    { op(a, b) } -> std::convertible_to<T>;
    { Op::identity() } -> std::convertible_to<T>;
};

} // namespace mx
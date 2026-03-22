#pragma once
#include <concepts>

namespace mx {

template <class>
inline constexpr bool always_false_v = false;

template <auto>
inline constexpr bool always_false_value_v = false;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

template<typename T>
concept Multipliable = requires(T a, T b) {
    { a * b } -> std::same_as<T>;
};

template<typename T>
concept Comparable = requires(T a, T b) {
    { a > b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
};

template<typename Op, typename T>
concept BinaryOp = requires(Op op, T a, T b) {
    { op(a, b) } -> std::same_as<T>;
    { Op::identity() } -> std::same_as<T>;
    requires noexcept(Op::identity());  // should be trivially evaluated at compile-time i.e. constexpr  
};

} // namespace mx
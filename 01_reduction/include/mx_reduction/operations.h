#pragma once

namespace mx{

template <typename T>
struct Sum{
    T operator()(T a, T b) { return a + b; }
    T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct Multiply{
    T operator()(T a, T b) { return a * b; }
    T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct Max{
    T operator()(T a, T b) { return (a > b) ? a : b; }
    T operator()(T a, T b) const { return (a > b) ? a : b; }
};

template <typename T>
struct Min{
    T operator()(T a, T b) { return (a < b) ? a : b; }
    T operator()(T a, T b) const { return (a < b) ? a : b; }
};

}
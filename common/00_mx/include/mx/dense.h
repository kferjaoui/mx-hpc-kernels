#pragma once
#include <vector>
#include <initializer_list>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "dense_view.h"
#include "types.h"
#include "layout.h"

#include <Eigen/Dense>

namespace mx {

template<typename T, class Layout = RowMajor>
class Dense {
    index_t _rows{0};
    index_t _cols{0};
    index_t _size{0}; 
    std::vector<T> _data;

    static constexpr bool row_major = std::is_same_v<Layout, RowMajor>;

    [[nodiscard]] index_t idx(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        if constexpr (row_major) {
            return j + _cols * i;
        } else { // ColMajor
            return i + _rows * j;
        }
    }

public:
    Dense() = default;
    
    Dense(index_t rows, index_t cols):
        _rows(rows), _cols(cols), _size(rows*cols), _data(rows*cols) {
        assert(rows >= 0 && cols >= 0);
    }
    
    Dense(index_t rows, index_t cols, const T& init):
        _rows(rows), _cols(cols), _size(rows*cols), _data(rows*cols, init) {
        assert(rows >= 0 && cols >= 0);
    }

    Dense(index_t rows, index_t cols, std::initializer_list<T> init):
        _rows(rows), _cols(cols), _size(rows*cols), _data(init) {
        assert(rows >= 0 && cols >= 0);
        assert(static_cast<index_t>(init.size()) == rows * cols && 
               "Initializer size must match rows*cols");
    }

    [[nodiscard]] T& operator()(index_t i, index_t j) noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _data[idx(i,j)];
    }
    
    [[nodiscard]] const T& operator()(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _data[idx(i,j)];
    }

    bool operator==(const Dense& other) const noexcept {
        if(_size != other._size) return false;

        for(index_t idx = 0; idx < _size; idx++) {
            if(_data[idx] != other._data[idx]) return false;
        }

        return true;
    }

    [[nodiscard]] index_t rows() const noexcept { return _rows; }
    [[nodiscard]] index_t cols() const noexcept { return _cols; }
    [[nodiscard]] index_t size() const noexcept { return _size; }

    T*       data() noexcept       { return _data.data(); }
    const T* data() const noexcept { return _data.data(); }

    // Expose row-major view
    DenseView<T,Layout>       view() noexcept       { return DenseView<T,Layout>(_data.data(), _rows, _cols); }
    DenseView<const T,Layout> view() const noexcept { return DenseView<const T,Layout>(_data.data(), _rows, _cols); }

    // Iterator support
    T*       begin() noexcept { return _data.data(); }
    const T* begin() const noexcept { return _data.data(); }
    
    T*       end() noexcept { return _data.data() + _size; }
    const T* end() const noexcept { return _data.data() + _size; }

    // Element pointer access
    T*       at(index_t i, index_t j) noexcept       { return _data.data() + idx(i,j); }
    const T* at(index_t i, index_t j) const noexcept { return _data.data() + idx(i,j); }

    // Fill the dense matrix with a specific value
    void fill(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    // MX -> Eigen conversion
    auto to_eigen() const {
        using MatType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, row_major ? Eigen::RowMajor : Eigen::ColMajor>;
        MatType eigenMatrix(_rows, _cols);
        std::copy(_data.begin(), _data.end(), eigenMatrix.data());
        return eigenMatrix;
    }

};

}
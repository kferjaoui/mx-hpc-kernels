#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "types.h"
#include "layout.h"

#include <Eigen/Dense>
namespace mx {
    
template<typename T, class Layout = RowMajor>
class DenseView {
    index_t _rows{0};
    index_t _cols{0};
    index_t _size{0};
    std::array<index_t, 2> _strides{0, 0};
    T* _buffer{nullptr};

    static constexpr bool row_major = std::is_same_v<Layout, RowMajor>;
    using OppositeLayout = std::conditional_t<row_major, ColMajor, RowMajor>;

    [[nodiscard]] index_t idx(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _strides[0] * i + _strides[1] * j;
    }

    // Put this constructor as private to avoid misuse (used internally only)
    DenseView(T* ptr, index_t rows, index_t cols, index_t row_stride, index_t col_stride): 
        _buffer(ptr), _rows(rows), _cols(cols), _size(rows*cols), _strides{row_stride, col_stride} 
    {
        assert(rows >= 0 && cols >= 0);
        assert(row_stride >= 0 && col_stride >= 0);
        assert(ptr != nullptr || (rows == 0 && cols == 0));
    }

public:
    DenseView() = default;
    DenseView(const DenseView& other) = default; // default copy constructor

    // Contiguous  
    DenseView(T* ptr, index_t rows, index_t cols): 
        _buffer(ptr), _rows(rows), _cols(cols), _size(rows*cols)
    {
        assert(rows >= 0 && cols >= 0);
        assert(ptr != nullptr || (rows == 0 && cols == 0));
        if constexpr (row_major) {
            _strides = {cols, 1};
        } else {
            _strides = {1, rows};
        }
    }

    // To allow conversion of DenseView<T> to DenseView<const T>
    template<typename U>
    requires std::convertible_to<U*, T*> && (!std::same_as<U, const U>)
    DenseView(const DenseView<U, Layout>& other)
        : _rows(other.rows()), 
          _cols(other.cols()),
          _size(other.size()),
          _strides{other.row_stride(), other.col_stride()},
          _buffer(other.data())
    {}

    T& operator()(index_t i, index_t j) noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer[idx(i,j)];
    }

    const T& operator()(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer[idx(i,j)];
    }

    // Public compile-time constant
    static constexpr bool is_row_major = row_major;
    static constexpr bool is_col_major = !row_major;

    // Create subview maintaining parent strides
    [[nodiscard]] DenseView<const T, Layout> subview(index_t i0, index_t j0, index_t n_rows, index_t n_cols) const noexcept {
        assert(i0 >= 0 && j0 >= 0);
        assert(i0 + n_rows <= _rows && j0 + n_cols <= _cols);
        return DenseView(_buffer + idx(i0,j0), n_rows, n_cols, _strides[0], _strides[1]); 
    }
    
    // Transposed view (swap dimensions and strides)
    [[nodiscard]] DenseView<T, Layout> transpose_same_layout() noexcept { 
        return DenseView(_buffer, _cols, _rows, _strides[1], _strides[0]); 
    }

    [[nodiscard]] DenseView<const T, Layout> transpose_same_layout() const noexcept { 
        return DenseView(_buffer, _cols, _rows, _strides[1], _strides[0]); 
    }

    [[nodiscard]] DenseView<T, OppositeLayout> transpose() noexcept {
        assert(is_contiguous() && "Transpose requires contiguous view");
        return DenseView<T, OppositeLayout>(_buffer, _cols, _rows);
    }

    [[nodiscard]] DenseView<const T, OppositeLayout> transpose() const noexcept {
        assert(is_contiguous() && "Transpose requires contiguous view");
        return DenseView<const T, OppositeLayout>(_buffer, _cols, _rows);
    }

    // Accessors
    [[nodiscard]] index_t rows() const noexcept { return _rows; }
    [[nodiscard]] index_t cols() const noexcept { return _cols; }
    [[nodiscard]] index_t size() const noexcept { return _size; }
    [[nodiscard]] index_t row_stride() const noexcept { return _strides[0]; }
    [[nodiscard]] index_t col_stride() const noexcept { return _strides[1]; }
    
    // Leading dimension (BLAS style)
    [[nodiscard]] index_t leading_dim() const noexcept {
        if constexpr (row_major) {
            return _strides[0]; // For row-major: LDA = row_stride
        } else {
            return _strides[1]; // For col-major: LDA = col_stride
        }
    }

    // Direct buffer access
    T*       data() noexcept       { return _buffer; }
    const T* data() const noexcept { return _buffer; }

    // Iterator support (only valid for contiguous views)
    T*       begin() noexcept { return _buffer; }
    const T* begin() const noexcept { return _buffer; }
    
    T*       end() noexcept { return _buffer + _size; }
    const T* end() const noexcept { return _buffer + _size; }

    // Element pointer access
    T* at(index_t i, index_t j) noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer + idx(i,j);
    }
    
    const T* at(index_t i, index_t j) const noexcept {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _buffer + idx(i,j);
    }

    // Check if view is contiguous in memory
    [[nodiscard]] bool is_contiguous() const noexcept {
        if constexpr (row_major) {
            return (_strides[0] == _cols && _strides[1] == 1);
        } else {
            return (_strides[0] == 1 && _strides[1] == _rows);
        }
    }

    // MX -> Eigen conversion
    auto to_eigen() const {
        using MatType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, row_major ? Eigen::RowMajor : Eigen::ColMajor>;
        MatType eigenMatrix(_rows, _cols);

        if(is_contiguous()){
            std::copy(_buffer, _buffer + _size, eigenMatrix.data());
        } else{
            // copy element by element respecting strides
            for(index_t i=0; i<_rows; i++){
                for(index_t j=0; j<_cols; j++){
                    eigenMatrix(i,j) = (*this)(i,j);
                }
            }
        }
        return eigenMatrix;
    }
};

}
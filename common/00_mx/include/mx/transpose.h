#pragma once 
#include "dense.h"
#include "dense_view.h"
#include "layout.h"
#include "types.h"

namespace mx {

template<typename T, class Layout>
void transpose_matrix_tiled(const Dense<T, Layout>& M, Dense<T, Layout>& MT, const index_t TILE_SIZE = 32)
{
    transpose_matrix_tiled(M.view(), MT.view(), TILE_SIZE);
}

template<typename T>
void transpose_matrix_tiled(const DenseView<const T, RowMajor> M, DenseView<T, RowMajor> MT, const index_t TILE_SIZE = 32)
{
    if (M.rows() != MT.cols() || M.cols() != MT.rows())
        throw std::invalid_argument("Matrix dimensions incompatible for transpose");

    for(index_t c0=0; c0<M.cols(); c0+=TILE_SIZE){
        for(index_t r0=0; r0<M.rows(); r0+=TILE_SIZE){
            index_t c0_end = std::min(M.cols(), c0 + TILE_SIZE);
            index_t r0_end = std::min(M.rows(), r0 + TILE_SIZE);
            for(index_t c=c0; c<c0_end; c++){
                for(index_t r=r0; r<r0_end; r++){
                    MT(c,r) = M(r,c);
                }  // r: row indices inside the tile 
            }  // c: column indices inside the tile
        }  // r0: row start index of tile
    }  // c0: column start index of tile
}

template<typename T>
void transpose_matrix_tiled(const DenseView<const T, ColMajor> M, DenseView<T, ColMajor> MT, const index_t TILE_SIZE = 32)
{
    if (M.rows() != MT.cols() || M.cols() != MT.rows())
        throw std::invalid_argument("Matrix dimensions incompatible for transpose");

    for(index_t r0=0; r0<M.rows(); r0+=TILE_SIZE){
        for(index_t c0=0; c0<M.cols(); c0+=TILE_SIZE){
            index_t r0_end = std::min(M.rows(), r0 + TILE_SIZE);
            index_t c0_end = std::min(M.cols(), c0 + TILE_SIZE);
            for(index_t r=r0; r<r0_end; r++){
                for(index_t c=c0; c<c0_end; c++){
                    MT(c,r) = M(r,c);
                } 
            }
        }
    }
}

}
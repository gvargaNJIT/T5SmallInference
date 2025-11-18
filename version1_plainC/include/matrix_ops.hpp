#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include "tensor.hpp" 

namespace t5 {
namespace serial {

void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

void matmul_transposed(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void matmul_transposed(const Tensor& A, const Tensor& B, Tensor& C);
}
}
#endif
#include "matrix_ops.hpp"
#include <stdexcept>

namespace t5 {
namespace serial {

void matmul(const float* A,const float* B,float* C,int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_transposed(const float* A,const float* B,float* C,int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
        throw std::runtime_error("matmul: All tensors must be 2D");
    }
    
    int M = A_shape[0];
    int K = A_shape[1];
    int N = B_shape[1];
    
    if (B_shape[0] != K) {
        throw std::runtime_error("matmul: Dimension mismatch (A.cols != B.rows)");
    }
    
    if (C_shape[0] != M || C_shape[1] != N) {
        throw std::runtime_error("matmul: Output tensor has wrong shape");
    }
    
    matmul(A.data(), B.data(), C.data(), M, K, N);
}

void matmul_transposed(const Tensor& A, const Tensor& B, Tensor& C) {
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
        throw std::runtime_error("matmul_transposed: All tensors must be 2D");
    }
    
    int M = A_shape[0];
    int K = A_shape[1];
    int N = B_shape[0];
    
    if (B_shape[1] != K) {
        throw std::runtime_error("matmul_transposed: Dimension mismatch");
    }
    
    if (C_shape[0] != M || C_shape[1] != N) {
        throw std::runtime_error("matmul_transposed: Output tensor has wrong shape");
    }
    
    matmul_transposed(A.data(), B.data(), C.data(), M, K, N);
}
}
}

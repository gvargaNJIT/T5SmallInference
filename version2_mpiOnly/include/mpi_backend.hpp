#pragma once
#include "tensor.hpp"

namespace mpi_backend {

Tensor matmul(const Tensor& A, const Tensor& B);
Tensor softmax(const Tensor& A);


}

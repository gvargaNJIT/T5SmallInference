#ifndef RMS_NORM_HPP
#define RMS_NORM_HPP

#include "tensor.hpp"

class RMSNorm {
public:
    Tensor weight;
    float eps;
    RMSNorm(int hidden_size, float epsilon = 1e-6f);
    Tensor forward(const Tensor& x);
};

#endif

#ifndef LAYER_NORM_HPP
#define LAYER_NORM_HPP

#include "tensor.hpp"

class LayerNorm {
public:
    Tensor weight;
    float eps;
    LayerNorm(int hidden_size, float epsilon = 1e-6f);
    Tensor forward(const Tensor& x);
};

#endif

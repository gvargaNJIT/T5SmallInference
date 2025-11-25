#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "tensor.hpp"
#include <vector>

class Linear {
public:
    Tensor weight;
    Tensor bias;
    bool use_bias;
    int in_features;
    int out_features;
    Linear(int in_feat, int out_feat, bool use_bias_ = true);
    Tensor forward(const Tensor& x);
};

#endif

#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

class Linear {
public:
    Tensor weight;
    int in_features;
    int out_features;

    Linear(int in_feat, int out_feat);

    Tensor forward(const Tensor& x);
};

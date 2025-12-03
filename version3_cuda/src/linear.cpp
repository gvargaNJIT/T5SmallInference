#include "linear.hpp"
#include <cmath>

Linear::Linear(int in_feat, int out_feat)
    : in_features(in_feat), out_features(out_feat)
{
    weight = Tensor({out_feat, in_feat}, 0.0f);
}

Tensor Linear::forward(const Tensor& x)
{
    Tensor tmp = x;
    tmp.shape.pop_back();

    Tensor x_2d = x.reshape({Tensor::numel(tmp.shape), in_features});

    Tensor result_2d = x_2d.matmul(weight.transpose());

    tmp.shape.push_back(out_features);

    return result_2d.reshape(tmp.shape);
}

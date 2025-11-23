#include "linear.hpp"
#include <cmath>

Linear::Linear(int in_feat, int out_feat, bool use_bias_)
    : use_bias(use_bias_), in_features(in_feat), out_features(out_feat) {
    float std = std::sqrt(2.0f / (in_feat + out_feat));
    weight = Tensor::randn({out_feat, in_feat}, 0.0f, std);
    if (use_bias) {
        bias = Tensor::zeros({out_feat});
    }
}

Tensor Linear::forward(const Tensor& x) {
    std::vector<int> batch_shape;
    for (size_t i = 0; i < x.shape.size() - 1; i++) {
        batch_shape.push_back(x.shape[i]);
    }
    int batch_size = 1;
    for (int dim : batch_shape) batch_size *= dim;
    Tensor x_2d = x.reshape({batch_size, in_features});
    Tensor result_2d;  
    if (weight.shape[0] == in_features && weight.shape[1] == out_features) {
        result_2d = x_2d.matmul(weight);
    } else if (weight.shape[0] == out_features && weight.shape[1] == in_features) {
        result_2d = x_2d.matmul(weight.transpose());
    } else {
        throw std::runtime_error("Linear weight shape mismatch");
    }
    if (use_bias) {
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < out_features; f++) {
                result_2d.data[b * out_features + f] += bias.data[f];
            }
        }
    }
    batch_shape.push_back(out_features);
    return result_2d.reshape(batch_shape);
}

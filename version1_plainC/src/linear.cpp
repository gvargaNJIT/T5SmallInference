#include "linear.hpp"
#include <cmath>
#include <stdexcept>

Linear::Linear(int in_feat, int out_feat, bool use_bias_)
    : use_bias(use_bias_), in_features(in_feat), out_features(out_feat) {
    float std = std::sqrt(2.0f / (in_feat + out_feat));
    weight = Tensor::randn({in_features, out_features}, 0.0f, std);  // note: shape = [in, out] to match PyTorch .T
    if (use_bias) {
        bias = Tensor::zeros({out_features});
    }
}

Tensor Linear::forward(const Tensor& x) {
    std::vector<int> batch_shape;
    for (size_t i = 0; i < x.shape.size() - 1; i++)
        batch_shape.push_back(x.shape[i]);
    int batch_size = 1;
    for (int dim : batch_shape) batch_size *= dim;
    if (x.shape.back() != in_features)
        throw std::runtime_error("Input last dimension mismatch with Linear in_features");

    Tensor x_2d = x.reshape({batch_size, in_features});
    Tensor result_2d({batch_size, out_features});
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += x_2d.data[b * in_features + i] * weight.data[i * out_features + o];
            }
            if (use_bias) sum += bias.data[o];
            result_2d.data[b * out_features + o] = sum;
        }
    }

    batch_shape.push_back(out_features);
    return result_2d.reshape(batch_shape);
}

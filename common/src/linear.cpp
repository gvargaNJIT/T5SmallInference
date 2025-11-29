#include "linear.hpp"
#include <cmath>
#include <stdexcept>

Linear::Linear(int in_feat, int out_feat)
    : in_features(in_feat), out_features(out_feat) {
    weight = Tensor({out_feat, in_feat}, 0.f);
}

Tensor Linear::forward(const Tensor& x) {
    std::vector<int> batch_shape = x.shape;
    batch_shape.pop_back();

    int batch_size = 1;
    for (int dim : batch_shape) {
        batch_size *= dim;
    }

    std::vector<int> shape_2d = {batch_size, in_features};
    Tensor x_2d = x.reshape(shape_2d);
    Tensor result_2d = x_2d.matmul(weight);
    
    std::vector<int> output_shape = batch_shape;
    output_shape.push_back(out_features);
    
    return result_2d.reshape(output_shape);
}
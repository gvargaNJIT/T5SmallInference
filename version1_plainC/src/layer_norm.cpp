#include "layer_norm.hpp"
#include <cmath>
#include <stdexcept>

namespace t5 {

LayerNorm::LayerNorm(size_t hidden_size, float eps)
    : hidden_size_(hidden_size)
    , eps_(eps)
    , weight_({hidden_size}, 1.0f) 
{}

void LayerNorm::set_weight(const Tensor& weight) {
    if (weight.size() != hidden_size_) {
        throw std::runtime_error("LayerNorm: weight size mismatch");
    }
    weight_.copy_from(weight);
}

void LayerNorm::forward(const Tensor& input, Tensor& output) {
    auto input_shape = input.shape();
    size_t outer_dim = 1;
    size_t inner_dim = hidden_size_;
    
    if (input_shape.size() == 3) {
        outer_dim = input_shape[0] * input_shape[1];
        inner_dim = input_shape[2];
    } else if (input_shape.size() == 2) {
        outer_dim = input_shape[0];
        inner_dim = input_shape[1];
    } else {
        throw std::runtime_error("LayerNorm: input must be 2D or 3D");
    }
    if (inner_dim != hidden_size_) throw std::runtime_error("LayerNorm: input hidden_size mismatch");
    if (output.shape() != input_shape) output = Tensor(input_shape);
    
    const float* in_data = input.data();
    float* out_data = output.data();
    const float* weight_data = weight_.data();

    for (size_t i = 0; i < outer_dim; i++) {
        const float* in_vec = in_data + i * inner_dim;
        float* out_vec = out_data + i * inner_dim;
        float mean_square = 0.0f;
        for (size_t j = 0; j < inner_dim; j++) {
            mean_square += in_vec[j] * in_vec[j];
        }
        mean_square /= inner_dim;

        float inv_rms = 1.0f / std::sqrt(mean_square + eps_);

        for (size_t j = 0; j < inner_dim; j++) {
            out_vec[j] = in_vec[j] * inv_rms * weight_data[j];
        }
    }
}
}

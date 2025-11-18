#ifndef T5_LAYER_NORM_HPP
#define T5_LAYER_NORM_HPP

#include "tensor.hpp"
#include <cstddef>

namespace t5 {

class LayerNorm {
public:
    LayerNorm(size_t hidden_size, float eps = 1e-6f);
    void forward(const Tensor& input, Tensor& output);
    void set_weight(const Tensor& weight);
    const Tensor& weight() const { return weight_; }
private:
    size_t hidden_size_;
    float eps_;
    Tensor weight_;
};
}
#endif
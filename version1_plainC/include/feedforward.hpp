#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "tensor.hpp"
#include "config.hpp"

namespace t5 {
namespace serial {

class FeedForward {
public:
    FeedForward(const Config& config);
    void forward(const Tensor& input, Tensor& output);
    void set_wi_weight(const Tensor& weight);
    void set_wo_weight(const Tensor& weight);
    
private:
    size_t d_model_;
    size_t d_ff_;
    
    Tensor wi_weight_;
    Tensor wo_weight_;
};
}
}

#endif
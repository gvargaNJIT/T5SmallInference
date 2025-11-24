#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "config.hpp"
#include "linear.hpp"
#include "tensor.hpp"

class FeedForward {
public:
    Linear wi;
    Linear wo;
    FeedForward(const T5Config& config);
    Tensor forward(const Tensor& x);
};

#endif

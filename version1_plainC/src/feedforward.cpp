#include "feedforward.hpp"

FeedForward::FeedForward(const T5Config& config)
    : wi(config.d_model, config.d_ff, false),
      wo(config.d_ff, config.d_model, false)
{
}

Tensor FeedForward::forward(const Tensor& x) {
    Tensor h = wi.forward(x);
    h = activation::relu(h);
    h = wo.forward(h);
    return h;
}

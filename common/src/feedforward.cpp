#include "feedforward.hpp"

FeedForward::FeedForward()
    : wi(T5Config::d_model, T5Config::d_ff),
      wo(T5Config::d_ff, T5Config::d_model)
{
}

Tensor FeedForward::forward(const Tensor& x) {
    
    Tensor h = wi.forward(x);
    h = activation::relu(h);
    h = wo.forward(h);
    return h;
}

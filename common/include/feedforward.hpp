#pragma once

#include "config.hpp"
#include "linear.hpp"
#include "tensor.hpp"

class FeedForward
{
public:
    Linear wi;
    Linear wo;
    FeedForward();
    Tensor forward(const Tensor &x);
};

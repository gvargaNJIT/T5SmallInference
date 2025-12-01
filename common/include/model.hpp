#pragma once

#include "config.hpp"
#include "stack.hpp"
#include "linear.hpp"
#include "tensor.hpp"
#include <vector>

class T5Model
{
public:
    T5Stack encoder;
    T5Stack decoder;
    Linear lm_head;

    T5Model();

    Tensor forward(const Tensor &input_ids, const Tensor &decoder_input_ids);

    std::vector<int> generate(const Tensor &input_ids, int max_length = 50);
};

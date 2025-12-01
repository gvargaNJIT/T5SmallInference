#pragma once

#include "config.hpp"
#include "embedding.hpp"
#include "block.hpp"
#include "rms_norm.hpp"
#include "tensor.hpp"
#include <vector>

class T5Stack {
public:
    Embedding embed;
    std::vector<T5Block> blocks;
    RMSNorm final_rms_norm;

    T5Stack(bool is_decoder = false);

    Tensor forward(const Tensor& input_ids,
                   const Tensor* encoder_hidden_states = nullptr);
};

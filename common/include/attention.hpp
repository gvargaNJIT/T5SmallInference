#pragma once

#include "linear.hpp"
#include "tensor.hpp"
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cmath>

class MultiHeadAttention
{
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;

    Tensor relative_attention_bias;

    int d_model;
    int n_heads;
    int d_kv;
    int inner_dim;
    bool has_relative_bias;
    bool is_decoder;
    int num_buckets;
    int max_distance;

    MultiHeadAttention(bool has_bias = false, bool is_decoder = false);

    void compute_relative_position_bias(Tensor &bias, int q_len, int k_len);

    int relative_position_to_bucket(int relative_position);

    std::pair<Tensor, Tensor> forward(
        const Tensor &hidden_states,
        const Tensor *key_value_states = nullptr,
        const Tensor *position_bias = nullptr);
};

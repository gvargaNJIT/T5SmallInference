#pragma once

#include "rms_norm.hpp"
#include "attention.hpp"
#include "feedforward.hpp"
#include "tensor.hpp"
#include <memory>
#include <utility>

class T5Block {
public:
    RMSNorm rms_norm_before_self_attention;

    MultiHeadAttention self_attention_layer;

    RMSNorm rms_norm_before_cross_attention;

    MultiHeadAttention cross_attention_layer;

    RMSNorm rms_norm_before_feedforward;

    FeedForward feedforward_layer;

    bool is_decoder_block;

    T5Block(bool has_relative_bias = false,bool is_decoder = false);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor* position_bias = nullptr,
                                      const Tensor* encoder_hidden_states = nullptr);
};

#include "block.hpp"

T5Block::T5Block(bool has_relative_bias, bool is_decoder)
    : rms_norm_before_self_attention(T5Config::d_model, T5Config::rms_norm_epsilon),
      self_attention_layer(has_relative_bias, is_decoder),
      rms_norm_before_cross_attention(T5Config::d_model, T5Config::rms_norm_epsilon),
      cross_attention_layer(false, false),
      rms_norm_before_feedforward(T5Config::d_model, T5Config::rms_norm_epsilon),
      feedforward_layer(),
      is_decoder_block(is_decoder)
{
}

std::pair<Tensor, Tensor> T5Block::forward(
    const Tensor &hidden_states,
    const Tensor *position_bias,
    const Tensor *encoder_hidden_states)
{
    Tensor x_norm = rms_norm_before_self_attention.forward(hidden_states);

    auto [self_attention_output, new_position_bias] =
        self_attention_layer.forward(x_norm, nullptr, position_bias);

    Tensor x = hidden_states + self_attention_output;

    if (is_decoder_block && encoder_hidden_states)
    {
        x_norm = rms_norm_before_cross_attention.forward(x);

        auto [cross_attention_output, _] =
            cross_attention_layer.forward(x_norm, encoder_hidden_states);

        x = x + cross_attention_output;
    }

    x_norm = rms_norm_before_feedforward.forward(x);
    x = x + feedforward_layer.forward(x_norm);

    return {x, new_position_bias};
}

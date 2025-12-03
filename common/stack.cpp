#include "stack.hpp"

T5Stack::T5Stack(bool is_decoder)
    : embed(T5Config::vocab_size, T5Config::d_model),
      final_rms_norm(T5Config::d_model, T5Config::rms_norm_epsilon)
{
    for (int i = 0; i < T5Config::num_layers; i++) {
        T5Block tmp = T5Block(i==0,is_decoder);
        blocks.push_back(tmp);
    }
}

Tensor T5Stack::forward(
        const Tensor& input_ids,
        const Tensor* encoder_hidden_states)
{
    Tensor hidden = embed.forward(input_ids);

    Tensor position_bias;
    bool has_position_bias = false;

    for (auto& block : blocks)
    {
        if (!has_position_bias)
        {
            auto [new_hidden, new_bias] =
                block.forward(hidden, nullptr, encoder_hidden_states);

            hidden = new_hidden;
            position_bias = new_bias;
            has_position_bias = true;
        }
        else
        {
            auto [new_hidden, _] =
                block.forward(hidden, &position_bias, encoder_hidden_states);

            hidden = new_hidden;
        }
    }

    hidden = final_rms_norm.forward(hidden);

    return hidden;
}

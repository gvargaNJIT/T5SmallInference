#include "model.hpp"
#include <algorithm>

T5Model::T5Model()
    : encoder(),
      decoder(true),
      lm_head(T5Config::d_model, T5Config::vocab_size)
{
    decoder.embed.weight = encoder.embed.weight;
    lm_head.weight       = encoder.embed.weight;
}

Tensor T5Model::forward(const Tensor &input_ids, const Tensor &decoder_input_ids)
{
    Tensor encoder_output = encoder.forward(input_ids);

    Tensor decoder_output = decoder.forward(decoder_input_ids, &encoder_output);

    Tensor logits = lm_head.forward(decoder_output);

    return logits;
}

std::vector<int> T5Model::generate(const Tensor &input_ids, int max_length)
{
    Tensor encoder_output = encoder.forward(input_ids);

    std::vector<int> generated = { T5Config::pad_token_id };

    for (int step = 0; step < max_length; step++)
    {
        Tensor decoder_input({ static_cast<int>(generated.size()) });
        for (size_t i = 0; i < generated.size(); i++) {
            decoder_input.data[i] = static_cast<float>(generated[i]);
        }

        Tensor decoder_output = decoder.forward(decoder_input, &encoder_output);  
        Tensor logits = lm_head.forward(decoder_output);

        int vocab_size = logits.shape[1];
        int last_pos   = generated.size() - 1;

        std::vector<float> last_logits(vocab_size);
        for (int v = 0; v < vocab_size; v++) {
            last_logits[v] = logits.data[last_pos * vocab_size + v];
        }

        int next_token = std::max_element(last_logits.begin(), last_logits.end())
                         - last_logits.begin();

        if (next_token == T5Config::eos_token_id)
            break;

        generated.push_back(next_token);
    }

    return generated;
}

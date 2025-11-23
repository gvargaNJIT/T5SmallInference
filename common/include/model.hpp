#ifndef MODEL_HPP
#define MODEL_HPP

#include "config.hpp"
#include "stack.hpp"
#include "linear.hpp"
#include "tensor.hpp"
#include <vector>
#include <algorithm>

// Complete T5 Model
class T5Model {
public:
    T5Config encoder_config;
    T5Config decoder_config;
    T5Stack encoder;
    T5Stack decoder;
    Linear lm_head;

    T5Model(const T5Config& config)
        : encoder_config(config),
          decoder_config(config),
          encoder(encoder_config),
          decoder([&config]() {
              T5Config dec_config = config;
              dec_config.is_decoder = true;
              return dec_config;
          }()),
          lm_head(config.d_model, config.vocab_size, false) {

        // Share embeddings between encoder, decoder, and lm_head
        decoder.embed.weight = encoder.embed.weight;
        lm_head.weight = encoder.embed.weight;
    }

    Tensor forward(const Tensor& input_ids, const Tensor& decoder_input_ids) {
        // Encode
        Tensor encoder_output = encoder.forward(input_ids);

        // Decode
        Tensor decoder_output = decoder.forward(decoder_input_ids, &encoder_output);

        // Project to vocabulary
        Tensor logits = lm_head.forward(decoder_output);

        return logits;
    }

    std::vector<int> generate(const Tensor& input_ids, int max_length = 50) {
        // Encode once
        Tensor encoder_output = encoder.forward(input_ids);

        // Start with pad token
        std::vector<int> generated = {decoder_config.pad_token_id};

        for (int step = 0; step < max_length; step++) {
            // Create decoder input tensor
            Tensor decoder_input({1, static_cast<int>(generated.size())});
            for (size_t i = 0; i < generated.size(); i++) {
                decoder_input.data[i] = static_cast<float>(generated[i]);
            }

            // Forward pass
            Tensor decoder_output = decoder.forward(decoder_input, &encoder_output);
            Tensor logits = lm_head.forward(decoder_output);

            // Get last token logits: [1, seq_len, vocab] -> [vocab]
            int vocab_size = logits.shape[2];
            int last_pos = generated.size() - 1;
            std::vector<float> last_logits(vocab_size);
            for (int v = 0; v < vocab_size; v++) {
                last_logits[v] = logits.data[last_pos * vocab_size + v];
            }

            // Greedy decode
            int next_token = std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin();

            if (next_token == decoder_config.eos_token_id) {
                break;
            }

            generated.push_back(next_token);
        }

        return generated;
    }
};

#endif // MODEL_HPP
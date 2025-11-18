#include "serial_model.hpp"
#include "matrix_ops.hpp"
#include <iostream>
#include <algorithm>

namespace t5 {
namespace serial {

T5Model::T5Model(const Config& config)
    : config_(config)
{
    shared_embedding_ = std::make_unique<Embedding>(config);
    encoder_ = std::make_unique<T5Stack>(config, false, shared_embedding_.get());
    decoder_ = std::make_unique<T5Stack>(config, true, shared_embedding_.get());
    lm_head_weight_ = Tensor({config.vocab_size, config.d_model});
}

void T5Model::load_weights(const std::string& weights_dir) {
    std::cout << "Loading T5 model weights...\n";
    auto weights = WeightLoader::load_weights(weights_dir);
    if (weights.empty()) {
        throw std::runtime_error("Failed to load weights");
    }
    if (weights.find("shared.weight") != weights.end()) {
        shared_embedding_->set_weight(weights["shared.weight"]);
        lm_head_weight_.copy_from(weights["shared.weight"]);
        std::cout << "Loaded shared embedding\n";
    }
    encoder_->load_weights(weights, "encoder");
    decoder_->load_weights(weights, "decoder");
    std::cout << "\nModel loaded successfully!\n";
    std::cout << "Total parameters: " << count_parameters() << "\n";
}

void T5Model::forward(
    const std::vector<int>& input_ids,
    const std::vector<int>& decoder_input_ids,
    Tensor& output
) {
    Tensor encoder_output;
    encoder_->forward(input_ids, encoder_output);
    Tensor decoder_output;
    decoder_->forward(decoder_input_ids, decoder_output, &encoder_output);
    size_t seq_len = decoder_output.shape()[0];
    size_t d_model = decoder_output.shape()[1];
    size_t vocab_size = lm_head_weight_.shape()[0];
    output = Tensor({seq_len, vocab_size});
    matmul_transposed(
        decoder_output.data(),
        lm_head_weight_.data(),
        output.data(),
        seq_len,
        d_model,
        vocab_size
    );
}

std::vector<int> T5Model::generate(
    const std::vector<int>& input_ids,
    size_t max_length
) {
    std::cout << "Generating with greedy decoding...\n";
    Tensor encoder_output;
    encoder_->forward(input_ids, encoder_output);
    std::vector<int> generated_ids = {config_.pad_token_id};
    for (size_t step = 0; step < max_length; step++) {
        Tensor decoder_output;
        decoder_->forward(generated_ids, decoder_output, &encoder_output);
        size_t seq_len = decoder_output.shape()[0];
        size_t d_model = decoder_output.shape()[1];
        size_t vocab_size = lm_head_weight_.shape()[0];
        
        Tensor logits({seq_len, vocab_size});
        matmul_transposed(
            decoder_output.data(),
            lm_head_weight_.data(),
            logits.data(),
            seq_len,
            d_model,
            vocab_size
        );
        const float* last_logits = logits.data() + (seq_len - 1) * vocab_size;
        int next_token = 0;
        float max_logit = last_logits[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (last_logits[v] > max_logit) {
                max_logit = last_logits[v];
                next_token = v;
            }
        }
        if (next_token == config_.eos_token_id) {
            break;
        }
        generated_ids.push_back(next_token);
        if ((step + 1) % 10 == 0) {
            std::cout << "  Generated " << (step + 1) << " tokens...\n";
        }
    }
    
    return generated_ids;
}

size_t T5Model::count_parameters() const {
    size_t total = 0;
    total += shared_embedding_->weight().size();
    return total;
}
}
}
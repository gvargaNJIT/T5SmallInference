/*
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
    
    // DEBUG: Check encoder output
    std::cout << "DEBUG: Encoder output shape: [";
    for (size_t i = 0; i < encoder_output.shape().size(); i++) {
        std::cout << encoder_output.shape()[i];
        if (i < encoder_output.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "DEBUG: First 5 encoder values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << encoder_output[i] << " ";
    }
    std::cout << std::endl;
    
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
        
        // DEBUG: Print logits for first step
        if (step == 0) {
            std::cout << "\nDEBUG: First 10 logits at step 0: ";
            for (int i = 0; i < 10; i++) {
                std::cout << last_logits[i] << " ";
            }
            std::cout << std::endl;
            
            // Find top 5 tokens
            std::vector<std::pair<int, float>> top_tokens;
            for (size_t v = 0; v < vocab_size; v++) {
                top_tokens.push_back({v, last_logits[v]});
            }
            std::sort(top_tokens.begin(), top_tokens.end(), 
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            std::cout << "DEBUG: Top 5 tokens at step 0:" << std::endl;
            for (int i = 0; i < 5; i++) {
                std::cout << "  Token " << top_tokens[i].first << ": " << top_tokens[i].second << std::endl;
            }
        }
        
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

}
}
*/
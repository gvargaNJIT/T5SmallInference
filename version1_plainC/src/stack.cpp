#include "stack.hpp"
#include <iostream>

namespace t5 {
namespace serial {

T5Stack::T5Stack(
    const Config& config, 
    bool is_decoder,
    Embedding* shared_embedding
)
    : is_decoder_(is_decoder)
    , num_layers_(is_decoder ? config.num_decoder_layers : config.num_layers)
    , embedding_(shared_embedding)
{
    for (size_t i = 0; i < num_layers_; i++) {
        bool has_cross_attention = is_decoder;
        blocks_.push_back(
            std::make_unique<TransformerBlock>(config, is_decoder, has_cross_attention)
        );
    }
    final_layer_norm_ = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_epsilon);
}

void T5Stack::forward(
    const std::vector<int>& input_ids,
    Tensor& output,
    const Tensor* encoder_hidden_states
) {
    Tensor hidden_states;
    embedding_->forward(input_ids, hidden_states);
    for (size_t i = 0; i < blocks_.size(); i++) {
        Tensor block_output;
        blocks_[i]->forward(hidden_states, block_output, encoder_hidden_states);
        hidden_states = block_output;
    }
    final_layer_norm_->forward(hidden_states, output);
}

void T5Stack::load_weights(
    const std::unordered_map<std::string, Tensor>& weights,
    const std::string& prefix
) {
    std::cout << "\nLoading weights for " << prefix << std::endl;
    for (size_t i = 0; i < blocks_.size(); i++) {
        std::string block_prefix = prefix + ".block." + std::to_string(i);
        blocks_[i]->load_weights(weights, block_prefix);
    }

    std::string final_ln_key = prefix + ".final.layer.norm.weight";
    if (weights.find(final_ln_key) != weights.end()) {
        final_layer_norm_->set_weight(weights.at(final_ln_key));
    }
}
}
}
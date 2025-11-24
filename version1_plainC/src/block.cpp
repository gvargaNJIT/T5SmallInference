/*
#include "block.hpp"
#include <iostream>

namespace t5 {
namespace serial {

TransformerBlock::TransformerBlock(
    const Config& config, 
    bool is_decoder, 
    bool has_cross_attention
)
    : is_decoder_(is_decoder)
    , has_cross_attention_(has_cross_attention)
{
    self_attn_norm_ = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_epsilon);
    self_attention_ = std::make_unique<Attention>(config, is_decoder);
    if (has_cross_attention) {
        cross_attn_norm_ = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_epsilon);
        cross_attention_ = std::make_unique<Attention>(config, false);
    }
    ff_norm_ = std::make_unique<LayerNorm>(config.d_model, config.layer_norm_epsilon);
    feedforward_ = std::make_unique<FeedForward>(config);
}

void TransformerBlock::forward(
    const Tensor& hidden_states,
    Tensor& output,
    const Tensor* encoder_hidden_states
) {
    Tensor current = hidden_states;
    {
        Tensor normed;
        self_attn_norm_->forward(current, normed);
        Tensor attn_output;
        self_attention_->forward(normed, attn_output);
        output = Tensor(current.shape());
        for (size_t i = 0; i < current.size(); i++) {
            output[i] = current[i] + attn_output[i];
        }
        current = output;
    }
    if (has_cross_attention_ && encoder_hidden_states != nullptr) {
        Tensor normed;
        cross_attn_norm_->forward(current, normed);
        Tensor attn_output;
        cross_attention_->forward(normed, attn_output, encoder_hidden_states);
        for (size_t i = 0; i < current.size(); i++) {
            current[i] = current[i] + attn_output[i];
        }
    }
    {
        Tensor normed;
        ff_norm_->forward(current, normed);
        Tensor ff_output;
        feedforward_->forward(normed, ff_output);
        output = Tensor(current.shape());
        for (size_t i = 0; i < current.size(); i++) {
            output[i] = current[i] + ff_output[i];
        }
    }
}

void TransformerBlock::load_weights(
    const std::unordered_map<std::string, Tensor>& weights,
    const std::string& prefix
) {
    std::cout << "Loading weights for " << prefix << std::endl;
    std::string ln0_key = prefix + ".layer.0.layer_norm.weight";
    if (weights.find(ln0_key) != weights.end()) {
        self_attn_norm_->set_weight(weights.at(ln0_key));
    }
    self_attention_->load_weights(weights, prefix + ".layer.0.SelfAttention");
    if (has_cross_attention_) {
        std::string ln1_key = prefix + ".layer.1.layer_norm.weight";
        if (weights.find(ln1_key) != weights.end()) {
            cross_attn_norm_->set_weight(weights.at(ln1_key));
        }
        cross_attention_->load_weights(weights, prefix + ".layer.1.EncDecAttention");
    }
    int ff_layer_idx = has_cross_attention_ ? 2 : 1;
    std::string ff_ln_key = prefix + ".layer." + std::to_string(ff_layer_idx) + ".layer_norm.weight";
    if (weights.find(ff_ln_key) != weights.end()) {
        ff_norm_->set_weight(weights.at(ff_ln_key));
    }
    std::string ff_prefix = prefix + ".layer." + std::to_string(ff_layer_idx) + ".DenseReluDense";
    std::string wi_key = ff_prefix + ".wi.weight";
    std::string wo_key = ff_prefix + ".wo.weight";
    
    if (weights.find(wi_key) != weights.end()) {
        feedforward_->set_wi_weight(weights.at(wi_key));
    }
    if (weights.find(wo_key) != weights.end()) {
        feedforward_->set_wo_weight(weights.at(wo_key));
    }
}
}
}
*/
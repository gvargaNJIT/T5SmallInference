/*
#ifndef BLOCK_HPP
#define BLOCK_HPP

#include "tensor.hpp"
#include "config.hpp"
#include "layer_norm.hpp"
#include "feedforward.hpp"
#include "attention.hpp"
#include <unordered_map>
#include <string>

namespace t5 {
namespace serial {

class TransformerBlock {
public:
    TransformerBlock(const Config& config, bool is_decoder, bool has_cross_attention);
    void forward(
        const Tensor& hidden_states,
        Tensor& output,
        const Tensor* encoder_hidden_states = nullptr
    );

    void load_weights(
        const std::unordered_map<std::string, Tensor>& weights,
        const std::string& prefix
    );

    Attention* get_self_attention() { return self_attention_.get(); }
    
private:
    bool is_decoder_;
    bool has_cross_attention_;
    std::unique_ptr<LayerNorm> self_attn_norm_;
    std::unique_ptr<Attention> self_attention_;
    std::unique_ptr<LayerNorm> cross_attn_norm_;
    std::unique_ptr<Attention> cross_attention_;
    std::unique_ptr<LayerNorm> ff_norm_;
    std::unique_ptr<FeedForward> feedforward_;
};
}
}

#endif
*/
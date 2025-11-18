#ifndef STACK_HPP
#define STACK_HPP

#include "tensor.hpp"
#include "config.hpp"
#include "embedding.hpp"
#include "block.hpp"
#include "layer_norm.hpp"
#include <vector>
#include <memory>

namespace t5 {
namespace serial {

class T5Stack {
public:
    T5Stack(const Config& config, bool is_decoder, Embedding* shared_embedding);
    void forward(
        const std::vector<int>& input_ids,
        Tensor& output,
        const Tensor* encoder_hidden_states = nullptr
    );

    void load_weights(
        const std::unordered_map<std::string, Tensor>& weights,
        const std::string& prefix
    );
    
private:
    bool is_decoder_;
    size_t num_layers_;
    Embedding* embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    std::unique_ptr<LayerNorm> final_layer_norm_;
};
}
}

#endif
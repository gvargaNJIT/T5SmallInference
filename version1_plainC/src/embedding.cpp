#include "embedding.hpp"
#include <stdexcept>
#include <cstring>

namespace t5 {
namespace serial {

Embedding::Embedding(const Config& config)
    : vocab_size_(config.vocab_size)
    , d_model_(config.d_model)
    , weight_({vocab_size_, d_model_})
{
}

void Embedding::set_weight(const Tensor& weight) {
    if (weight.shape().size() != 2 || 
        weight.shape()[0] != vocab_size_ || 
        weight.shape()[1] != d_model_) {
        throw std::runtime_error("Embedding: weight shape mismatch");
    }
    weight_.copy_from(weight);
}

void Embedding::forward(const std::vector<int>& token_ids, Tensor& output) {
    size_t seq_len = token_ids.size();
    output = Tensor({seq_len, d_model_});
    const float* weight_data = weight_.data();
    float* output_data = output.data();
    for (size_t i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= static_cast<int>(vocab_size_)) {
            throw std::runtime_error("Embedding: token_id out of bounds");
        }

        const float* embedding = weight_data + token_id * d_model_;
        float* output_pos = output_data + i * d_model_;
        std::memcpy(output_pos, embedding, d_model_ * sizeof(float));
    }
}
}
}
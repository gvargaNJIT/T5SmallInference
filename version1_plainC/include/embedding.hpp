#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "tensor.hpp"
#include "config.hpp"
#include <vector>

namespace t5 {
namespace serial {

class Embedding {
public:
    Embedding(const Config& config);
    void forward(const std::vector<int>& token_ids, Tensor& output);
    void set_weight(const Tensor& weight);
    const Tensor& weight() const { return weight_; }
    
private:
    size_t vocab_size_;
    size_t d_model_;
    
    Tensor weight_;
};
}
}

#endif
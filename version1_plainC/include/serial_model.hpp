/*
#ifndef SERIAL_MODEL_HPP
#define SERIAL_MODEL_HPP

#include "tensor.hpp"
#include "config.hpp"
#include "embedding.hpp"
#include "stack.hpp"
#include "weight_loader.hpp"
#include <vector>
#include <memory>

namespace t5 {
namespace serial {

class T5Model {
public:
    explicit T5Model(const Config& config);
    void load_weights(const std::string& weights_dir);
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        size_t max_length = 50
    );

    void forward(
        const std::vector<int>& input_ids,
        const std::vector<int>& decoder_input_ids,
        Tensor& output
    );
    
private:
    Config config_;
    std::unique_ptr<Embedding> shared_embedding_;
    std::unique_ptr<T5Stack> encoder_;
    std::unique_ptr<T5Stack> decoder_;
    Tensor lm_head_weight_;
};
}
}

#endif
*/
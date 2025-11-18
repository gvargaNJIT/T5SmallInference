#ifndef T5_CONFIG_HPP
#define T5_CONFIG_HPP

#include <string>
#include <cstddef>

namespace t5 {

struct Config {
    size_t d_model = 512;
    size_t d_kv = 64;
    size_t d_ff = 2048;
    size_t num_layers = 6;
    size_t num_decoder_layers = 6; 
    size_t num_heads = 8;
    size_t relative_attention_num_buckets = 32;
    size_t relative_attention_max_distance = 128;
    float dropout_rate = 0.1f;
    float layer_norm_epsilon = 1e-6f;
    size_t vocab_size = 32128;
    int pad_token_id = 0;
    int eos_token_id = 1;
    
    static Config load(const std::string& filepath);

    void print() const;
};

}
#endif 
#ifndef CONFIG_HPP
#define CONFIG_HPP

struct T5Config {
    int d_model = 512;
    int d_kv = 64;
    int d_ff = 2048;
    int num_layers = 6;
    int num_heads = 8;
    int relative_attention_num_buckets = 32;
    int relative_attention_max_distance = 128;
    float dropout_rate = 0.1f;
    float layer_norm_epsilon = 1e-6f;
    int vocab_size = 32128;
    int pad_token_id = 0;
    int eos_token_id = 1;
    bool is_decoder = false;
};

#endif
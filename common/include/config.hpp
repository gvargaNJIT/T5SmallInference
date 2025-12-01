#pragma once

struct T5Config {
    static inline constexpr int d_model = 512;
    static inline constexpr int d_kv = 64;
    static inline constexpr int d_ff = 2048;
    static inline constexpr int num_layers = 6;
    static inline constexpr int num_heads = 8;
    static inline constexpr int num_buckets = 32;
    static inline constexpr int max_distance = 128;
    static inline constexpr float rms_norm_epsilon = 1e-6f;
    static inline constexpr int vocab_size = 32128;
    static inline constexpr int pad_token_id = 0;
    static inline constexpr int eos_token_id = 1;
};

def get_t5_config():
    return {
        'd_ff': 2048,
        'd_model': 512,
        'd_kv': 64,
        'num_layers': 6,
        'num_heads': 8,
        'vocab_size': 32128,
        'relative_attention_num_buckets': 32,
        'dropout_rate': 0.1,
        'layer_norm_epsilon': 1e-6,
    }

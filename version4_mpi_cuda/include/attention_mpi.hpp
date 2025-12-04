#pragma once
#include "tensor.hpp"


int relative_position_to_bucket(int relative_position, bool is_decoder, int num_buckets);

void compute_relative_position_bias(
    Tensor& bias, const Tensor& relative_attention_bias,
    int query_len, int key_len, int n_heads, bool is_decoder, int num_buckets);

Tensor compute_attention_parallel(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& bias,
    int n_heads, int seq_len, int k_len, int d_kv);


Tensor serial_matmul(const Tensor& a, const Tensor& b);

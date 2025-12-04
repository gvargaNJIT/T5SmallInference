#include <algorithm>
#include <cmath>
#include "tensor.hpp"
#include "config.hpp"
#include "attention.hpp"


extern "C" void compute_relative_position_bias_cuda(
    Tensor& bias,
    const Tensor& relative_attention_bias,
    int query_len,
    int key_len,
    int n_heads,
    bool is_decoder,
    int num_buckets,
    int max_distance);

void MultiHeadAttention::compute_relative_position_bias(Tensor &bias, int query_len, int key_len)
{
    if (!has_relative_bias)
    {
        bias = Tensor({bias.size()});
        return;
    }

    compute_relative_position_bias_cuda(
        bias,
        relative_attention_bias,
        query_len,
        key_len,
        n_heads,
        is_decoder,
        T5Config::num_buckets,
        T5Config::max_distance);
}
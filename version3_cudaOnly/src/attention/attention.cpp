#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "config.hpp"

// CUDA kernel declaration
extern "C" void cuda_compute_relative_position_bias(
    Tensor& bias,
    const Tensor& relative_attention_bias,
    int query_len,
    int key_len,
    int n_heads,
    bool is_decoder,
    int num_buckets,
    int max_distance);

MultiHeadAttention::MultiHeadAttention(bool has_bias, bool is_decoder)
    : q_proj(T5Config::d_model, T5Config::num_heads * T5Config::d_kv),
      k_proj(T5Config::d_model, T5Config::num_heads * T5Config::d_kv),
      v_proj(T5Config::d_model, T5Config::num_heads * T5Config::d_kv),
      o_proj(T5Config::num_heads * T5Config::d_kv, T5Config::d_model),
      d_model(T5Config::d_model),
      n_heads(T5Config::num_heads),
      d_kv(T5Config::d_kv),
      inner_dim(T5Config::num_heads * T5Config::d_kv),
      has_relative_bias(has_bias),
      is_decoder(is_decoder)
{
    if (has_relative_bias)
        relative_attention_bias = Tensor({T5Config::num_buckets, n_heads});
}

void MultiHeadAttention::compute_relative_position_bias(
    Tensor &bias, int query_len, int key_len)
{
    if (!has_relative_bias)
    {
        bias = Tensor({bias.size()});
        return;
    }

    cuda_compute_relative_position_bias(
        bias, 
        relative_attention_bias, 
        query_len, 
        key_len, 
        n_heads,
        is_decoder,
        T5Config::num_buckets,
        T5Config::max_distance);
}
std::pair<Tensor, Tensor> MultiHeadAttention::forward(
    const Tensor &hidden_states,
    const Tensor *key_value_states,
    const Tensor *position_bias)
{

    int seq_len = hidden_states.shape[0];

    Tensor q = q_proj.forward(hidden_states);

    // key_value_states are used for cross-attention (decoder)
    // hidden_states are used for self-attention (encoder/decoder)
    const Tensor &kv_in = key_value_states ? *key_value_states : hidden_states;
    Tensor k = k_proj.forward(kv_in);
    Tensor v = v_proj.forward(kv_in);

    int k_len = kv_in.shape[0];

    // [Seq, Heads*Dim] -> [Heads, Seq, Dim]
    q = q.reshape({seq_len, n_heads, d_kv}).permute({1, 0, 2});
    k = k.reshape({k_len, n_heads, d_kv}).permute({1, 0, 2});
    v = v.reshape({k_len, n_heads, d_kv}).permute({1, 0, 2});

    Tensor bias({n_heads, seq_len, k_len});
    if (position_bias)
    {
        bias = *position_bias;
    }
    else
    {
        compute_relative_position_bias(bias, seq_len, k_len);
    }

    Tensor context_layer({n_heads, seq_len, d_kv});

    int q_head_size = seq_len * d_kv;
    int k_head_size = k_len * d_kv;
    int bias_head_size = seq_len * k_len;

    for (int h = 0; h < n_heads; ++h)
    {

        Tensor q_h({seq_len, d_kv});
        memcpy(q_h.data.data(),
               q.data.data() + h * q_head_size,
               q_head_size * sizeof(float));

        Tensor k_h({k_len, d_kv});
        memcpy(k_h.data.data(),
               k.data.data() + h * k_head_size,
               k_head_size * sizeof(float));

        Tensor v_h({k_len, d_kv});
        memcpy(v_h.data.data(),
               v.data.data() + h * k_head_size,
               k_head_size * sizeof(float));

        // Attention(Q,K,V) = Softmax((Q * K^T ) + B)*V

        Tensor scores = q_h.matmul(k_h.transpose());

        int bias_offset = h * bias_head_size;

        for (int i = 0; i < bias_head_size; ++i)
        {
            scores.data[i] += bias.data[bias_offset + i];
        }

        scores = scores.softmax();

        Tensor head_out = scores.matmul(v_h);

        memcpy(context_layer.data.data() + h * q_head_size,
               head_out.data.data(),
               q_head_size * sizeof(float));
    }

    // [n_heads, seq, d_kv] -> [seq, n_heads, d_kv] -> [seq, inner_dim]
    context_layer = context_layer.permute({1, 0, 2})
                        .reshape({seq_len, inner_dim});

    Tensor output = o_proj.forward(context_layer);

    return {output, bias};
}
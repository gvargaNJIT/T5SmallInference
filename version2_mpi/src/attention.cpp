#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "config.hpp"
#include "attention_mpi.hpp"

MultiHeadAttention::MultiHeadAttention(bool has_bias,bool is_decoder)
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

int MultiHeadAttention::relative_position_to_bucket(int relative_position)
{
    return ::relative_position_to_bucket(relative_position, is_decoder, T5Config::num_buckets);
}

void MultiHeadAttention::compute_relative_position_bias(
    Tensor &bias, int query_len, int key_len)
{
    if (!has_relative_bias)
    {
        bias = Tensor({bias.size()});
        return;
    }

    ::compute_relative_position_bias(bias, relative_attention_bias, query_len, key_len, n_heads, is_decoder, T5Config::num_buckets);
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

    Tensor context_layer = compute_attention_parallel(q, k, v, bias, n_heads, seq_len, k_len, d_kv);

    // [n_heads, seq, d_kv] -> [seq, n_heads, d_kv] -> [seq, inner_dim]
    context_layer = context_layer.permute({1, 0, 2}).reshape({seq_len, inner_dim});

    Tensor output = o_proj.forward(context_layer);

    return {output, bias};
}
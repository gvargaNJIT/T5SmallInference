#include <algorithm>
#include <cmath>
#include "tensor.hpp"
#include "config.hpp"
#include "attention_mpi.hpp"

// Reference: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
// https://arxiv.org/abs/1910.10683
// https://colab.research.google.com/drive/19WgoxCSzAnzi0MueBRP6KY99Ewre1FFy?usp=sharing#scrollTo=W4zmUH7Etndk

int relative_position_to_bucket(int relative_position, bool is_decoder, int num_buckets)
{
    int number_buckets = num_buckets;
    int bucket = 0;

    // In the bidirectional Encoder, we can see both directions of our input
    //(previous words and next words), so the buckets are split.
    if (!is_decoder)
    {
        number_buckets = number_buckets / 2;

        // If key > query this is forward direction
        if (relative_position > 0)
            bucket += number_buckets;

        relative_position = std::abs(relative_position);
    }
    else
    {
        // The Decoder only uses past positions, so the relative position
        // (key - query) is always negative or zero.
        // We negate it to get a positive distance.
        relative_position = -std::min(relative_position, 0);
    }

    int max_exact = number_buckets / 2;

    // if is small -> exact buckets
    if (relative_position < max_exact)
        return bucket + relative_position;

    // if is large: log buckets
    float log_ratio = std::log((float)relative_position / max_exact);
    float log_max = std::log((float)T5Config::max_distance / max_exact);

    float pos = max_exact + (log_ratio / log_max) * (num_buckets - max_exact);
    bucket += std::min((int)pos, number_buckets - 1);

    return bucket;
}

void compute_relative_position_bias(
    Tensor& bias, const Tensor& relative_attention_bias,
    int query_len, int key_len, int n_heads, bool is_decoder, int num_buckets)
{
    for (int head = 0; head < n_heads; ++head)
    {
        for (int query = 0; query < query_len; ++query)
        {
            for (int key = 0; key < key_len; ++key)
            {
                int relative_position = key - query;
                int bucket_idx = relative_position_to_bucket(relative_position, is_decoder, num_buckets);

                int index = head * (query_len * key_len) + query * key_len + key;

                bias.data[index] = relative_attention_bias.data[bucket_idx * n_heads + head];
            }
        }
    }
}

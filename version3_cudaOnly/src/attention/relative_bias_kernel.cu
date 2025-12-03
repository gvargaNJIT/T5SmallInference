#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "tensor.hpp"
#include "config.hpp"

// Reference: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
// https://arxiv.org/abs/1910.10683
// https://colab.research.google.com/drive/19WgoxCSzAnzi0MueBRP6KY99Ewre1FFy?usp=sharing#scrollTo=W4zmUH7Etndk

__device__ int relative_position_to_bucket_device(int relative_position, bool is_decoder, int num_buckets, int max_distance)
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

        relative_position = abs(relative_position);
    }
    else
    {
        // The Decoder only uses past positions, so the relative position
        // (key - query) is always negative or zero.
        // We negate it to get a positive distance.
        relative_position = -min(relative_position, 0);
    }

    int max_exact = number_buckets / 2;

    // if is small -> exact buckets
    if (relative_position < max_exact)
        return bucket + relative_position;

    // if is large: log buckets
    float log_ratio = logf((float)relative_position / max_exact);
    float log_max = logf((float)max_distance / max_exact);

    float pos = max_exact + (log_ratio / log_max) * (num_buckets - max_exact);
    bucket += min((int)pos, number_buckets - 1);

    return bucket;
}

__global__ void compute_relative_position_bias_kernel(
    float* bias,
    const float* relative_attention_bias,
    int query_len,
    int key_len,
    int n_heads,
    bool is_decoder,
    int num_buckets,
    int max_distance)
{
    int head = blockIdx.x;
    int query = blockIdx.y;
    int key = threadIdx.x + blockIdx.z * blockDim.x;

    if (head >= n_heads || query >= query_len || key >= key_len)
        return;

    int relative_position = key - query;
    int bucket_idx = relative_position_to_bucket_device(relative_position, is_decoder, num_buckets, max_distance);

    int index = head * (query_len * key_len) + query * key_len + key;

    bias[index] = relative_attention_bias[bucket_idx * n_heads + head];
}

extern "C" void cuda_compute_relative_position_bias(
    Tensor& bias,
    const Tensor& relative_attention_bias,
    int query_len,
    int key_len,
    int n_heads,
    bool is_decoder,
    int num_buckets,
    int max_distance)
{
    float *d_bias, *d_relative_attention_bias;
    
    size_t bias_size = bias.size() * sizeof(float);
    size_t rel_bias_size = relative_attention_bias.size() * sizeof(float);

    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_relative_attention_bias, rel_bias_size);

    cudaMemcpy(d_relative_attention_bias, relative_attention_bias.data.data(), 
               rel_bias_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks(n_heads, query_len, (key_len + 255) / 256);

    compute_relative_position_bias_kernel<<<numBlocks, threadsPerBlock>>>(
        d_bias, d_relative_attention_bias, query_len, key_len, n_heads, 
        is_decoder, num_buckets, max_distance);

    cudaMemcpy(bias.data.data(), d_bias, bias_size, cudaMemcpyDeviceToHost);

    cudaFree(d_bias);
    cudaFree(d_relative_attention_bias);
}
